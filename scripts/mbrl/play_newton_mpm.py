#!/usr/bin/env python3

"""Run an ldm-quad TD-MPC checkpoint inside Newton's MPM Go2 example.

This script intentionally does not copy Newton code into this repository.  It
loads the existing ``mpm_go2_multi`` example, replaces its RSL-RL ``Go2Policy``
hook with a small TD-MPC wrapper, and lets Newton own simulation/rendering.
"""

from __future__ import annotations

import argparse
import json
from types import MethodType
import math
import os
import random
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "source" / "ldm_quad"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
LDM_QUAD_PACKAGE_ROOT = SOURCE_ROOT / "ldm_quad"
if str(LDM_QUAD_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LDM_QUAD_PACKAGE_ROOT))

import torch
import warp as wp

from mbrl import DynamicsEnsemble, LatentWorldModel, StateWorldModel, build_planner


DEFAULT_NEWTON_MPM_CANDIDATES = (
    PROJECT_ROOT.parent / "Newton_stuff" / "newton" / "examples" / "mpm" / "mpm_go2_multi",
    PROJECT_ROOT.parent / "Newton_stuff" / "mpm_go2_multi",
)


class JsonlLogger:
    def __init__(self, path: Path | None):
        self.path = path
        self._file = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", encoding="utf-8")

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def write(self, record: dict[str, Any]) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(record, sort_keys=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def quat_xyzw_to_rpy(quat_xyzw) -> tuple[float, float, float]:
    x, y, z, w = [float(value) for value in quat_xyzw]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rotate_world_to_body(vec, quat_xyzw) -> list[float]:
    x, y, z, w = [float(value) for value in quat_xyzw]
    vx, vy, vz = [float(value) for value in vec]
    # Rotate by the inverse unit quaternion: v_body = q^-1 * v_world * q.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx - w * tx + (y * tz - z * ty),
        vy - w * ty + (z * tx - x * tz),
        vz - w * tz + (x * ty - y * tx),
    ]


def tensor_list(tensor: torch.Tensor, precision: int = 5) -> list[float]:
    return [round(float(value), precision) for value in tensor.detach().cpu().view(-1)]


def parse_index_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    value = str(value)
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_newton_mpm_path(path_arg: str | None) -> Path:
    candidates: list[Path] = []
    if path_arg:
        candidates.append(Path(path_arg).expanduser())
    env_path = os.environ.get("NEWTON_MPM_GO2_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(DEFAULT_NEWTON_MPM_CANDIDATES)

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "example_mpm_go2_multi.py").is_file() and (candidate / "load_go2_policy.py").is_file():
            return candidate

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find Newton mpm_go2_multi. Pass --newton-mpm-path or set NEWTON_MPM_GO2_PATH.\n"
        f"Searched:\n  {searched}"
    )


def add_newton_paths(mpm_path: Path) -> None:
    # Current Newton examples import as ``newton.examples.mpm.mpm_go2_multi.*``,
    # which requires the directory containing ``newton/`` on sys.path.  Older
    # copies import as bare ``mpm_go2_multi.*``, which requires the mpm parent.
    path_candidates = [
        mpm_path.parents[3],
        mpm_path.parent,
        mpm_path.parent / "vendor",
        mpm_path.parents[3] / "vendor",
    ]
    for path in reversed(path_candidates):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def checkpoint_obs_dim(checkpoint: dict[str, Any], default: int = 48) -> int:
    args = checkpoint.get("args", {})
    if "obs_dim" in args:
        return int(args["obs_dim"])
    state = checkpoint.get("model", {})
    for key, value in state.items():
        if key.endswith("encoder.0.0.weight") and hasattr(value, "shape"):
            return int(value.shape[1])
        if key.endswith("members.0.net.0.weight") and hasattr(value, "shape"):
            return int(value.shape[1])
    return default


def build_tdmpc_model(checkpoint: dict[str, Any], obs_dim: int, action_dim: int, device: torch.device):
    args = checkpoint.get("args", {})
    model_type = args.get("model_type", "dynamics")
    if model_type == "latent":
        model = LatentWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=args.get("latent_dim", 128),
            hidden_dim=args["hidden_dim"],
            depth=args["model_depth"],
            num_q=args.get("num_q", 5),
            discount=args["discount"],
            tau=args.get("target_tau", 0.01),
            rho=args.get("rho", 0.5),
            entropy_coef=args.get("entropy_coef", 1e-4),
            num_bins=args.get("num_bins", 101),
            vmin=args.get("vmin", -10.0),
            vmax=args.get("vmax", 10.0),
            simnorm_dim=args.get("simnorm_dim", 8),
            q_dropout=args.get("q_dropout", 0.01),
            physical_feature_indices=parse_index_list(args.get("latent_physical_indices", "")),
        ).to(device)
    elif model_type == "state":
        model = StateWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=args["ensemble_size"],
            hidden_dim=args["hidden_dim"],
            depth=args["model_depth"],
            discount=args["discount"],
            tau=args.get("target_tau", 0.01),
            rho=args.get("rho", 0.5),
            entropy_coef=args.get("entropy_coef", 1e-4),
            num_bins=args.get("num_bins", 101),
            vmin=args.get("vmin", -10.0),
            vmax=args.get("vmax", 10.0),
            value_coef=args.get("state_value_coef", args.get("value_coef", 0.1)),
            reward_coef=args.get("reward_coef", 0.1),
            continue_coef=args.get("continue_coef", 1.0),
        ).to(device)
    else:
        model = DynamicsEnsemble(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=args["ensemble_size"],
            hidden_dim=args["hidden_dim"],
            depth=args["model_depth"],
        ).to(device)

    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if hasattr(model, "sync_detached_qs"):
        model.sync_detached_qs()
    if missing:
        print(f"[INFO] Missing {len(missing)} model keys while loading checkpoint.")
    if unexpected:
        print(f"[INFO] Ignored {len(unexpected)} unexpected model keys while loading checkpoint.")
    model.eval()
    return model, model_type


def build_tdmpc_planner(
    checkpoint: dict[str, Any],
    model,
    model_type: str,
    action_dim: int,
    device: torch.device,
    overrides: argparse.Namespace,
):
    args = checkpoint.get("args", {})
    action_low = torch.full((action_dim,), float(overrides.action_low), device=device)
    action_high = torch.full((action_dim,), float(overrides.action_high), device=device)
    return build_planner(
        planner_name=args.get("planner", "mppi"),
        model=model,
        action_low=action_low,
        action_high=action_high,
        horizon=args["horizon"],
        candidates=overrides.candidates or args["candidates"],
        elites=args.get("elites", 32),
        iterations=args.get("planner_iterations", args.get("cem_iterations", 4)),
        discount=args["discount"],
        temperature=args.get("planner_temperature", 0.5),
        lambda_=args.get("mppi_lambda", 1.0),
        min_std=overrides.min_std if overrides.min_std is not None else args.get("min_std", 0.05),
        max_std=overrides.max_std if overrides.max_std is not None else args.get("max_std", 2.0),
        num_pi_trajs=args.get("num_pi_trajs", 24),
        action_noise=False,
        use_continue_model=args.get("planner_use_continue_model", False),
        hard_continue_model=args.get("planner_hard_continue_model", False),
        continue_threshold=args.get("planner_continue_threshold", 0.5),
        action_spline_knots=args.get("action_spline_knots", 0),
        action_bounds_finite=args.get("action_bounds_finite", True),
        planner_velocity_objective_weight=(
            overrides.planner_velocity_objective_weight
            if overrides.planner_velocity_objective_weight is not None
            else args.get("planner_velocity_objective_weight", 0.0)
        ),
        planner_velocity_target_x=args.get("planner_velocity_target_x", 0.0),
        planner_velocity_target_y=args.get("planner_velocity_target_y", 0.0),
        planner_velocity_target_yaw=args.get("planner_velocity_target_yaw", 0.0),
        use_best_candidate=args.get("planner_use_best_candidate", False),
        terminal_value=model_type == "state" and args.get("state_terminal_value", True),
        disagreement_penalty=args.get("state_disagreement_penalty", 0.0) if model_type == "state" else 0.0,
        model_policy_candidate_count=args.get("num_pi_trajs", 24) if model_type == "state" else 0,
    )


def make_tdmpc_policy_class(checkpoint_path: Path, runner_args: argparse.Namespace):
    try:
        from newton.examples.robot.example_robot_go2 import compute_obs, lab_to_mujoco, mujoco_to_lab
        from newton.examples.mpm.mpm_go2_multi.load_go2_policy import ROBOT_LAB_JOINT_SWAP
    except ModuleNotFoundError:
        from mpm_go2_multi.example_robot_go2 import compute_obs, lab_to_mujoco, mujoco_to_lab
        from mpm_go2_multi.load_go2_policy import ROBOT_LAB_JOINT_SWAP

    class TDMPCGo2Policy:
        """Newton-compatible policy hook backed by ldm-quad TD-MPC."""

        def __init__(
            self,
            _policy_path: str,
            device,
            joint_pos_initial: torch.Tensor,
            action_scale: float = 0.5,
            obs_dim: int = 48,
            act_dim: int = 12,
            hidden_dims=(512, 256, 128),
            search_relative_to: Path | None = None,
        ):
            del obs_dim, hidden_dims, search_relative_to
            self.device = torch.device(device)
            self.joint_pos_initial = joint_pos_initial.to(self.device)
            self.act_dim = act_dim
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.obs_dim = checkpoint_obs_dim(checkpoint)
            if self.obs_dim not in {45, 48}:
                raise ValueError(f"Expected a 45-D or 48-D Go2 checkpoint observation, got obs_dim={self.obs_dim}.")

            self.model, self.model_type = build_tdmpc_model(checkpoint, self.obs_dim, act_dim, self.device)
            self.planner = build_tdmpc_planner(checkpoint, self.model, self.model_type, act_dim, self.device, runner_args)

            self.include_base_lin_vel = self.obs_dim >= 48
            self.ang_vel_scale = 1.0 if self.include_base_lin_vel else 0.25
            self.joint_vel_scale = 1.0 if self.include_base_lin_vel else 0.05

            identity_indices = list(range(act_dim))
            if runner_args.joint_order == "legacy_48":
                obs_indices = mujoco_to_lab
                act_indices = lab_to_mujoco
                action_scale_vec: float | list[float] = (
                    float(runner_args.action_scale) if runner_args.action_scale is not None else float(action_scale)
                )
                resolved_joint_order = "legacy_48"
            elif runner_args.joint_order == "legacy_obs_isaac_action":
                obs_indices = mujoco_to_lab
                act_indices = identity_indices
                action_scale_vec = float(runner_args.action_scale) if runner_args.action_scale is not None else float(action_scale)
                resolved_joint_order = "legacy_obs_isaac_action"
            elif runner_args.joint_order == "isaac_obs_legacy_action":
                obs_indices = identity_indices
                act_indices = lab_to_mujoco
                action_scale_vec = float(runner_args.action_scale) if runner_args.action_scale is not None else float(action_scale)
                resolved_joint_order = "isaac_obs_legacy_action"
            elif runner_args.joint_order == "isaaclab_48":
                obs_indices = identity_indices
                act_indices = identity_indices
                action_scale_vec = float(runner_args.action_scale) if runner_args.action_scale is not None else float(action_scale)
                resolved_joint_order = "isaaclab_48"
            elif runner_args.joint_order == "robot_lab_45":
                obs_indices = ROBOT_LAB_JOINT_SWAP
                act_indices = ROBOT_LAB_JOINT_SWAP
                action_scale_vec = [0.125, 0.25, 0.25] * 4
                resolved_joint_order = "robot_lab_45"
            elif self.include_base_lin_vel:
                obs_indices = identity_indices
                act_indices = identity_indices
                action_scale_vec = float(runner_args.action_scale) if runner_args.action_scale is not None else float(action_scale)
                resolved_joint_order = "isaaclab_48"
            else:
                obs_indices = ROBOT_LAB_JOINT_SWAP
                act_indices = ROBOT_LAB_JOINT_SWAP
                action_scale_vec = [0.125, 0.25, 0.25] * 4
                resolved_joint_order = "robot_lab_45"

            self.obs_joint_indices = torch.tensor(obs_indices, device=self.device)
            self.act_joint_indices = torch.tensor(act_indices, device=self.device)
            self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).unsqueeze(0)
            self.last_action = torch.zeros(1, act_dim, device=self.device, dtype=torch.float32)
            self._padding = torch.zeros(6, device=self.device, dtype=torch.float32)
            self.last_target = torch.cat([self._padding, self.joint_pos_initial.squeeze(0)]).detach().clone()
            if isinstance(action_scale_vec, list):
                self.action_scale_vec = torch.tensor(action_scale_vec, device=self.device, dtype=torch.float32).unsqueeze(0)
            else:
                self.action_scale_vec = float(action_scale_vec)
            self.debug_policy_every = int(runner_args.debug_policy_every or 0)
            self.debug_logger = getattr(runner_args, "debug_logger", None)
            self.action_slew_per_step = (
                None
                if runner_args.action_slew_rate is None or runner_args.action_slew_rate <= 0.0
                else float(runner_args.action_slew_rate) / float(runner_args.control_rate_hz)
            )
            self.action_smoothing = min(max(float(runner_args.action_smoothing), 0.0), 1.0)
            self._steps = 0

            print(
                "[INFO] Loaded TD-MPC Newton policy "
                f"checkpoint={checkpoint_path} obs_dim={self.obs_dim} joint_order={resolved_joint_order} "
                f"action_scale={action_scale_vec}"
            )

        def _filter_action(self, raw_action: torch.Tensor) -> torch.Tensor:
            action = raw_action
            if self.action_slew_per_step is not None:
                delta = (action - self.last_action).clamp(-self.action_slew_per_step, self.action_slew_per_step)
                action = self.last_action + delta
            if self.action_smoothing > 0.0:
                alpha = self.action_smoothing
                action = (1.0 - alpha) * action + alpha * self.last_action
            return action

        def _debug_policy_step(
            self,
            state,
            command: torch.Tensor,
            obs: torch.Tensor,
            raw_action: torch.Tensor,
            action: torch.Tensor,
            target: torch.Tensor,
            step_index: int,
        ) -> None:
            root_pos = torch.tensor(state.joint_q[:3], device=self.device, dtype=torch.float32)
            root_vel = torch.tensor(state.joint_qd[:3], device=self.device, dtype=torch.float32)
            joint_q = torch.tensor(state.joint_q[7:], device=self.device, dtype=torch.float32)
            joint_qd = torch.tensor(state.joint_qd[6:], device=self.device, dtype=torch.float32)
            root_pos_list = [round(float(v), 4) for v in root_pos.detach().cpu()]
            root_vel_list = [round(float(v), 4) for v in root_vel.detach().cpu()]
            command_list = [round(float(v), 4) for v in command.squeeze(0).detach().cpu()]
            obs_detached = obs.detach()
            raw_action_detached = raw_action.detach()
            action_detached = action.detach()
            target_detached = target.detach()
            if getattr(runner_args, "print_policy_debug", False):
                print(
                    "[TDMPC DEBUG] "
                    f"step={step_index} root_pos={root_pos_list} root_vel={root_vel_list} command={command_list} "
                    f"obs=[{float(obs_detached.min()):+.3f},{float(obs_detached.max()):+.3f}] "
                    f"action=[{float(action_detached.min()):+.3f},{float(action_detached.max()):+.3f}] "
                    f"|action|={float(action_detached.norm()):.3f} "
                    f"target=[{float(target_detached.min()):+.3f},{float(target_detached.max()):+.3f}] "
                    f"joint_q=[{float(joint_q.min()):+.3f},{float(joint_q.max()):+.3f}] "
                    f"|joint_qd|max={float(joint_qd.abs().max()):.3f}",
                    flush=True,
                )
            if self.debug_logger is not None:
                diagnostics = getattr(self.planner, "last_diagnostics", {})
                self.debug_logger.write(
                    {
                        "event": "tdmpc_policy",
                        "step": int(step_index),
                        "root_pos": tensor_list(root_pos),
                        "root_lin_vel_world": tensor_list(root_vel),
                        "command": tensor_list(command.squeeze(0)),
                        "obs_min": float(obs_detached.min().item()),
                        "obs_max": float(obs_detached.max().item()),
                        "obs_norm": float(obs_detached.norm().item()),
                        "raw_action": tensor_list(raw_action_detached.squeeze(0)),
                        "raw_action_min": float(raw_action_detached.min().item()),
                        "raw_action_max": float(raw_action_detached.max().item()),
                        "raw_action_norm": float(raw_action_detached.norm().item()),
                        "action": tensor_list(action_detached.squeeze(0)),
                        "action_min": float(action_detached.min().item()),
                        "action_max": float(action_detached.max().item()),
                        "action_norm": float(action_detached.norm().item()),
                        "action_max_delta": float((action_detached - self.last_action).abs().max().item()),
                        "action_slew_per_step": self.action_slew_per_step,
                        "action_smoothing": self.action_smoothing,
                        "target": tensor_list(target_detached.squeeze(0)),
                        "target_min": float(target_detached.min().item()),
                        "target_max": float(target_detached.max().item()),
                        "joint_q_min": float(joint_q.min().item()),
                        "joint_q_max": float(joint_q.max().item()),
                        "joint_qd_abs_max": float(joint_qd.abs().max().item()),
                        "planner": {key: float(value) for key, value in diagnostics.items()},
                    }
                )

        @torch.no_grad()
        def compute_joint_targets(self, state, command) -> wp.array:
            obs = compute_obs(
                self.last_action,
                state,
                self.joint_pos_initial,
                self.device,
                self.obs_joint_indices,
                self.gravity_vec,
                command,
                include_base_lin_vel=self.include_base_lin_vel,
                ang_vel_scale=self.ang_vel_scale,
                joint_vel_scale=self.joint_vel_scale,
            )
            step_index = self._steps
            if runner_args.planner_track_command:
                command_cpu = command.squeeze(0).detach().to("cpu")
                self.planner.planner_velocity_target_x = float(command_cpu[0])
                self.planner.planner_velocity_target_y = float(command_cpu[1])
                self.planner.planner_velocity_target_yaw = float(command_cpu[2])
            raw_action = self.planner.plan(obs, eval_mode=True, t0=step_index == 0)
            action = self._filter_action(raw_action)
            rearranged = torch.gather(action, 1, self.act_joint_indices.unsqueeze(0))
            target = self.joint_pos_initial + self.action_scale_vec * rearranged
            if self.debug_policy_every > 0 and step_index % self.debug_policy_every == 0:
                self._debug_policy_step(state, command, obs, raw_action, action, target, step_index)
            self.last_action = action
            self._steps += 1
            padded = torch.cat([self._padding, target.squeeze(0)])
            self.last_target = padded.detach().clone()
            return wp.from_torch(padded, dtype=wp.float32, requires_grad=False)

    return TDMPCGo2Policy


def patch_mujoco_contact_budget(mpm_example, nconmax: int | None, njmax: int | None) -> None:
    if nconmax is None and njmax is None:
        return

    original_create_solvers = mpm_example.Example._create_solvers

    def create_solvers_with_contact_budget(self, mpm_model, mpm_options):
        if nconmax is None and njmax is None:
            return original_create_solvers(self, mpm_model, mpm_options)

        ms = self.cfg["mujoco_solver"]
        resolved_njmax = int(ms["njmax"]) if njmax is None else max(int(ms["njmax"]), int(njmax))
        kwargs = {
            "ls_parallel": bool(ms["ls_parallel"]),
            "njmax": resolved_njmax,
        }
        if nconmax is not None:
            kwargs["nconmax"] = int(nconmax)
        self.solver = mpm_example.newton.solvers.SolverMuJoCo(self.model, **kwargs)
        self.mpm_solver = mpm_example.SolverImplicitMPM(mpm_model, mpm_options)
        print(f"[INFO] MuJoCo contact budget: nconmax={nconmax} njmax={resolved_njmax}")

    mpm_example.Example._create_solvers = create_solvers_with_contact_budget


def patch_isaaclab_deployment_config(mpm_example, runner_args: argparse.Namespace) -> None:
    """Force Newton's Go2 example onto the TD-MPC deployment robot settings."""

    original_load_config = mpm_example.Example._load_config

    def load_isaaclab_config(self, options):
        cfg = original_load_config(self, options)

        robot = cfg.setdefault("robot", {})
        robot["initial_position"] = [0.0, -1.5, float(runner_args.spawn_z)]
        robot["initial_yaw_axis"] = [0.0, 0.0, 1.0]
        robot["initial_yaw_angle_pi_mult"] = 0.5
        robot["enable_self_collisions"] = False

        policy = cfg.setdefault("policy", {})
        if runner_args.actuator_model == "dcmotor":
            # Disable Newton/MuJoCo's native position actuators; direct torques
            # are written to control.joint_f by attach_dcmotor_actuator().
            policy["pd_gains_ke"] = 0.0
            policy["pd_gains_kd"] = 0.0
        else:
            policy["pd_gains_ke"] = float(runner_args.pd_kp)
            policy["pd_gains_kd"] = float(runner_args.pd_kd)
        policy["effort_limit"] = float(runner_args.effort_limit)
        # Flat-Unitree-Go2-train-v0 inherits UnitreeGo2RoughEnvCfg, which
        # overrides JointPositionActionCfg to scale=0.25 with default offsets.
        policy["action_scale"] = float(runner_args.config_action_scale)
        policy["initial_joint_q"] = {
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        }

        shape = cfg.setdefault("builder_defaults", {}).setdefault("shape", {})
        shape["mu"] = float(runner_args.ground_mu)

        simulation = cfg.setdefault("simulation", {})
        simulation["fps"] = 50
        simulation["sim_substeps"] = 4

        cfg.setdefault("control", {})["auto_forward"] = False
        print(
            "[INFO] TD-MPC Newton deployment config forced: "
            f"spawn=(0,-1.5,{float(runner_args.spawn_z):.3g}), yaw=+90deg, "
            f"actuator={runner_args.actuator_model}, kp={float(runner_args.pd_kp):.3g}, kd={float(runner_args.pd_kd):.3g}, "
            f"effort={float(runner_args.effort_limit):.3g}, "
            f"action_scale={float(runner_args.config_action_scale):.3g}, mu={float(runner_args.ground_mu):.3g}"
        )
        return cfg

    mpm_example.Example._load_config = load_isaaclab_config


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def quat_xyzw_to_yaw(quat: Any) -> float:
    x, y, z, w = (float(quat[i]) for i in range(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def attach_command_driver(
    example,
    enabled: bool,
    command: tuple[float, float, float],
    wander: bool,
    wander_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    wander_resample_range: tuple[float, float],
    goal_xy: tuple[float | None, float | None] | None,
    goal_kp: float,
    goal_max_speed: float,
    goal_min_speed: float,
    goal_tolerance: float,
    goal_command_mode: str,
    goal_heading_kp: float,
    goal_max_yaw_rate: float,
    yaw_command_sign: float,
    forward_command_sign: float,
) -> None:
    """Replace George's hardcoded auto-forward with Isaac-style velocity commands."""

    def read_command(self):
        fwd = lat = rot = 0.0
        manual = False
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            manual = bool(fwd or lat or rot)

        if manual:
            self._fixed_command_enabled = False
            self._wander_command_enabled = False
            self._goal_command_enabled = False
            self.command[0, 0] = float(self._forward_command_sign) * float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(self._yaw_command_sign) * float(rot)
            return

        if getattr(self, "_wander_command_enabled", False):
            sim_time = float(self.sim_time)
            if sim_time >= float(self._wander_next_resample_time):
                xr, yr, yawr = self._wander_ranges
                self._wander_command = (
                    random.uniform(*xr),
                    random.uniform(*yr),
                    random.uniform(*yawr),
                )
                self._wander_next_resample_time = sim_time + random.uniform(*self._wander_resample_range)
            self.command[0, 0] = float(self._forward_command_sign) * float(self._wander_command[0])
            self.command[0, 1] = float(self._wander_command[1])
            self.command[0, 2] = float(self._yaw_command_sign) * float(self._wander_command[2])
            return

        if getattr(self, "_goal_command_enabled", False):
            q = self.state_0.joint_q.numpy()
            goal_x, goal_y = self._goal_xy
            dx = 0.0 if goal_x is None else float(goal_x) - float(q[0])
            dy = 0.0 if goal_y is None else float(goal_y) - float(q[1])
            distance = math.hypot(dx, dy)
            if distance <= float(self._goal_tolerance):
                self.command.zero_()
                return

            yaw = quat_xyzw_to_yaw(q[3:7])
            speed = clamp(float(self._goal_kp) * distance, float(self._goal_min_speed), float(self._goal_max_speed))
            if self._goal_command_mode == "forward":
                self.command[0, 0] = float(self._forward_command_sign) * speed
                self.command[0, 1] = 0.0
                self.command[0, 2] = 0.0
            elif self._goal_command_mode == "xy":
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)
                body_x_error = cos_yaw * dx + sin_yaw * dy
                body_y_error = -sin_yaw * dx + cos_yaw * dy
                self.command[0, 0] = float(self._forward_command_sign) * clamp(
                    speed * body_x_error / max(distance, 1e-6),
                    -float(self._goal_max_speed),
                    float(self._goal_max_speed),
                )
                self.command[0, 1] = clamp(
                    speed * body_y_error / max(distance, 1e-6),
                    -float(self._goal_max_speed),
                    float(self._goal_max_speed),
                )
                self.command[0, 2] = 0.0
            else:
                desired_yaw = (math.pi / 2.0 if dy >= 0.0 else -math.pi / 2.0) if goal_x is None else math.atan2(dy, dx)
                yaw_error = wrap_to_pi(desired_yaw - yaw)
                heading_scale = clamp((math.cos(yaw_error) + 1.0) * 0.5, 0.0, 1.0)
                self.command[0, 0] = float(self._forward_command_sign) * speed * heading_scale
                self.command[0, 1] = 0.0
                self.command[0, 2] = clamp(
                    float(self._yaw_command_sign) * float(self._goal_heading_kp) * yaw_error,
                    -float(self._goal_max_yaw_rate),
                    float(self._goal_max_yaw_rate),
                )
            return

        if getattr(self, "_fixed_command_enabled", False):
            self.command[0, 0] = float(self._forward_command_sign) * float(command[0])
            self.command[0, 1] = float(command[1])
            self.command[0, 2] = float(self._yaw_command_sign) * float(command[2])
        else:
            self.command.zero_()

    example._fixed_command_enabled = bool(enabled)
    example._fixed_command = command
    example._wander_command_enabled = bool(wander)
    example._wander_ranges = wander_ranges
    example._wander_resample_range = wander_resample_range
    example._wander_command = (
        random.uniform(*wander_ranges[0]),
        random.uniform(*wander_ranges[1]),
        random.uniform(*wander_ranges[2]),
    )
    example._wander_next_resample_time = 0.0
    example._goal_command_enabled = goal_xy is not None
    example._goal_xy = goal_xy
    example._goal_kp = float(goal_kp)
    example._goal_max_speed = float(goal_max_speed)
    example._goal_min_speed = float(goal_min_speed)
    example._goal_tolerance = float(goal_tolerance)
    example._goal_command_mode = goal_command_mode
    example._goal_heading_kp = float(goal_heading_kp)
    example._goal_max_yaw_rate = float(goal_max_yaw_rate)
    example._yaw_command_sign = float(yaw_command_sign)
    example._forward_command_sign = float(forward_command_sign)
    example._auto_forward = False
    example._read_command = MethodType(read_command, example)


def attach_dcmotor_actuator(example, kp: float, kd: float, effort_limit: float) -> None:
    """Use IsaacLab-style DCMotor torque control instead of Newton native target PD."""

    example._dcmotor_kp = float(kp)
    example._dcmotor_kd = float(kd)
    example._dcmotor_effort_limit = float(effort_limit)

    def apply_control(self):
        target = self.policy.compute_joint_targets(self.state_0, self.command)
        target_t = wp.to_torch(target)
        joint_q = wp.to_torch(self.state_0.joint_q)
        joint_qd = wp.to_torch(self.state_0.joint_qd)
        joint_f = wp.to_torch(self.control.joint_f)

        torque = float(self._dcmotor_kp) * (target_t[6:] - joint_q[7:]) - float(self._dcmotor_kd) * joint_qd[6:]
        limit = float(self._dcmotor_effort_limit)
        if limit > 0.0:
            torque = torque.clamp(-limit, limit)
        joint_f.zero_()
        joint_f[6:] = torque

        if self._action_norm_history is not None:
            with torch.no_grad():
                self._action_norm_history.append(self.policy.last_action.norm().item())

    example.apply_control = MethodType(apply_control, example)


def attach_state_debugger(example, every: int, stop_below_z: float | None, debug_logger: JsonlLogger | None) -> None:
    if every <= 0 and stop_below_z is None:
        return

    original_step = example.step
    step_counter = 0

    def debug_step(self):
        nonlocal step_counter
        original_step()
        if getattr(self, "_precomputed_body_q", None) and not getattr(self, "_precomputing", False):
            return

        step_counter += 1
        should_print = every > 0 and step_counter % every == 0
        root_q = self.state_0.joint_q.numpy()[:7]
        root_qd = self.state_0.joint_qd.numpy()[:6]
        roll, pitch, yaw = quat_xyzw_to_rpy(root_q[3:7])
        root_lin_vel_body = rotate_world_to_body(root_qd[:3], root_q[3:7])
        command = self.command.detach().cpu().numpy()[0] if hasattr(self.command, "detach") else [0.0, 0.0, 0.0]
        root_z = float(root_q[2])
        mjw_data = getattr(getattr(self, "solver", None), "mjw_data", None)
        contact_count: int | str = "n/a"
        if mjw_data is not None and hasattr(mjw_data, "nacon"):
            nacon = mjw_data.nacon
            try:
                contact_count = int(nacon.numpy()[0])
            except (AttributeError, IndexError, TypeError, ValueError):
                try:
                    contact_count = int(nacon)
                except (TypeError, ValueError):
                    contact_count = str(nacon)

        if should_print and getattr(self, "_print_state_debug", False):
            print(
                "[STATE DEBUG] "
                f"step={step_counter} t={float(self.sim_time):.3f} "
                f"root_pos=({root_q[0]:+.4f},{root_q[1]:+.4f},{root_q[2]:+.4f}) "
                f"root_quat=({root_q[3]:+.4f},{root_q[4]:+.4f},{root_q[5]:+.4f},{root_q[6]:+.4f}) "
                f"root_rpy=({roll:+.3f},{pitch:+.3f},{yaw:+.3f}) "
                f"root_lin_vel=({root_qd[0]:+.4f},{root_qd[1]:+.4f},{root_qd[2]:+.4f}) "
                f"root_lin_vel_body=({root_lin_vel_body[0]:+.4f},{root_lin_vel_body[1]:+.4f},{root_lin_vel_body[2]:+.4f}) "
                f"root_ang_vel=({root_qd[3]:+.4f},{root_qd[4]:+.4f},{root_qd[5]:+.4f}) "
                f"command=({command[0]:+.3f},{command[1]:+.3f},{command[2]:+.3f}) "
                f"mujoco_contacts={contact_count}",
                flush=True,
            )
        if should_print and debug_logger is not None:
            record: dict[str, Any] = {
                "event": "state",
                "step": int(step_counter),
                "sim_time": float(self.sim_time),
                "root_pos": [float(value) for value in root_q[:3]],
                "root_quat_xyzw": [float(value) for value in root_q[3:7]],
                "root_rpy": [roll, pitch, yaw],
                "root_lin_vel_world": [float(value) for value in root_qd[:3]],
                "root_lin_vel_body": root_lin_vel_body,
                "root_ang_vel_world": [float(value) for value in root_qd[3:6]],
                "command": [float(value) for value in command[:3]],
                "mujoco_contacts": contact_count,
            }
            if hasattr(self, "control") and hasattr(self, "pd_ke") and hasattr(self, "pd_kd"):
                joint_q = self.state_0.joint_q.numpy()[7:]
                joint_qd = self.state_0.joint_qd.numpy()[6:]
                target = self.control.joint_target_pos.numpy()[6:]
                if hasattr(self, "_dcmotor_kp"):
                    target = self.policy.last_target.detach().cpu().numpy()[6:]
                kp = float(getattr(self, "_dcmotor_kp", self.pd_ke))
                kd = float(getattr(self, "_dcmotor_kd", self.pd_kd))
                err = [float(target[i] - joint_q[i]) for i in range(len(target))]
                p_term = [float(kp * value) for value in err]
                d_term = [float(-kd * joint_qd[i]) for i in range(len(joint_qd))]
                effort_limit = float(getattr(self, "_dcmotor_effort_limit", getattr(self, "effort_limit", 0.0)))
                summed_torque = [p_term[i] + d_term[i] for i in range(len(p_term))]
                record["pd"] = {
                    "kp": kp,
                    "kd": kd,
                    "effort_limit": effort_limit,
                    "joint_names": list(getattr(self, "leg_joint_names", [])),
                    "target": [float(value) for value in target],
                    "joint_q": [float(value) for value in joint_q],
                    "joint_qd": [float(value) for value in joint_qd],
                    "err": err,
                    "p_term": p_term,
                    "d_term": d_term,
                    "summed_torque": summed_torque,
                    "max_abs_err": max((abs(value) for value in err), default=0.0),
                    "max_abs_p": max((abs(value) for value in p_term), default=0.0),
                    "max_abs_d": max((abs(value) for value in d_term), default=0.0),
                    "p_clamped_count": sum(1 for value in p_term if effort_limit > 0.0 and abs(value) > effort_limit),
                    "sum_clamped_count": sum(
                        1 for value in summed_torque if effort_limit > 0.0 and abs(value) > effort_limit
                    ),
                }
            debug_logger.write(record)

        if stop_below_z is not None and root_z < stop_below_z:
            if debug_logger is not None:
                debug_logger.write(
                    {
                        "event": "stop_below_z",
                        "step": int(step_counter),
                        "sim_time": float(self.sim_time),
                        "root_z": root_z,
                        "threshold": float(stop_below_z),
                    }
                )
            raise RuntimeError(
                f"Base dropped below z threshold: root_z={root_z:.4f} < {stop_below_z:.4f} "
                f"at step={step_counter}, sim_time={float(self.sim_time):.3f}."
            )

    example.step = MethodType(debug_step, example)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an ldm-quad TD-MPC checkpoint in Newton MPM Go2.",
        add_help=True,
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to ldm-quad MBRL checkpoint.")
    parser.add_argument("--newton-mpm-path", type=str, default=None, help="Path to Newton's mpm_go2_multi folder.")
    parser.add_argument(
        "--force-isaaclab-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force IsaacLab-like robot config.",
    )
    parser.add_argument(
        "--pd-kp",
        "--pd_kp",
        type=float,
        default=25.0,
        help="Newton joint position gain. IsaacLab/George Go2 configs both use 25.0.",
    )
    parser.add_argument(
        "--pd-kd",
        "--pd_kd",
        type=float,
        default=0.5,
        help="Newton joint damping gain. Defaults to the IsaacLab Go2 DCMotor damping value.",
    )
    parser.add_argument(
        "--effort-limit",
        "--effort_limit",
        type=float,
        default=23.5,
        help="Per-joint Newton actuator effort limit.",
    )
    parser.add_argument(
        "--actuator-model",
        "--actuator_model",
        choices=["dcmotor", "native_pd"],
        default="dcmotor",
        help="Use IsaacLab-style summed/clamped DCMotor torque control or Newton's native target-position PD.",
    )
    parser.add_argument(
        "--config-action-scale",
        "--config_action_scale",
        type=float,
        default=0.25,
        help="Action scale written into the Newton config. Matches Flat-Unitree-Go2-train-v0 inherited action scale.",
    )
    parser.add_argument(
        "--ground-mu",
        "--ground_mu",
        type=float,
        default=1.0,
        help="Rigid ground Coulomb friction in Newton before the mud patch. Defaults to the IsaacLab ground friction value.",
    )
    parser.add_argument(
        "--spawn-z",
        "--spawn_z",
        type=float,
        default=0.4,
        help="Initial floating-base z position in Newton. Defaults to the IsaacLab initial root height.",
    )
    parser.add_argument(
        "--joint-order",
        choices=[
            "auto",
            "isaaclab_48",
            "legacy_48",
            "legacy_obs_isaac_action",
            "isaac_obs_legacy_action",
            "robot_lab_45",
        ],
        default="auto",
        help="Observation/action joint order. auto uses checkpoint obs_dim: 48=isaaclab_48 identity, 45=robot_lab_45.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=None,
        help="Override joint target scale for 48-D checkpoints. Defaults to the IsaacLab deployment config value.",
    )
    parser.add_argument("--action-low", type=float, default=-1.0, help="Planner action lower bound.")
    parser.add_argument("--action-high", type=float, default=1.0, help="Planner action upper bound.")
    parser.add_argument(
        "--action-slew-rate",
        "--action_slew_rate",
        type=float,
        default=None,
        help=(
            "Limit TD-MPC action change per second before converting to joint targets. "
            "Omit to use a Newton deployment default when a command/goal is active. Set <=0 to disable."
        ),
    )
    parser.add_argument(
        "--action-smoothing",
        "--action_smoothing",
        type=float,
        default=None,
        help=(
            "EMA smoothing weight on filtered actions in [0, 1). "
            "Omit to use a Newton deployment default when a command/goal is active. 0 disables smoothing."
        ),
    )
    parser.add_argument(
        "--control-rate-hz",
        "--control_rate_hz",
        type=float,
        default=50.0,
        help="Control rate used to convert --action-slew-rate into a per-step action delta.",
    )
    parser.add_argument("--candidates", type=int, default=None, help="Override planner candidate count for slow MPM runs.")
    parser.add_argument("--min-std", type=float, default=None, help="Override planner minimum std.")
    parser.add_argument("--max-std", type=float, default=None, help="Override planner maximum std.")
    parser.add_argument(
        "--planner-track-command",
        "--planner_track_command",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set the planner's velocity objective target to the current command each control step.",
    )
    parser.add_argument(
        "--planner-velocity-objective-weight",
        "--planner_velocity_objective_weight",
        type=float,
        default=None,
        help=(
            "Override planner-only velocity tracking reward weight. Omit to use a Newton deployment default "
            "when a command/goal is active, otherwise the checkpoint value."
        ),
    )
    parser.add_argument("--command-x", "--command_x", type=float, default=None, help="Fixed forward velocity command in m/s.")
    parser.add_argument("--command-y", "--command_y", type=float, default=0.0, help="Fixed lateral velocity command in m/s.")
    parser.add_argument("--command-yaw", "--command_yaw", type=float, default=0.0, help="Fixed yaw velocity command in rad/s.")
    parser.add_argument(
        "--yaw-command-sign",
        "--yaw_command_sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help="Sign conversion for yaw-rate commands sent to the TD-MPC policy in Newton.",
    )
    parser.add_argument(
        "--forward-command-sign",
        "--forward_command_sign",
        type=float,
        choices=[-1.0, 1.0],
        default=-1.0,
        help="Sign conversion for forward velocity commands sent to the TD-MPC policy in Newton.",
    )
    parser.add_argument(
        "--wander",
        action="store_true",
        default=False,
        help="Sample movement commands using the same defaults as scripts/mbrl/play.py.",
    )
    parser.add_argument("--wander-x-min", "--wander_x_min", type=float, default=-0.8, help="Minimum wander forward velocity command.")
    parser.add_argument("--wander-x-max", "--wander_x_max", type=float, default=0.8, help="Maximum wander forward velocity command.")
    parser.add_argument("--wander-y-min", "--wander_y_min", type=float, default=-0.4, help="Minimum wander lateral velocity command.")
    parser.add_argument("--wander-y-max", "--wander_y_max", type=float, default=0.4, help="Maximum wander lateral velocity command.")
    parser.add_argument("--wander-yaw-min", "--wander_yaw_min", type=float, default=-0.8, help="Minimum wander yaw velocity command.")
    parser.add_argument("--wander-yaw-max", "--wander_yaw_max", type=float, default=0.8, help="Maximum wander yaw velocity command.")
    parser.add_argument("--wander-resample-min", "--wander_resample_min", type=float, default=3.0, help="Minimum wander command resample time.")
    parser.add_argument("--wander-resample-max", "--wander_resample_max", type=float, default=5.0, help="Maximum wander command resample time.")
    parser.add_argument("--goal-x", "--goal_x", type=float, default=None, help="World-frame x target for goal-command mode.")
    parser.add_argument("--goal-y", "--goal_y", type=float, default=None, help="World-frame y target for goal-command mode.")
    parser.add_argument("--goal-kp", "--goal_kp", type=float, default=0.8, help="Velocity command gain for goal-command mode.")
    parser.add_argument(
        "--goal-command-mode",
        "--goal_command_mode",
        choices=["forward", "heading", "xy"],
        default="forward",
        help=(
            "forward commands body-frame forward speed only; heading commands forward speed plus yaw-to-goal; "
            "xy commands body-frame x/y velocity with zero yaw."
        ),
    )
    parser.add_argument(
        "--goal-heading-kp",
        "--goal_heading_kp",
        type=float,
        default=0.5,
        help="Yaw-rate gain for --goal-command-mode heading, matching IsaacLab heading_control_stiffness by default.",
    )
    parser.add_argument(
        "--goal-max-yaw-rate",
        "--goal_max_yaw_rate",
        type=float,
        default=1.0,
        help="Maximum absolute yaw-rate command for heading goal mode.",
    )
    parser.add_argument(
        "--goal-max-speed",
        "--goal_max_speed",
        type=float,
        default=0.5,
        help="Maximum x/y velocity command magnitude for goal-command mode.",
    )
    parser.add_argument(
        "--goal-min-speed",
        "--goal_min_speed",
        type=float,
        default=0.05,
        help="Minimum nonzero command speed while outside the goal tolerance.",
    )
    parser.add_argument(
        "--goal-tolerance",
        "--goal_tolerance",
        type=float,
        default=0.15,
        help="Stop commanding motion once the base is this close to the goal.",
    )
    parser.add_argument(
        "--auto-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a fixed velocity command without keyboard input. Use --no-auto-forward to disable.",
    )
    parser.add_argument(
        "--nconmax",
        type=int,
        default=256,
        help="MuJoCo-Warp contact capacity. Newton's auto-estimate can be zero before initial contact; set <=0 to leave unchanged.",
    )
    parser.add_argument(
        "--njmax",
        type=int,
        default=512,
        help="MuJoCo-Warp constraint capacity floor. Set <=0 to leave the Newton config value unchanged.",
    )
    parser.add_argument(
        "--debug-policy-every",
        type=int,
        default=0,
        help="Print TD-MPC policy diagnostics every N control steps. Disabled when 0.",
    )
    parser.add_argument(
        "--debug-state-every",
        type=int,
        default=0,
        help="Print Newton root pose/velocity after physics every N simulation steps. Disabled when 0.",
    )
    parser.add_argument(
        "--stop-below-z",
        type=float,
        default=None,
        help="Abort once the floating base z position drops below this value.",
    )
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=None,
        help="Write structured TD-MPC/state debug records to this JSONL file.",
    )

    # Newton's example parser owns rendering/video/simulation flags.  Parse only
    # our flags first, then rebuild sys.argv so Newton sees the rest.
    runner_args, newton_argv = parser.parse_known_args()
    motion_command_requested = (
        bool(runner_args.wander)
        or runner_args.goal_x is not None
        or runner_args.goal_y is not None
        or runner_args.command_x is not None
        or abs(float(runner_args.command_y)) > 0.0
        or abs(float(runner_args.command_yaw)) > 0.0
    )
    if runner_args.planner_velocity_objective_weight is None and motion_command_requested:
        # The saved checkpoint has this at 0.0 because IsaacLab evaluation can
        # trust the learned reward under matching dynamics. In Newton, the same
        # reward can prefer yawing in place, so deployment needs a small
        # command-tracking tie-breaker. Keep this modest; too much weight makes
        # the planner chase predicted velocity by pitching/falling forward.
        runner_args.planner_velocity_objective_weight = 1.0
        runner_args.planner_velocity_objective_source = "newton_command_default"
    elif runner_args.planner_velocity_objective_weight is None:
        runner_args.planner_velocity_objective_source = "checkpoint"
    else:
        runner_args.planner_velocity_objective_source = "cli"
    if runner_args.action_slew_rate is None and motion_command_requested:
        runner_args.action_slew_rate = 0.0
        runner_args.action_filter_source = "isaaclab_default"
    elif runner_args.action_slew_rate is None:
        runner_args.action_slew_rate = 0.0
        runner_args.action_filter_source = "checkpoint"
    else:
        runner_args.action_filter_source = "cli"
    if runner_args.action_smoothing is None and motion_command_requested:
        runner_args.action_smoothing = 0.0
    elif runner_args.action_smoothing is None:
        runner_args.action_smoothing = 0.0
    runner_args.print_policy_debug = runner_args.debug_policy_every > 0
    runner_args.print_state_debug = runner_args.debug_state_every > 0
    debug_logger = JsonlLogger(runner_args.log_jsonl.expanduser() if runner_args.log_jsonl is not None else None)
    runner_args.debug_logger = debug_logger if debug_logger.enabled else None
    if debug_logger.enabled:
        if runner_args.debug_policy_every <= 0:
            runner_args.debug_policy_every = 20
        if runner_args.debug_state_every <= 0:
            runner_args.debug_state_every = 20
        debug_logger.write(
            {
                "event": "run_start",
                "checkpoint": str(runner_args.checkpoint),
                "controller": "tdmpc",
                "debug_policy_every": int(runner_args.debug_policy_every),
                "debug_state_every": int(runner_args.debug_state_every),
                "action_slew_rate": float(runner_args.action_slew_rate),
                "action_smoothing": float(runner_args.action_smoothing),
                "action_filter_source": runner_args.action_filter_source,
                "control_rate_hz": float(runner_args.control_rate_hz),
                "joint_order": runner_args.joint_order,
                "action_scale_override": None if runner_args.action_scale is None else float(runner_args.action_scale),
                "pd_kp": float(runner_args.pd_kp),
                "pd_kd": float(runner_args.pd_kd),
                "effort_limit": float(runner_args.effort_limit),
                "actuator_model": runner_args.actuator_model,
                "config_action_scale": float(runner_args.config_action_scale),
                "ground_mu": float(runner_args.ground_mu),
                "spawn_z": float(runner_args.spawn_z),
                "planner_velocity_objective_weight": (
                    None
                    if runner_args.planner_velocity_objective_weight is None
                    else float(runner_args.planner_velocity_objective_weight)
                ),
                "planner_velocity_objective_source": runner_args.planner_velocity_objective_source,
                "goal_command_mode": runner_args.goal_command_mode,
                "goal_heading_kp": float(runner_args.goal_heading_kp),
                "goal_max_yaw_rate": float(runner_args.goal_max_yaw_rate),
                "goal_x": None if runner_args.goal_x is None else float(runner_args.goal_x),
                "goal_y": None if runner_args.goal_y is None else float(runner_args.goal_y),
                "yaw_command_sign": float(runner_args.yaw_command_sign),
                "forward_command_sign": float(runner_args.forward_command_sign),
            }
        )
        print(f"[INFO] Writing structured debug log to {runner_args.log_jsonl.expanduser()}")
    if runner_args.planner_velocity_objective_weight is not None:
        print(
            "[INFO] Planner velocity objective "
            f"weight={float(runner_args.planner_velocity_objective_weight):.3g} "
            f"source={runner_args.planner_velocity_objective_source}"
        )
    if runner_args.action_slew_rate > 0.0 or runner_args.action_smoothing > 0.0:
        print(
            "[INFO] Action filter "
            f"slew_rate={float(runner_args.action_slew_rate):.3g}/s "
            f"smoothing={float(runner_args.action_smoothing):.3g} "
            f"source={runner_args.action_filter_source}"
        )
    checkpoint_path = runner_args.checkpoint.expanduser().resolve()
    try:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        mpm_path = resolve_newton_mpm_path(runner_args.newton_mpm_path)
        add_newton_paths(mpm_path)

        try:
            import newton.examples.mpm.mpm_go2_multi.example_mpm_go2_multi as mpm_example
        except ModuleNotFoundError:
            import mpm_go2_multi.example_mpm_go2_multi as mpm_example

        tdmpc_policy_cls = make_tdmpc_policy_class(checkpoint_path, runner_args)
        mpm_example.Go2Policy = tdmpc_policy_cls

        if runner_args.force_isaaclab_config:
            patch_isaaclab_deployment_config(mpm_example, runner_args)
        patch_mujoco_contact_budget(
            mpm_example,
            runner_args.nconmax if runner_args.nconmax and runner_args.nconmax > 0 else None,
            runner_args.njmax if runner_args.njmax and runner_args.njmax > 0 else None,
        )

        # Newton's example still expects --policy-path; after monkey-patching
        # this is only a constructor argument, so point it at the TD-MPC checkpoint.
        sys.argv = [str(mpm_path / "example_mpm_go2_multi.py"), "--policy-path", str(checkpoint_path), *newton_argv]

        newton_parser = mpm_example.newton.examples.create_parser()
        newton_parser.add_argument("--config", "-c", type=str, default=None)
        newton_parser.add_argument("--voxel-size", "-dx", type=float, default=None)
        newton_parser.add_argument("--max-iterations", "-it", type=int, default=None)
        newton_parser.add_argument("--tolerance", "-tol", type=float, default=None)
        newton_parser.add_argument("--policy-path", "-cp", type=str, default=None)
        newton_parser.add_argument("--precompute-frames", type=int, default=0)
        newton_parser.add_argument("--video", type=str, default=None)
        newton_parser.add_argument("--video-fps", type=int, default=50)
        newton_parser.add_argument("--debug-forces", action="store_true")
        newton_parser.add_argument("--debug-pd", action="store_true")
        newton_parser.add_argument("--plot-actions", type=str, default=None)
        newton_parser.add_argument("--plot-forces", type=str, default=None)
        newton_parser.add_argument("--plot-forces-foot", type=str, default="FL_calf")
        newton_parser.add_argument("--plot-forces-mode", choices=["magnitude", "xyz"], default="magnitude")
        newton_parser.add_argument("--plot-joint-angles", type=str, default=None)

        viewer, args = mpm_example.newton.examples.init(newton_parser)
        if wp.get_device().is_cpu:
            raise RuntimeError("Newton MPM Go2 requires a GPU Warp device.")

        example = mpm_example.Example(viewer, args)
        if runner_args.actuator_model == "dcmotor":
            attach_dcmotor_actuator(example, runner_args.pd_kp, runner_args.pd_kd, runner_args.effort_limit)
            print(
                "[INFO] IsaacLab DCMotor actuator enabled: "
                f"torque=clamp(kp*(target-q)-kd*qd, +/-{float(runner_args.effort_limit):.3g})"
            )
        command_x = 0.0 if runner_args.command_x is None else float(runner_args.command_x)
        fixed_command = (command_x, float(runner_args.command_y), float(runner_args.command_yaw))
        wander_ranges = (
            (float(runner_args.wander_x_min), float(runner_args.wander_x_max)),
            (float(runner_args.wander_y_min), float(runner_args.wander_y_max)),
            (float(runner_args.wander_yaw_min), float(runner_args.wander_yaw_max)),
        )
        wander_resample_range = (float(runner_args.wander_resample_min), float(runner_args.wander_resample_max))
        goal_xy: tuple[float | None, float | None] | None = None
        if runner_args.goal_x is not None or runner_args.goal_y is not None:
            start_q = example.state_0.joint_q.numpy()
            goal_xy = (
                None if runner_args.goal_x is None else float(runner_args.goal_x),
                float(start_q[1]) if runner_args.goal_y is None else float(runner_args.goal_y),
            )
        attach_command_driver(
            example,
            runner_args.auto_forward,
            fixed_command,
            runner_args.wander,
            wander_ranges,
            wander_resample_range,
            goal_xy,
            runner_args.goal_kp,
            runner_args.goal_max_speed,
            runner_args.goal_min_speed,
            runner_args.goal_tolerance,
            runner_args.goal_command_mode,
            runner_args.goal_heading_kp,
            runner_args.goal_max_yaw_rate,
            runner_args.yaw_command_sign,
            runner_args.forward_command_sign,
        )
        if runner_args.wander:
            print(
                "[INFO] Velocity command=wander "
                f"x=({wander_ranges[0][0]:.3f}, {wander_ranges[0][1]:.3f}) "
                f"y=({wander_ranges[1][0]:.3f}, {wander_ranges[1][1]:.3f}) "
                f"yaw=({wander_ranges[2][0]:.3f}, {wander_ranges[2][1]:.3f})"
            )
        elif goal_xy is not None:
            goal_x_text = "free" if goal_xy[0] is None else f"{goal_xy[0]:.3f}"
            print(
                "[INFO] Goal command enabled: "
                f"goal=({goal_x_text}, {goal_xy[1]:.3f}) "
                f"mode={runner_args.goal_command_mode}, speed<= {runner_args.goal_max_speed:.3f}, "
                f"yaw_rate<= {runner_args.goal_max_yaw_rate:.3f}, tolerance={runner_args.goal_tolerance:.3f}; "
                "manual keyboard input overrides it."
        )
        elif runner_args.auto_forward and any(abs(value) > 0.0 for value in fixed_command):
            print(
                "[INFO] Fixed command enabled: "
                f"command=({fixed_command[0]:.3f}, {fixed_command[1]:.3f}, {fixed_command[2]:.3f}); "
                "manual keyboard input overrides it."
            )
        attach_state_debugger(example, runner_args.debug_state_every, runner_args.stop_below_z, runner_args.debug_logger)
        example._print_state_debug = bool(runner_args.print_state_debug)
        if getattr(args, "precompute_frames", 0) > 0:
            example.precompute(args.precompute_frames)
        mpm_example.newton.examples.run(example, args)
    finally:
        debug_logger.close()


if __name__ == "__main__":
    main()
