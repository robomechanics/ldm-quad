#!/usr/bin/env python3

"""Run an ldm-quad TD-MPC checkpoint inside Newton's MPM Go2 example.

This script intentionally does not copy Newton code into this repository.  It
loads the existing ``mpm_go2_multi`` example, replaces its RSL-RL ``Go2Policy``
hook with a small TD-MPC wrapper, and lets Newton own simulation/rendering.
"""

from __future__ import annotations

import argparse
import os
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
        action_bounds_finite=True,
        planner_velocity_objective_weight=args.get("planner_velocity_objective_weight", 0.0),
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
            action_scale: float = 0.25,
            obs_dim: int = 48,
            act_dim: int = 12,
            hidden_dims=(512, 256, 128),
            search_relative_to: Path | None = None,
        ):
            del action_scale, obs_dim, hidden_dims, search_relative_to
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

            if runner_args.joint_order == "legacy_48":
                obs_indices = mujoco_to_lab
                act_indices = lab_to_mujoco
                action_scale_vec: float | list[float] = runner_args.action_scale
            elif runner_args.joint_order == "robot_lab_45":
                obs_indices = ROBOT_LAB_JOINT_SWAP
                act_indices = ROBOT_LAB_JOINT_SWAP
                action_scale_vec = [0.125, 0.25, 0.25] * 4
            elif self.include_base_lin_vel:
                obs_indices = mujoco_to_lab
                act_indices = lab_to_mujoco
                action_scale_vec = runner_args.action_scale
            else:
                obs_indices = ROBOT_LAB_JOINT_SWAP
                act_indices = ROBOT_LAB_JOINT_SWAP
                action_scale_vec = [0.125, 0.25, 0.25] * 4

            self.obs_joint_indices = torch.tensor(obs_indices, device=self.device)
            self.act_joint_indices = torch.tensor(act_indices, device=self.device)
            self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).unsqueeze(0)
            self.last_action = torch.zeros(1, act_dim, device=self.device, dtype=torch.float32)
            self._padding = torch.zeros(6, device=self.device, dtype=torch.float32)
            if isinstance(action_scale_vec, list):
                self.action_scale_vec = torch.tensor(action_scale_vec, device=self.device, dtype=torch.float32).unsqueeze(0)
            else:
                self.action_scale_vec = float(action_scale_vec)
            self._steps = 0

            print(
                "[INFO] Loaded TD-MPC Newton policy "
                f"checkpoint={checkpoint_path} obs_dim={self.obs_dim} joint_order={runner_args.joint_order}"
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
            action = self.planner.plan(obs, eval_mode=True, t0=self._steps == 0)
            self._steps += 1
            self.last_action = action
            rearranged = torch.gather(action, 1, self.act_joint_indices.unsqueeze(0))
            target = self.joint_pos_initial + self.action_scale_vec * rearranged
            padded = torch.cat([self._padding, target.squeeze(0)])
            return wp.from_torch(padded, dtype=wp.float32, requires_grad=False)

    return TDMPCGo2Policy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an ldm-quad TD-MPC checkpoint in Newton MPM Go2.",
        add_help=True,
    )
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to ldm-quad MBRL checkpoint.")
    parser.add_argument("--newton-mpm-path", type=str, default=None, help="Path to Newton's mpm_go2_multi folder.")
    parser.add_argument(
        "--joint-order",
        choices=["auto", "legacy_48", "robot_lab_45"],
        default="auto",
        help="Observation/action joint remap convention. auto uses checkpoint obs_dim: 48=legacy_48, 45=robot_lab_45.",
    )
    parser.add_argument("--action-scale", type=float, default=0.25, help="Joint target scale for 48-D checkpoints.")
    parser.add_argument("--action-low", type=float, default=-1.0, help="Planner action lower bound.")
    parser.add_argument("--action-high", type=float, default=1.0, help="Planner action upper bound.")
    parser.add_argument("--candidates", type=int, default=None, help="Override planner candidate count for slow MPM runs.")
    parser.add_argument("--min-std", type=float, default=None, help="Override planner minimum std.")
    parser.add_argument("--max-std", type=float, default=None, help="Override planner maximum std.")

    # Newton's example parser owns rendering/video/simulation flags.  Parse only
    # our flags first, then rebuild sys.argv so Newton sees the rest.
    runner_args, newton_argv = parser.parse_known_args()
    checkpoint_path = runner_args.checkpoint.expanduser().resolve()
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

    # The Newton example still expects --policy-path; after monkey-patching this
    # is only a constructor argument, so point it at the TD-MPC checkpoint.
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
    newton_parser.add_argument("--plot-actions", type=str, default=None)
    newton_parser.add_argument("--plot-forces", type=str, default=None)
    newton_parser.add_argument("--plot-forces-foot", type=str, default="FL_calf")
    newton_parser.add_argument("--plot-forces-mode", choices=["magnitude", "xyz"], default="magnitude")

    viewer, args = mpm_example.newton.examples.init(newton_parser)
    if wp.get_device().is_cpu:
        raise RuntimeError("Newton MPM Go2 requires a GPU Warp device.")

    example = mpm_example.Example(viewer, args)
    if getattr(args, "precompute_frames", 0) > 0:
        example.precompute(args.precompute_frames)
    mpm_example.newton.examples.run(example, args)


if __name__ == "__main__":
    main()
