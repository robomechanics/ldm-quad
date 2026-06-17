#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play or evaluate an MBRL checkpoint on the Go2 walking task."""

import argparse
from copy import deepcopy
import os
import random
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "source", "ldm_quad")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play or evaluate an MBRL checkpoint on an Isaac Lab task.")
parser.add_argument("--video", action="store_true", default=False, help="Record a rollout video.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task to evaluate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to an MBRL checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of completed episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=4000, help="Maximum environment steps to run.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--command_x", type=float, default=None, help="Fixed forward velocity command in m/s.")
parser.add_argument("--command_y", type=float, default=None, help="Fixed lateral velocity command in m/s.")
parser.add_argument("--command_yaw", type=float, default=None, help="Fixed yaw velocity command in rad/s.")
parser.add_argument(
    "--mismatch",
    type=str,
    default="nominal",
    choices=["nominal", "low_friction", "mass", "motor_weakness", "rough", "push"],
    help="Held-out system/terrain mismatch to apply for play/evaluation.",
)
parser.add_argument("--mismatch_friction", type=float, default=0.35, help="Static/dynamic friction used by --mismatch low_friction.")
parser.add_argument("--mismatch_mass_add", type=float, default=3.0, help="Additional base mass in kg for --mismatch mass.")
parser.add_argument("--mismatch_motor_scale", type=float, default=0.6, help="Action scale multiplier for --mismatch motor_weakness.")
parser.add_argument("--mismatch_push_velocity", type=float, default=1.0, help="Max planar push velocity for --mismatch push.")
parser.add_argument("--mismatch_push_interval", type=float, default=3.0, help="Push interval in seconds for --mismatch push.")
parser.add_argument("--mismatch_rough_height", type=float, default=0.04, help="Max box height for --mismatch rough video terrain.")
parser.add_argument("--mismatch_rough_noise", type=float, default=0.025, help="Max random roughness noise for --mismatch rough video terrain.")
parser.add_argument(
    "--mismatch_mode",
    type=str,
    default="immediate",
    choices=["immediate", "delayed"],
    help="Apply mismatch from reset, or delay runtime-capable mismatches for adaptation videos.",
)
parser.add_argument(
    "--mismatch_start_step",
    type=int,
    default=250,
    help="Start applying runtime-capable mismatch after this many steps. Supported for motor_weakness and push.",
)
parser.add_argument(
    "--mismatch_ramp_steps",
    type=int,
    default=0,
    help="Linearly ramp runtime motor weakness over this many steps after --mismatch_start_step.",
)
parser.add_argument(
    "--showcase",
    action="store_true",
    default=False,
    help="Line up env groups in one vectorized play run with runtime mismatches.",
)
parser.add_argument(
    "--showcase_groups",
    nargs="+",
    default=["nominal", "motor_weakness", "push"],
    choices=["nominal", "motor_weakness", "push"],
    help="Runtime mismatch groups to line up when --showcase is enabled.",
)
parser.add_argument("--prior_checkpoint", type=str, default=None, help="Override policy prior checkpoint.")
parser.add_argument(
    "--prior_type",
    type=str,
    default=None,
    choices=["auto", "skrl", "torchscript", "rsl_jit"],
    help="Override locomotion prior format. auto tries TorchScript first, then skrl.",
)
parser.add_argument(
    "--prior_obs_adapter",
    type=str,
    default=None,
    help="Override observation adapter for TorchScript priors.",
)
parser.add_argument(
    "--prior_action_adapter",
    type=str,
    default=None,
    help="Override action adapter for TorchScript priors.",
)
parser.add_argument("--prior_task", type=str, default=None, help="Override task used to load the prior policy config.")
parser.add_argument("--prior_algorithm", type=str, default=None, help="Override skrl algorithm for the prior policy.")
parser.add_argument("--prior_agent", type=str, default=None, help="Override skrl prior agent config entry point.")
parser.add_argument("--prior_only", action="store_true", default=False, help="Run the locomotion prior without MBRL/MPPI.")
parser.add_argument(
    "--prior_control_mode",
    type=str,
    default=None,
    choices=["residual", "full_action"],
    help="Override how the planner uses the prior: residual actions around it, or full-action MPPI warm-started by it.",
)
parser.add_argument("--prior_residual_scale", type=float, default=None, help="Override residual scale around the prior.")
parser.add_argument("--prior_residual_penalty", type=float, default=None, help="Override residual penalty around the prior.")
parser.add_argument(
    "--prior_candidate_fraction",
    type=float,
    default=None,
    help="Override fraction of planner candidates reserved for pure or near-prior residual rollouts.",
)
parser.add_argument(
    "--prior_candidate_noise",
    type=float,
    default=None,
    help="Override residual noise std for extra prior-centered candidates.",
)
parser.add_argument(
    "--prior_command_candidate_fraction",
    type=float,
    default=None,
    help="Override fraction of full-action planner candidates generated by rolling the prior with perturbed commands.",
)
parser.add_argument("--prior_command_noise", type=float, default=None, help="Override Gaussian noise std added to prior command candidates.")
parser.add_argument(
    "--prior_command_start",
    type=int,
    default=None,
    help="Override start index of velocity command observations for command-perturbed prior candidates.",
)
parser.add_argument("--prior_command_dim", type=int, default=None, help="Override number of command observation dimensions to perturb.")
parser.add_argument(
    "--prior_fallback",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override fallback to the pure prior unless the residual plan improves predicted return.",
)
parser.add_argument(
    "--prior_acceptance_margin",
    type=float,
    default=None,
    help="Override required predicted-return improvement before using residual actions.",
)
parser.add_argument(
    "--planner_use_best_candidate",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override using the best evaluated candidate as the final MPPI plan.",
)
parser.add_argument("--num_pi_trajs", type=int, default=None, help="Override TD-MPC2-style policy trajectories in latent planning.")
parser.add_argument("--min_std", type=float, default=None, help="Override minimum latent planner action std.")
parser.add_argument("--max_std", type=float, default=None, help="Override maximum latent planner action std.")
parser.add_argument(
    "--planner_use_continue_model",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override using the learned continuation model during latent planning.",
)
parser.add_argument(
    "--planner_hard_continue_model",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override using a hard learned continuation mask during latent planning.",
)
parser.add_argument("--planner_continue_threshold", type=float, default=None, help="Override hard continuation threshold.")
parser.add_argument("--planner_velocity_objective_weight", type=float, default=None, help="Override planner-only velocity objective weight.")
parser.add_argument("--planner_velocity_target_x", type=float, default=None, help="Override planner-only target body x velocity.")
parser.add_argument("--planner_velocity_target_y", type=float, default=None, help="Override planner-only target body y velocity.")
parser.add_argument("--planner_velocity_target_yaw", type=float, default=None, help="Override planner-only target yaw rate.")
parser.add_argument("--debug_actions", action="store_true", default=False, help="Print action and velocity-command stats.")
parser.add_argument(
    "--wander",
    action="store_true",
    default=False,
    help="Sample nonzero movement commands instead of using the play environment's standing/heading mix.",
)
parser.add_argument("--wander_x_min", type=float, default=-0.8, help="Minimum wander forward velocity command.")
parser.add_argument("--wander_x_max", type=float, default=0.8, help="Maximum wander forward velocity command.")
parser.add_argument("--wander_y_min", type=float, default=-0.4, help="Minimum wander lateral velocity command.")
parser.add_argument("--wander_y_max", type=float, default=0.4, help="Maximum wander lateral velocity command.")
parser.add_argument("--wander_yaw_min", type=float, default=-0.8, help="Minimum wander yaw velocity command.")
parser.add_argument("--wander_yaw_max", type=float, default=0.8, help="Maximum wander yaw velocity command.")
parser.add_argument("--wander_resample_min", type=float, default=3.0, help="Minimum wander command resample time.")
parser.add_argument("--wander_resample_max", type=float, default=5.0, help="Maximum wander command resample time.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import ldm_quad.tasks  # noqa: F401
from ldm_quad.mbrl import DynamicsEnsemble, LatentWorldModel, StateWorldModel, build_planner, load_policy_prior


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_obs(obs: object, device: torch.device) -> torch.Tensor:
    if isinstance(obs, dict):
        if "policy" in obs:
            obs = obs["policy"]
        elif "obs" in obs and isinstance(obs["obs"], dict) and "policy" in obs["obs"]:
            obs = obs["obs"]["policy"]
        else:
            raise KeyError(f"Unsupported observation dictionary keys: {list(obs.keys())}")
    if not isinstance(obs, torch.Tensor):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    return obs.float().view(obs.shape[0], -1)


def parse_index_list(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    value = str(value)
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def to_tensor(x: object, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def get_action_bounds(action_space: gym.Space, device: torch.device, action_dim: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return -torch.ones(action_dim, device=device), torch.ones(action_dim, device=device), False

    low_t = torch.as_tensor(low, dtype=torch.float32, device=device)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device)
    if low_t.ndim > 1:
        low_t = low_t[0]
    if high_t.ndim > 1:
        high_t = high_t[0]
    low_t = low_t.view(-1)
    high_t = high_t.view(-1)
    bounds_finite = bool(torch.isfinite(low_t).all().item() and torch.isfinite(high_t).all().item())
    low_t = torch.where(torch.isfinite(low_t), low_t, -torch.ones_like(low_t))
    high_t = torch.where(torch.isfinite(high_t), high_t, torch.ones_like(high_t))
    return low_t, high_t, bounds_finite


def infer_play_task(train_task: str | None) -> str:
    if train_task == "Flat-Unitree-Go2-train-v0":
        return "Random-Agent-Unitree-Go2-Play-v0"
    return train_task or "Random-Agent-Unitree-Go2-Play-v0"


def resolve_velocity_command(checkpoint_args: dict) -> tuple[float | None, float | None, float | None]:
    command_x = args_cli.command_x if args_cli.command_x is not None else checkpoint_args.get("command_x")
    command_y = args_cli.command_y if args_cli.command_y is not None else checkpoint_args.get("command_y")
    command_yaw = args_cli.command_yaw if args_cli.command_yaw is not None else checkpoint_args.get("command_yaw")
    return command_x, command_y, command_yaw


def apply_fixed_velocity_command(env_cfg: object, checkpoint_args: dict) -> tuple[float | None, float | None, float | None]:
    command_x, command_y, command_yaw = resolve_velocity_command(checkpoint_args)
    if not args_cli.wander and command_x is None and command_y is None and command_yaw is None:
        return None, None, None

    command_x = command_x or 0.0
    command_y = command_y or 0.0
    command_yaw = command_yaw or 0.0

    command_cfg = env_cfg.commands.base_velocity
    if args_cli.wander:
        command_cfg.ranges.lin_vel_x = (args_cli.wander_x_min, args_cli.wander_x_max)
        command_cfg.ranges.lin_vel_y = (args_cli.wander_y_min, args_cli.wander_y_max)
        command_cfg.ranges.ang_vel_z = (args_cli.wander_yaw_min, args_cli.wander_yaw_max)
        command_cfg.resampling_time_range = (args_cli.wander_resample_min, args_cli.wander_resample_max)
    else:
        command_cfg.ranges.lin_vel_x = (command_x, command_x)
        command_cfg.ranges.lin_vel_y = (command_y, command_y)
        command_cfg.ranges.ang_vel_z = (command_yaw, command_yaw)
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0
    command_cfg.debug_vis = True
    return command_x, command_y, command_yaw


def _set_physics_material_friction(env_cfg: object, static_friction: float, dynamic_friction: float) -> None:
    material = getattr(getattr(env_cfg, "scene", None), "terrain", None).physics_material
    material.static_friction = static_friction
    material.dynamic_friction = dynamic_friction
    if getattr(env_cfg, "sim", None) is not None:
        env_cfg.sim.physics_material = material


def apply_mismatch(env_cfg: object, mismatch: str) -> None:
    if mismatch == "nominal":
        return

    if mismatch == "low_friction":
        _set_physics_material_friction(env_cfg, args_cli.mismatch_friction, args_cli.mismatch_friction)
        if getattr(env_cfg.events, "physics_material", None) is not None:
            env_cfg.events.physics_material.params["static_friction_range"] = (
                args_cli.mismatch_friction,
                args_cli.mismatch_friction,
            )
            env_cfg.events.physics_material.params["dynamic_friction_range"] = (
                args_cli.mismatch_friction,
                args_cli.mismatch_friction,
            )
        return

    if mismatch == "mass":
        env_cfg.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (args_cli.mismatch_mass_add, args_cli.mismatch_mass_add),
                "operation": "add",
            },
        )
        return

    if mismatch == "motor_weakness":
        if args_cli.mismatch_mode == "delayed":
            return
        env_cfg.actions.joint_pos.scale = float(env_cfg.actions.joint_pos.scale) * args_cli.mismatch_motor_scale
        return

    if mismatch == "rough":
        env_cfg.scene.terrain.terrain_type = "generator"
        env_cfg.scene.terrain.terrain_generator = deepcopy(ROUGH_TERRAINS_CFG)
        env_cfg.scene.terrain.max_init_terrain_level = None
        env_cfg.curriculum.terrain_levels = None
        if env_cfg.scene.terrain.terrain_generator is not None:
            env_cfg.scene.terrain.terrain_generator.num_rows = 5
            env_cfg.scene.terrain.terrain_generator.num_cols = 5
            env_cfg.scene.terrain.terrain_generator.curriculum = False
            if "boxes" in env_cfg.scene.terrain.terrain_generator.sub_terrains:
                env_cfg.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (
                    0.01,
                    args_cli.mismatch_rough_height,
                )
            if "random_rough" in env_cfg.scene.terrain.terrain_generator.sub_terrains:
                env_cfg.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (
                    0.005,
                    args_cli.mismatch_rough_noise,
                )
                env_cfg.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        return

    if mismatch == "push":
        if args_cli.mismatch_mode == "delayed":
            return
        speed = args_cli.mismatch_push_velocity
        interval = args_cli.mismatch_push_interval
        env_cfg.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(interval, interval),
            params={"velocity_range": {"x": (-speed, speed), "y": (-speed, speed)}},
        )
        return

    raise ValueError(f"Unsupported mismatch: {mismatch}")


def build_showcase_groups(num_envs: int, group_names: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    group_count = max(1, len(group_names))
    base_size = num_envs // group_count
    remainder = num_envs % group_count
    groups: dict[str, torch.Tensor] = {}
    start = 0
    for group_idx, name in enumerate(group_names):
        size = base_size + (1 if group_idx < remainder else 0)
        end = start + size
        groups[name] = torch.arange(start, end, dtype=torch.long, device=device)
        start = end
    return groups


def apply_showcase_action_mismatches(actions: torch.Tensor, groups: dict[str, torch.Tensor]) -> torch.Tensor:
    weak_ids = groups.get("motor_weakness")
    if weak_ids is not None and weak_ids.numel() > 0:
        actions = actions.clone()
        actions[weak_ids] = actions[weak_ids] * args_cli.mismatch_motor_scale
    return actions


def runtime_motor_scale(steps: int) -> float:
    if args_cli.mismatch_mode != "delayed" or args_cli.mismatch != "motor_weakness":
        return 1.0
    if steps < args_cli.mismatch_start_step:
        return 1.0
    if args_cli.mismatch_ramp_steps <= 0:
        return args_cli.mismatch_motor_scale
    alpha = min(1.0, (steps - args_cli.mismatch_start_step) / max(1, args_cli.mismatch_ramp_steps))
    return (1.0 - alpha) + alpha * args_cli.mismatch_motor_scale


def apply_runtime_motor_weakness(actions: torch.Tensor, steps: int) -> torch.Tensor:
    scale = runtime_motor_scale(steps)
    if scale == 1.0:
        return actions
    return actions * scale


def write_push_velocity(env: gym.Env, env_ids: torch.Tensor) -> bool:
    if env_ids.numel() == 0:
        return True
    try:
        robot = env.unwrapped.scene["robot"]
        root_velocity = torch.zeros((env_ids.numel(), 6), device=env_ids.device)
        root_velocity[:, 0:2] = (
            2.0 * torch.rand((env_ids.numel(), 2), device=env_ids.device) - 1.0
        ) * args_cli.mismatch_push_velocity
        robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        return True
    except Exception as exc:
        print(f"[WARN] Runtime push unavailable in this Isaac Lab build: {exc}", flush=True)
        return False


def apply_showcase_push(env: gym.Env, groups: dict[str, torch.Tensor], steps: int, dt: float) -> bool:
    push_ids = groups.get("push")
    if push_ids is None or push_ids.numel() == 0:
        return True

    interval_steps = max(1, int(round(args_cli.mismatch_push_interval / max(dt, 1e-6))))
    if steps == 0 or steps % interval_steps != 0:
        return True

    return write_push_velocity(env, push_ids)


def apply_delayed_push(env: gym.Env, steps: int, dt: float, device: torch.device, num_envs: int) -> bool:
    if args_cli.mismatch_mode != "delayed" or args_cli.mismatch != "push":
        return True
    if steps < args_cli.mismatch_start_step:
        return True

    interval_steps = max(1, int(round(args_cli.mismatch_push_interval / max(dt, 1e-6))))
    if (steps - args_cli.mismatch_start_step) % interval_steps != 0:
        return True

    env_ids = torch.arange(num_envs, dtype=torch.long, device=device)
    return write_push_velocity(env, env_ids)


def print_showcase_groups(groups: dict[str, torch.Tensor]) -> None:
    print("[SHOWCASE] Runtime mismatch groups:", flush=True)
    for name, env_ids in groups.items():
        if env_ids.numel() == 0:
            continue
        print(
            f"[SHOWCASE] {name}: env_ids={int(env_ids[0].item())}-{int(env_ids[-1].item())} count={env_ids.numel()}",
            flush=True,
        )


def video_run_name() -> str:
    if args_cli.showcase:
        return "showcase_" + "_".join(args_cli.showcase_groups)
    if args_cli.mismatch_mode == "delayed":
        return f"mismatch_{args_cli.mismatch}_delayed_start{args_cli.mismatch_start_step}"
    return f"mismatch_{args_cli.mismatch}"


def main() -> None:
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})

    if args_cli.mismatch_mode == "delayed" and args_cli.mismatch not in ("motor_weakness", "push"):
        print(
            f"[WARN] Delayed mismatch is only runtime-safe for motor_weakness and push. "
            f"{args_cli.mismatch} will still be applied from reset.",
            flush=True,
        )
    if args_cli.showcase and args_cli.mismatch_mode == "delayed":
        print(
            "[WARN] --showcase uses immediate per-group runtime mismatches; "
            "--mismatch_mode delayed only affects single-mismatch play.",
            flush=True,
        )

    seed = args_cli.seed if args_cli.seed is not None else checkpoint_args.get("seed", 42)
    set_seed(seed)

    task_name = args_cli.task or infer_play_task(checkpoint_args.get("task"))
    use_fabric = not (args_cli.disable_fabric or checkpoint_args.get("disable_fabric", False))

    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=use_fabric,
    )
    env_cfg.seed = seed
    command_x, command_y, command_yaw = apply_fixed_velocity_command(env_cfg, checkpoint_args)
    apply_mismatch(env_cfg, args_cli.mismatch)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        video_folder = os.path.join(run_dir, "videos", video_run_name())
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording video to: {video_folder}")

    device = torch.device(env.unwrapped.device)
    obs, _ = env.reset()
    obs = flatten_obs(obs, device)
    action_shape = env.action_space.shape
    action_dim = int(action_shape[-1]) if len(action_shape) > 0 else int(np.prod(action_shape))
    action_low, action_high, action_bounds_finite = get_action_bounds(env.action_space, device, action_dim)
    prior_checkpoint = args_cli.prior_checkpoint or checkpoint_args.get("prior_checkpoint")
    action_prior = None
    if prior_checkpoint:
        prior_type = args_cli.prior_type or checkpoint_args.get("prior_type", "auto")
        action_prior = load_policy_prior(
            env=env,
            checkpoint_path=prior_checkpoint,
            task_name=args_cli.prior_task or checkpoint_args.get("prior_task") or checkpoint_args.get("task", task_name),
            prior_type=prior_type,
            algorithm=args_cli.prior_algorithm or checkpoint_args.get("prior_algorithm", "PPO"),
            agent_cfg_entry_point=args_cli.prior_agent or checkpoint_args.get("prior_agent"),
            obs_adapter=args_cli.prior_obs_adapter or checkpoint_args.get("prior_obs_adapter", "go2_rsl_rough"),
            action_adapter=args_cli.prior_action_adapter or checkpoint_args.get("prior_action_adapter", "none"),
        )
        print(f"[INFO] Loaded locomotion prior ({prior_type}): {os.path.abspath(prior_checkpoint)}")

    planner_name = checkpoint_args.get("planner", "mppi")
    planner = None
    if not args_cli.prior_only:
        model_type = checkpoint_args.get("model_type", "dynamics")
        if model_type == "latent":
            model = LatentWorldModel(
                obs_dim=obs.shape[-1],
                action_dim=action_dim,
                latent_dim=checkpoint_args.get("latent_dim", 128),
                hidden_dim=checkpoint_args["hidden_dim"],
                depth=checkpoint_args["model_depth"],
                num_q=checkpoint_args.get("num_q", 5),
                discount=checkpoint_args["discount"],
                tau=checkpoint_args.get("target_tau", 0.01),
                rho=checkpoint_args.get("rho", 0.5),
                entropy_coef=checkpoint_args.get("entropy_coef", 1e-4),
                num_bins=checkpoint_args.get("num_bins", 101),
                vmin=checkpoint_args.get("vmin", -10.0),
                vmax=checkpoint_args.get("vmax", 10.0),
                simnorm_dim=checkpoint_args.get("simnorm_dim", 8),
                q_dropout=checkpoint_args.get("q_dropout", 0.01),
                physical_feature_indices=parse_index_list(checkpoint_args.get("latent_physical_indices", "")),
            ).to(device)
        elif model_type == "state":
            model = StateWorldModel(
                obs_dim=obs.shape[-1],
                action_dim=action_dim,
                ensemble_size=checkpoint_args["ensemble_size"],
                hidden_dim=checkpoint_args["hidden_dim"],
                depth=checkpoint_args["model_depth"],
                discount=checkpoint_args["discount"],
                tau=checkpoint_args.get("target_tau", 0.01),
                rho=checkpoint_args.get("rho", 0.5),
                entropy_coef=checkpoint_args.get("entropy_coef", 1e-4),
                num_bins=checkpoint_args.get("num_bins", 101),
                vmin=checkpoint_args.get("vmin", -10.0),
                vmax=checkpoint_args.get("vmax", 10.0),
                value_coef=checkpoint_args.get("state_value_coef", checkpoint_args.get("value_coef", 0.1)),
                reward_coef=checkpoint_args.get("reward_coef", 0.1),
                continue_coef=checkpoint_args.get("continue_coef", 1.0),
            ).to(device)
        else:
            model = DynamicsEnsemble(
                obs_dim=obs.shape[-1],
                action_dim=action_dim,
                ensemble_size=checkpoint_args["ensemble_size"],
                hidden_dim=checkpoint_args["hidden_dim"],
                depth=checkpoint_args["model_depth"],
            ).to(device)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        if hasattr(model, "sync_detached_qs"):
            model.sync_detached_qs()
        if missing_keys:
            preview = ", ".join(missing_keys[:4])
            suffix = "..." if len(missing_keys) > 4 else ""
            print(f"[INFO] Checkpoint missing {len(missing_keys)} model keys initialized from current code: {preview}{suffix}")
        if unexpected_keys:
            preview = ", ".join(unexpected_keys[:4])
            suffix = "..." if len(unexpected_keys) > 4 else ""
            print(f"[INFO] Checkpoint has {len(unexpected_keys)} unused model keys: {preview}{suffix}")
        model.eval()

        planner = build_planner(
            planner_name=planner_name,
            model=model,
            action_low=action_low,
            action_high=action_high,
            horizon=checkpoint_args["horizon"],
            candidates=checkpoint_args["candidates"],
            elites=checkpoint_args.get("elites", 32),
            iterations=checkpoint_args.get("planner_iterations", checkpoint_args.get("cem_iterations", 4)),
            discount=checkpoint_args["discount"],
            temperature=checkpoint_args.get("planner_temperature", 0.5),
            lambda_=checkpoint_args.get("mppi_lambda", 1.0),
            min_std=args_cli.min_std if args_cli.min_std is not None else checkpoint_args.get("min_std", 0.05),
            max_std=args_cli.max_std if args_cli.max_std is not None else checkpoint_args.get("max_std", 2.0),
            num_pi_trajs=(
                args_cli.num_pi_trajs if args_cli.num_pi_trajs is not None else checkpoint_args.get("num_pi_trajs", 24)
            ),
            action_noise=False,
            use_continue_model=(
                args_cli.planner_use_continue_model
                if args_cli.planner_use_continue_model is not None
                else checkpoint_args.get("planner_use_continue_model", False)
            ),
            hard_continue_model=(
                args_cli.planner_hard_continue_model
                if args_cli.planner_hard_continue_model is not None
                else checkpoint_args.get("planner_hard_continue_model", False)
            ),
            continue_threshold=(
                args_cli.planner_continue_threshold
                if args_cli.planner_continue_threshold is not None
                else checkpoint_args.get("planner_continue_threshold", 0.5)
            ),
            action_spline_knots=checkpoint_args.get("action_spline_knots", 0),
            action_prior=action_prior,
            prior_residual_scale=(
                args_cli.prior_residual_scale
                if args_cli.prior_residual_scale is not None
                else checkpoint_args.get("prior_residual_scale", 0.3)
            ),
            prior_residual_penalty=(
                args_cli.prior_residual_penalty
                if args_cli.prior_residual_penalty is not None
                else checkpoint_args.get("prior_residual_penalty", 0.0)
            ),
            prior_acceptance_margin=(
                args_cli.prior_acceptance_margin
                if args_cli.prior_acceptance_margin is not None
                else checkpoint_args.get("prior_acceptance_margin", 0.0)
            ),
            prior_fallback=(
                args_cli.prior_fallback
                if args_cli.prior_fallback is not None
                else checkpoint_args.get("prior_fallback", True)
            ),
            prior_candidate_fraction=(
                args_cli.prior_candidate_fraction
                if args_cli.prior_candidate_fraction is not None
                else checkpoint_args.get("prior_candidate_fraction", 0.1)
            ),
            prior_candidate_noise=(
                args_cli.prior_candidate_noise
                if args_cli.prior_candidate_noise is not None
                else checkpoint_args.get("prior_candidate_noise", 0.02)
            ),
            prior_command_candidate_fraction=(
                args_cli.prior_command_candidate_fraction
                if args_cli.prior_command_candidate_fraction is not None
                else checkpoint_args.get("prior_command_candidate_fraction", 0.0)
            ),
            prior_command_noise=(
                args_cli.prior_command_noise
                if args_cli.prior_command_noise is not None
                else checkpoint_args.get("prior_command_noise", 0.1)
            ),
            prior_command_start=(
                args_cli.prior_command_start
                if args_cli.prior_command_start is not None
                else checkpoint_args.get("prior_command_start", -1)
            ),
            prior_command_dim=(
                args_cli.prior_command_dim
                if args_cli.prior_command_dim is not None
                else checkpoint_args.get("prior_command_dim", 3)
            ),
            prior_control_mode=(
                args_cli.prior_control_mode
                if args_cli.prior_control_mode is not None
                else checkpoint_args.get("prior_control_mode", "residual")
            ),
            action_bounds_finite=checkpoint_args.get("action_bounds_finite", action_bounds_finite),
            planner_velocity_objective_weight=(
                args_cli.planner_velocity_objective_weight
                if args_cli.planner_velocity_objective_weight is not None
                else checkpoint_args.get("planner_velocity_objective_weight", 0.0)
            ),
            planner_velocity_target_x=(
                args_cli.planner_velocity_target_x
                if args_cli.planner_velocity_target_x is not None
                else checkpoint_args.get("planner_velocity_target_x", 0.0)
            ),
            planner_velocity_target_y=(
                args_cli.planner_velocity_target_y
                if args_cli.planner_velocity_target_y is not None
                else checkpoint_args.get("planner_velocity_target_y", 0.0)
            ),
            planner_velocity_target_yaw=(
                args_cli.planner_velocity_target_yaw
                if args_cli.planner_velocity_target_yaw is not None
                else checkpoint_args.get("planner_velocity_target_yaw", 0.0)
            ),
            use_best_candidate=(
                args_cli.planner_use_best_candidate
                if args_cli.planner_use_best_candidate is not None
                else checkpoint_args.get("planner_use_best_candidate", False)
            ),
            terminal_value=model_type == "state" and checkpoint_args.get("state_terminal_value", True),
            disagreement_penalty=checkpoint_args.get("state_disagreement_penalty", 0.0) if model_type == "state" else 0.0,
            model_policy_candidate_count=(
                (args_cli.num_pi_trajs if args_cli.num_pi_trajs is not None else checkpoint_args.get("num_pi_trajs", 24))
                if model_type == "state"
                else 0
            ),
        )

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    episode_returns = torch.zeros(obs.shape[0], dtype=torch.float32, device=device)
    episode_lengths = torch.zeros(obs.shape[0], dtype=torch.float32, device=device)
    showcase_groups = build_showcase_groups(obs.shape[0], args_cli.showcase_groups, device) if args_cli.showcase else {}
    showcase_completed_returns: dict[str, list[float]] = {name: [] for name in showcase_groups}
    showcase_completed_lengths: dict[str, list[float]] = {name: [] for name in showcase_groups}
    showcase_push_available = True

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(
        f"[INFO] Evaluating task={task_name} mismatch={args_cli.mismatch} "
        f"mismatch_mode={args_cli.mismatch_mode} num_envs={args_cli.num_envs} "
        f"num_episodes={args_cli.num_episodes}"
    )
    if args_cli.mismatch_mode == "delayed" and args_cli.mismatch in ("motor_weakness", "push"):
        print(
            f"[INFO] Delayed mismatch starts at step {args_cli.mismatch_start_step} "
            f"ramp_steps={args_cli.mismatch_ramp_steps}",
            flush=True,
        )
    print(f"[INFO] Planner={'prior_only' if args_cli.prior_only else planner_name}")
    if args_cli.showcase:
        print_showcase_groups(showcase_groups)
    if args_cli.wander:
        print(
            "[INFO] Velocity command=wander "
            f"x=({args_cli.wander_x_min:.3f}, {args_cli.wander_x_max:.3f}) "
            f"y=({args_cli.wander_y_min:.3f}, {args_cli.wander_y_max:.3f}) "
            f"yaw=({args_cli.wander_yaw_min:.3f}, {args_cli.wander_yaw_max:.3f})"
        )
    elif command_x is not None:
        print(f"[INFO] Velocity command=({command_x:.3f}, {command_y:.3f}, {command_yaw:.3f})")

    steps = 0
    delayed_mismatch_announced = False
    delayed_push_available = True
    while (
        simulation_app is not None
        and simulation_app.is_running()
        and steps < args_cli.max_steps
        and (args_cli.video or len(completed_returns) < args_cli.num_episodes)
    ):
        start_time = time.time()

        with torch.inference_mode():
            if args_cli.prior_only:
                if action_prior is None:
                    raise RuntimeError("--prior_only requires a prior checkpoint in the MBRL checkpoint or --prior_checkpoint.")
                actions = action_prior(obs)
            else:
                actions = planner.plan(obs, eval_mode=True, t0=steps == 0)
            if args_cli.showcase:
                actions = apply_showcase_action_mismatches(actions, showcase_groups)
                if showcase_push_available:
                    showcase_push_available = apply_showcase_push(env, showcase_groups, steps, dt)
            else:
                actions = apply_runtime_motor_weakness(actions, steps)
                if delayed_push_available:
                    delayed_push_available = apply_delayed_push(env, steps, dt, device, obs.shape[0])
            if (
                args_cli.mismatch_mode == "delayed"
                and args_cli.mismatch in ("motor_weakness", "push")
                and not delayed_mismatch_announced
                and steps >= args_cli.mismatch_start_step
            ):
                print(f"[INFO] Delayed {args_cli.mismatch} active at step {steps}", flush=True)
                delayed_mismatch_announced = True
            if args_cli.debug_actions and steps % 100 == 0:
                if obs.shape[-1] == 45:
                    command_obs = obs[:, 6:9]
                elif obs.shape[-1] >= 12:
                    command_obs = obs[:, 9:12]
                else:
                    command_obs = None
                print(
                    "[DEBUG] "
                    f"step={steps} "
                    f"obs_dim={obs.shape[-1]} "
                    f"action_abs_mean={actions.abs().mean().item():.4f} "
                    f"action_abs_max={actions.abs().max().item():.4f} "
                    f"action_first={actions[0].detach().cpu().tolist()} "
                    f"command_obs={command_obs[0].detach().cpu().tolist() if command_obs is not None else None}"
                )
            next_obs_raw, rewards, terminated, truncated, _ = env.step(actions)
            next_obs = flatten_obs(next_obs_raw, device)
            rewards = to_tensor(rewards, device).float().view(-1)
            done = to_tensor(terminated, device).bool().view(-1) | to_tensor(truncated, device).bool().view(-1)

            episode_returns += rewards
            episode_lengths += 1

            if done.any():
                done_mask = done
                done_returns = episode_returns[done_mask].detach().cpu().tolist()
                done_lengths = episode_lengths[done_mask].detach().cpu().tolist()
                completed_returns.extend(done_returns)
                completed_lengths.extend(done_lengths)
                if args_cli.showcase:
                    for group_name, group_ids in showcase_groups.items():
                        if group_ids.numel() == 0:
                            continue
                        group_done = done[group_ids]
                        if group_done.any():
                            showcase_completed_returns[group_name].extend(
                                episode_returns[group_ids][group_done].detach().cpu().tolist()
                            )
                            showcase_completed_lengths[group_name].extend(
                                episode_lengths[group_ids][group_done].detach().cpu().tolist()
                            )
                episode_returns[done_mask] = 0.0
                episode_lengths[done_mask] = 0.0
                if planner is not None:
                    planner.reset(done_mask)

            obs = next_obs
            steps += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        if args_cli.video and steps >= args_cli.video_length:
            break

    eval_returns = completed_returns[: args_cli.num_episodes]
    eval_lengths = completed_lengths[: args_cli.num_episodes]
    print(f"[EVAL] steps={steps} completed_episodes={len(eval_returns)}")
    if eval_returns:
        print(
            "[EVAL] "
            f"mean_return={float(np.mean(eval_returns)):.3f} "
            f"std_return={float(np.std(eval_returns)):.3f} "
            f"mean_length={float(np.mean(eval_lengths)):.2f}"
        )
    else:
        print("[EVAL] No episodes completed before the evaluation budget ended.")

    if args_cli.showcase:
        print("[SHOWCASE] Group metrics")
        for group_name in showcase_groups:
            returns = showcase_completed_returns[group_name]
            lengths = showcase_completed_lengths[group_name]
            if returns:
                print(
                    "[SHOWCASE] "
                    f"{group_name}: episodes={len(returns)} "
                    f"mean_return={float(np.mean(returns)):.3f} "
                    f"std_return={float(np.std(returns)):.3f} "
                    f"mean_length={float(np.mean(lengths)):.2f}"
                )
            else:
                print(f"[SHOWCASE] {group_name}: no completed episodes")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
