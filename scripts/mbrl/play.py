#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play or evaluate an MBRL checkpoint on the Go2 walking task."""

import argparse
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

import ldm_quad.tasks  # noqa: F401
from ldm_quad.mbrl import DynamicsEnsemble, build_planner, load_policy_prior


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


def main() -> None:
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})

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

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        video_folder = os.path.join(run_dir, "videos", "mbrl_play")
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
        model = DynamicsEnsemble(
            obs_dim=obs.shape[-1],
            action_dim=action_dim,
            ensemble_size=checkpoint_args["ensemble_size"],
            hidden_dim=checkpoint_args["hidden_dim"],
            depth=checkpoint_args["model_depth"],
        ).to(device)
        model.load_state_dict(checkpoint["model"])
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
        )

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    episode_returns = torch.zeros(obs.shape[0], dtype=torch.float32, device=device)
    episode_lengths = torch.zeros(obs.shape[0], dtype=torch.float32, device=device)

    print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    print(f"[INFO] Evaluating task={task_name} num_envs={args_cli.num_envs} num_episodes={args_cli.num_episodes}")
    print(f"[INFO] Planner={'prior_only' if args_cli.prior_only else planner_name}")
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
    while (
        simulation_app is not None
        and simulation_app.is_running()
        and steps < args_cli.max_steps
        and len(completed_returns) < args_cli.num_episodes
    ):
        start_time = time.time()

        with torch.inference_mode():
            if args_cli.prior_only:
                if action_prior is None:
                    raise RuntimeError("--prior_only requires a prior checkpoint in the MBRL checkpoint or --prior_checkpoint.")
                actions = action_prior(obs)
            else:
                actions = planner.plan(obs)
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

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
