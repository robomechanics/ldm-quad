# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a model-based controller on the Go2 walking task."""

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "source", "thomas_MBRL")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an MPC-style MBRL agent on an Isaac Lab task.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Flat-Unitree-Go2-train-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for training.")
parser.add_argument("--buffer_capacity", type=int, default=300000, help="Replay buffer capacity.")
parser.add_argument("--seed_steps", type=int, default=50, help="Initial random environment steps before planning.")
parser.add_argument("--train_steps", type=int, default=400, help="Number of environment interaction steps.")
parser.add_argument("--updates_per_step", type=int, default=8, help="Model updates after each environment step.")
parser.add_argument("--batch_size", type=int, default=4096, help="Replay batch size.")
parser.add_argument("--hidden_dim", type=int, default=512, help="Dynamics ensemble hidden dimension.")
parser.add_argument("--model_depth", type=int, default=3, help="Number of hidden layers per ensemble member.")
parser.add_argument("--ensemble_size", type=int, default=5, help="Number of dynamics models in the ensemble.")
parser.add_argument("--planner", type=str, default="mppi", choices=["cem", "mppi"], help="Sampling-based planner.")
parser.add_argument("--horizon", type=int, default=30, help="Planning horizon in environment steps.")
parser.add_argument("--candidates", type=int, default=512, help="Candidate action sequences for planning.")
parser.add_argument("--elites", type=int, default=64, help="Elite sequences kept each CEM iteration.")
parser.add_argument("--planner_iterations", type=int, default=5, help="Planner refinement iterations.")
parser.add_argument("--discount", type=float, default=0.99, help="Planning discount factor.")
parser.add_argument("--planner_temperature", type=float, default=0.35, help="Initial planner exploration scale.")
parser.add_argument("--mppi_lambda", type=float, default=1.0, help="MPPI reward temperature.")
parser.add_argument(
    "--action_spline_knots",
    type=int,
    default=0,
    help="Sample this many cubic action-spline knots instead of one action per horizon step. Disabled when <=1.",
)
parser.add_argument("--prior_checkpoint", type=str, default=None, help="skrl policy checkpoint used as locomotion prior.")
parser.add_argument("--prior_task", type=str, default=None, help="Task used to load the prior policy config.")
parser.add_argument("--prior_algorithm", type=str, default="PPO", help="skrl algorithm for the prior policy.")
parser.add_argument("--prior_agent", type=str, default=None, help="Optional skrl prior agent config entry point.")
parser.add_argument(
    "--seed_with_prior",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use the prior policy plus noise during seed collection when --prior_checkpoint is set.",
)
parser.add_argument("--seed_policy_noise", type=float, default=0.05, help="Gaussian action noise added to prior seed actions.")
parser.add_argument("--prior_residual_scale", type=float, default=0.25, help="Max residual action magnitude around the prior.")
parser.add_argument("--prior_residual_penalty", type=float, default=0.0, help="Planning penalty on squared residual actions.")
parser.add_argument("--lr", type=float, default=3e-4, help="Dynamics model learning rate.")
parser.add_argument("--eval_interval", type=int, default=10, help="Steps between console/log summaries.")
parser.add_argument("--save_interval", type=int, default=50, help="Steps between checkpoints.")
parser.add_argument("--early_stop", action="store_true", default=False, help="Stop training once performance has plateaued.")
parser.add_argument("--early_stop_min_steps", type=int, default=3000, help="Minimum environment steps before early stopping.")
parser.add_argument("--early_stop_patience", type=int, default=1500, help="Steps without metric improvement before stopping.")
parser.add_argument("--early_stop_min_delta", type=float, default=0.05, help="Minimum metric gain counted as improvement.")
parser.add_argument(
    "--early_stop_metric",
    type=str,
    default="mean_return",
    choices=["mean_return", "estimated_return"],
    help="Metric used for early-stop plateau detection.",
)
parser.add_argument(
    "--early_stop_return",
    type=float,
    default=None,
    help="Optional return threshold considered good enough once full-length episodes are reached.",
)
parser.add_argument(
    "--early_stop_length_fraction",
    type=float,
    default=0.98,
    help="Required fraction of max episode length before early stopping can trigger.",
)
parser.add_argument("--command_x", type=float, default=None, help="Fixed forward velocity command in m/s.")
parser.add_argument("--command_y", type=float, default=None, help="Fixed lateral velocity command in m/s.")
parser.add_argument("--command_yaw", type=float, default=None, help="Fixed yaw velocity command in rad/s.")
parser.add_argument(
    "--wander",
    action="store_true",
    default=False,
    help="Sample nonzero movement commands instead of using the default standing/heading mix.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import thomas_MBRL.tasks  # noqa: F401
from thomas_MBRL.mbrl import DynamicsEnsemble, ReplayBuffer, SkrlPolicyPrior, build_planner


@dataclass
class TrainState:
    env_steps: int = 0
    gradient_updates: int = 0
    episodes_finished: int = 0
    best_mean_return: float = float("-inf")


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


def get_action_bounds(action_space: gym.Space, device: torch.device, action_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return -torch.ones(action_dim, device=device), torch.ones(action_dim, device=device)

    low_t = torch.as_tensor(low, dtype=torch.float32, device=device)
    high_t = torch.as_tensor(high, dtype=torch.float32, device=device)
    if low_t.ndim > 1:
        low_t = low_t[0]
    if high_t.ndim > 1:
        high_t = high_t[0]
    low_t = low_t.view(-1)
    high_t = high_t.view(-1)
    low_t = torch.where(torch.isfinite(low_t), low_t, -torch.ones_like(low_t))
    high_t = torch.where(torch.isfinite(high_t), high_t, torch.ones_like(high_t))
    return low_t, high_t


def random_actions(batch_size: int, action_low: torch.Tensor, action_high: torch.Tensor) -> torch.Tensor:
    return action_low + torch.rand((batch_size, action_low.numel()), device=action_low.device) * (action_high - action_low)


def clip_actions(actions: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(actions, action_high.view(1, -1)), action_low.view(1, -1))


def make_log_dir() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join("logs", "mbrl", f"go2_walk_{timestamp}"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    return log_dir


def apply_fixed_velocity_command(env_cfg: object) -> None:
    if not args_cli.wander and args_cli.command_x is None and args_cli.command_y is None and args_cli.command_yaw is None:
        return

    command_cfg = env_cfg.commands.base_velocity
    if args_cli.wander:
        command_cfg.ranges.lin_vel_x = (-0.8, 0.8)
        command_cfg.ranges.lin_vel_y = (-0.4, 0.4)
        command_cfg.ranges.ang_vel_z = (-0.8, 0.8)
        command_cfg.resampling_time_range = (3.0, 5.0)
    else:
        command_cfg.ranges.lin_vel_x = (args_cli.command_x or 0.0, args_cli.command_x or 0.0)
        command_cfg.ranges.lin_vel_y = (args_cli.command_y or 0.0, args_cli.command_y or 0.0)
        command_cfg.ranges.ang_vel_z = (args_cli.command_yaw or 0.0, args_cli.command_yaw or 0.0)
    command_cfg.heading_command = False
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_standing_envs = 0.0


def append_metrics(csv_path: str, row: dict[str, float | int]) -> None:
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def infer_episode_horizon_steps(env: gym.Env, env_cfg: object) -> float:
    """Infer the max episode length so dense step rewards can be shown on a return scale."""
    max_episode_length = getattr(env.unwrapped, "max_episode_length", None)
    if max_episode_length is not None:
        return float(max_episode_length)

    episode_length_s = getattr(env_cfg, "episode_length_s", None)
    sim_cfg = getattr(env_cfg, "sim", None)
    sim_dt = getattr(sim_cfg, "dt", None)
    decimation = getattr(env_cfg, "decimation", None)
    if episode_length_s is not None and sim_dt is not None and decimation is not None:
        return float(episode_length_s) / (float(sim_dt) * float(decimation))

    return 1.0


def save_checkpoint(
    path: str,
    model: DynamicsEnsemble,
    optimizer: torch.optim.Optimizer,
    train_state: TrainState,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_state": asdict(train_state),
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    set_seed(args_cli.seed)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    apply_fixed_velocity_command(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = torch.device(env.unwrapped.device)
    log_dir = make_log_dir()
    metrics_path = os.path.join(log_dir, "metrics.csv")
    writer = SummaryWriter(log_dir=log_dir)
    episode_horizon_steps = infer_episode_horizon_steps(env, env_cfg)

    obs, _ = env.reset()
    obs = flatten_obs(obs, device)
    num_envs = obs.shape[0]
    action_shape = env.action_space.shape
    action_dim = int(action_shape[-1]) if len(action_shape) > 0 else int(np.prod(action_shape))
    obs_dim = obs.shape[-1]

    action_low, action_high = get_action_bounds(env.action_space, device, action_dim)
    action_prior = None
    if args_cli.prior_checkpoint:
        action_prior = SkrlPolicyPrior(
            env=env,
            checkpoint_path=args_cli.prior_checkpoint,
            task_name=args_cli.prior_task or args_cli.task,
            algorithm=args_cli.prior_algorithm,
            agent_cfg_entry_point=args_cli.prior_agent,
        )
        print(f"[MBRL] Loaded locomotion prior: {os.path.abspath(args_cli.prior_checkpoint)}")

    replay = ReplayBuffer(args_cli.buffer_capacity, obs_dim=obs_dim, action_dim=action_dim)
    model = DynamicsEnsemble(
        obs_dim=obs_dim,
        action_dim=action_dim,
        ensemble_size=args_cli.ensemble_size,
        hidden_dim=args_cli.hidden_dim,
        depth=args_cli.model_depth,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_cli.lr)
    model.eval()
    planner = build_planner(
        planner_name=args_cli.planner,
        model=model,
        action_low=action_low,
        action_high=action_high,
        horizon=args_cli.horizon,
        candidates=args_cli.candidates,
        elites=args_cli.elites,
        iterations=args_cli.planner_iterations,
        discount=args_cli.discount,
        temperature=args_cli.planner_temperature,
        lambda_=args_cli.mppi_lambda,
        action_spline_knots=args_cli.action_spline_knots,
        action_prior=action_prior,
        prior_residual_scale=args_cli.prior_residual_scale,
        prior_residual_penalty=args_cli.prior_residual_penalty,
    )

    train_state = TrainState()
    episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.float32, device=device)
    recent_returns: list[float] = []
    recent_lengths: list[float] = []
    recent_step_rewards: list[float] = []
    latest_losses = {"loss": 0.0, "delta_loss": 0.0, "reward_loss": 0.0, "continue_loss": 0.0}
    train_start_time = time.monotonic()
    early_stop_best_metric = float("-inf")
    early_stop_best_step = 0
    early_stop_reason: str | None = None

    with open(os.path.join(log_dir, "config.txt"), "w", encoding="utf-8") as f:
        for key, value in sorted(vars(args_cli).items()):
            f.write(f"{key}: {value}\n")

    def app_is_running() -> bool:
        return simulation_app is None or simulation_app.is_running()

    while app_is_running() and train_state.env_steps < args_cli.train_steps:
        with torch.inference_mode():
            if train_state.env_steps < args_cli.seed_steps or len(replay) < args_cli.batch_size:
                if action_prior is not None and args_cli.seed_with_prior:
                    actions = action_prior(obs)
                    if args_cli.seed_policy_noise > 0.0:
                        actions = actions + args_cli.seed_policy_noise * torch.randn_like(actions)
                    actions = clip_actions(actions, action_low, action_high)
                else:
                    actions = random_actions(obs.shape[0], action_low, action_high)
            else:
                actions = planner.plan(obs)

            next_obs_raw, rewards, terminated, truncated, _ = env.step(actions)
            next_obs = flatten_obs(next_obs_raw, device)
            rewards = to_tensor(rewards, device).float().view(-1, 1)
            terminated = to_tensor(terminated, device).bool().view(-1, 1)
            truncated = to_tensor(truncated, device).bool().view(-1, 1)
            done = terminated | truncated
            continues = (~done).float()

            replay.add_batch(
                obs.detach().cpu(),
                actions.detach().cpu(),
                rewards.detach().cpu(),
                next_obs.detach().cpu(),
                continues.detach().cpu(),
            )

            recent_step_rewards.append(float(rewards.mean().item()))
            episode_returns += rewards.squeeze(-1)
            episode_lengths += 1
            if done.any():
                done_mask = done.squeeze(-1)
                recent_returns.extend(episode_returns[done_mask].detach().cpu().tolist())
                recent_lengths.extend(episode_lengths[done_mask].detach().cpu().tolist())
                train_state.episodes_finished += int(done_mask.sum().item())
                episode_returns[done_mask] = 0.0
                episode_lengths[done_mask] = 0.0

            obs = next_obs
            train_state.env_steps += 1

        if len(replay) >= args_cli.batch_size:
            model.train()
            for _ in range(args_cli.updates_per_step):
                batch = replay.sample(args_cli.batch_size, device=device)
                loss, metrics = model.loss(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                latest_losses = metrics
                train_state.gradient_updates += 1
            model.eval()

        if train_state.env_steps % args_cli.eval_interval == 0:
            mean_return = float(np.mean(recent_returns[-100:])) if recent_returns else 0.0
            mean_length = float(np.mean(recent_lengths[-100:])) if recent_lengths else 0.0
            mean_step_reward_100 = float(np.mean(recent_step_rewards[-100:])) if recent_step_rewards else 0.0
            current_return_mean = float(episode_returns.mean().item())
            current_length_mean = float(episode_lengths.mean().item())
            estimated_return_100 = mean_step_reward_100 * episode_horizon_steps
            train_state.best_mean_return = max(train_state.best_mean_return, mean_return)
            elapsed_s = time.monotonic() - train_start_time
            remaining_steps = max(args_cli.train_steps - train_state.env_steps, 0)
            steps_per_second = train_state.env_steps / max(elapsed_s, 1e-6)
            eta_s = remaining_steps / max(steps_per_second, 1e-6)
            row = {
                "env_steps": train_state.env_steps,
                "gradient_updates": train_state.gradient_updates,
                "episodes_finished": train_state.episodes_finished,
                "buffer_size": len(replay),
                "mean_return_100": mean_return,
                "mean_length_100": mean_length,
                "mean_step_reward_100": mean_step_reward_100,
                "estimated_return_100": estimated_return_100,
                "current_return_mean": current_return_mean,
                "current_length_mean": current_length_mean,
                "best_mean_return": train_state.best_mean_return,
                **latest_losses,
            }
            append_metrics(metrics_path, row)
            writer.add_scalar("Reward / total_reward_mean", estimated_return_100, train_state.env_steps)
            writer.add_scalar("Reward / completed_total_reward_mean", mean_return, train_state.env_steps)
            writer.add_scalar("Reward / step_reward_mean", mean_step_reward_100, train_state.env_steps)
            writer.add_scalar("Episode / episode_length_mean", mean_length, train_state.env_steps)
            writer.add_scalar("Episode / current_episode_return_mean", current_return_mean, train_state.env_steps)
            writer.add_scalar("Episode / current_episode_length_mean", current_length_mean, train_state.env_steps)
            writer.add_scalar("Train / gradient_updates", train_state.gradient_updates, train_state.env_steps)
            writer.add_scalar("Train / buffer_size", len(replay), train_state.env_steps)
            writer.add_scalar("Train / episodes_finished", train_state.episodes_finished, train_state.env_steps)
            for loss_name, loss_value in latest_losses.items():
                writer.add_scalar(f"Loss / {loss_name}", loss_value, train_state.env_steps)
            writer.flush()
            print(
                "[MBRL] "
                f"step={train_state.env_steps} "
                f"buffer={len(replay)} "
                f"episodes={train_state.episodes_finished} "
                f"return100={mean_return:.3f} "
                f"estimated_return100={estimated_return_100:.3f} "
                f"step_reward100={mean_step_reward_100:.3f} "
                f"len100={mean_length:.2f} "
                f"loss={latest_losses['loss']:.4f} "
                f"elapsed={format_duration(elapsed_s)} "
                f"eta={format_duration(eta_s)}"
            )

            if args_cli.early_stop:
                stop_metric = mean_return if args_cli.early_stop_metric == "mean_return" else estimated_return_100
                full_length_ready = mean_length >= args_cli.early_stop_length_fraction * episode_horizon_steps
                min_steps_ready = train_state.env_steps >= args_cli.early_stop_min_steps
                has_completed_episodes = train_state.episodes_finished > 0

                if stop_metric > early_stop_best_metric + args_cli.early_stop_min_delta:
                    early_stop_best_metric = stop_metric
                    early_stop_best_step = train_state.env_steps

                no_improvement_steps = train_state.env_steps - early_stop_best_step
                target_ready = args_cli.early_stop_return is not None and stop_metric >= args_cli.early_stop_return
                plateau_ready = no_improvement_steps >= args_cli.early_stop_patience

                if min_steps_ready and has_completed_episodes and full_length_ready and (target_ready or plateau_ready):
                    if target_ready:
                        early_stop_reason = (
                            f"{args_cli.early_stop_metric}={stop_metric:.3f} reached target "
                            f"{args_cli.early_stop_return:.3f}"
                        )
                    else:
                        early_stop_reason = (
                            f"{args_cli.early_stop_metric} plateaued for {no_improvement_steps} steps "
                            f"(best={early_stop_best_metric:.3f} at step {early_stop_best_step})"
                        )
                    print(f"[MBRL] Early stopping: {early_stop_reason}")
                    break

        if train_state.env_steps % args_cli.save_interval == 0:
            checkpoint_path = os.path.join(log_dir, "checkpoints", f"model_{train_state.env_steps:05d}.pt")
            save_checkpoint(checkpoint_path, model, optimizer, train_state, args_cli)

    final_path = os.path.join(log_dir, "checkpoints", "model_final.pt")
    save_checkpoint(final_path, model, optimizer, train_state, args_cli)
    if early_stop_reason is not None:
        early_stop_path = os.path.join(log_dir, "early_stop.txt")
        with open(early_stop_path, "w", encoding="utf-8") as f:
            f.write(f"step: {train_state.env_steps}\n")
            f.write(f"reason: {early_stop_reason}\n")
            f.write(f"checkpoint: {final_path}\n")
    writer.close()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
