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
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "source", "ldm_quad")
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
parser.add_argument(
    "--planner_start_steps",
    type=int,
    default=None,
    help="Earliest step at which residual planning may control the env. Defaults to --seed_steps.",
)
parser.add_argument(
    "--planner_min_length_fraction",
    type=float,
    default=0.9,
    help="Require recent completed episodes to reach this fraction of max length before enabling the planner.",
)
parser.add_argument(
    "--planner_recovery_steps",
    type=int,
    default=1000,
    help="Number of pure-prior recovery steps after planner-controlled episodes get too short.",
)
parser.add_argument(
    "--planner_recent_episodes",
    type=int,
    default=100,
    help="Number of recent completed episodes used for planner gating.",
)
parser.add_argument("--train_steps", type=int, default=400, help="Number of environment interaction steps.")
parser.add_argument("--updates_per_step", type=int, default=8, help="Model updates after each environment step.")
parser.add_argument("--batch_size", type=int, default=4096, help="Replay batch size.")
parser.add_argument(
    "--model_type",
    type=str,
    default="latent",
    choices=["latent", "dynamics"],
    help="Use a TD-MPC-style latent world model or the legacy observation-delta dynamics ensemble.",
)
parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension for --model_type latent.")
parser.add_argument("--num_q", type=int, default=2, help="Number of Q heads for --model_type latent.")
parser.add_argument("--target_tau", type=float, default=0.01, help="Soft-update rate for latent target encoder/Q heads.")
parser.add_argument("--rho", type=float, default=0.5, help="Temporal loss discount for latent rollout losses.")
parser.add_argument("--entropy_coef", type=float, default=1e-4, help="Entropy coefficient for the latent stochastic actor.")
parser.add_argument("--hidden_dim", type=int, default=512, help="Model hidden dimension.")
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
    "--planner_use_best_candidate",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use the best evaluated candidate as the final MPPI plan instead of the weighted candidate mean.",
)
parser.add_argument("--planner_velocity_objective_weight", type=float, default=0.0, help="Planner-only velocity objective weight. Disabled when 0.")
parser.add_argument("--planner_velocity_target_x", type=float, default=0.0, help="Planner-only target body x velocity.")
parser.add_argument("--planner_velocity_target_y", type=float, default=0.0, help="Planner-only target body y velocity.")
parser.add_argument("--planner_velocity_target_yaw", type=float, default=0.0, help="Planner-only target yaw rate.")
parser.add_argument(
    "--action_spline_knots",
    type=int,
    default=0,
    help="Sample this many cubic action-spline knots instead of one action per horizon step. Disabled when <=1.",
)
parser.add_argument("--prior_checkpoint", type=str, default=None, help="Policy checkpoint used as locomotion prior.")
parser.add_argument(
    "--prior_type",
    type=str,
    default="auto",
    choices=["auto", "skrl", "torchscript", "rsl_jit"],
    help="Locomotion prior format. auto tries TorchScript first, then skrl.",
)
parser.add_argument(
    "--prior_obs_adapter",
    type=str,
    default="go2_rsl_rough",
    help="Observation adapter for TorchScript priors.",
)
parser.add_argument(
    "--prior_action_adapter",
    type=str,
    default="none",
    help="Action adapter for TorchScript priors.",
)
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
parser.add_argument(
    "--prior_control_mode",
    type=str,
    default="residual",
    choices=["residual", "full_action"],
    help="How the planner uses the prior: residual actions around it, or full-action MPPI warm-started by it.",
)
parser.add_argument("--prior_residual_scale", type=float, default=0.3, help="Max residual action magnitude around the prior.")
parser.add_argument("--prior_residual_penalty", type=float, default=0.0, help="Planning penalty on squared residual actions.")
parser.add_argument(
    "--prior_candidate_fraction",
    type=float,
    default=0.1,
    help="Fraction of planner candidates reserved for pure or near-prior residual rollouts.",
)
parser.add_argument(
    "--prior_candidate_noise",
    type=float,
    default=0.02,
    help="Residual noise std for extra prior-centered candidates. Candidate 0 is always the exact prior.",
)
parser.add_argument(
    "--prior_command_candidate_fraction",
    type=float,
    default=0.0,
    help="Fraction of full-action planner candidates generated by rolling the prior with perturbed commands.",
)
parser.add_argument("--prior_command_noise", type=float, default=0.1, help="Gaussian noise std added to prior command candidates.")
parser.add_argument(
    "--prior_command_start",
    type=int,
    default=-1,
    help="Start index of velocity command observations for command-perturbed prior candidates. -1 auto-detects 48-D/45-D layouts.",
)
parser.add_argument("--prior_command_dim", type=int, default=3, help="Number of command observation dimensions to perturb.")
parser.add_argument(
    "--prior_fallback",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Fall back to the pure prior unless the residual plan improves predicted return.",
)
parser.add_argument(
    "--prior_acceptance_margin",
    type=float,
    default=0.0,
    help="Required predicted-return improvement before using residual actions over the pure prior.",
)
parser.add_argument("--lr", type=float, default=3e-4, help="Dynamics model learning rate.")
parser.add_argument("--policy_lr", type=float, default=3e-4, help="Latent actor learning rate.")
parser.add_argument("--eval_interval", type=int, default=10, help="Steps between console/log summaries.")
parser.add_argument("--save_interval", type=int, default=50, help="Steps between checkpoints.")
parser.add_argument("--wandb", action="store_true", default=False, help="Log this run to Weights & Biases via TensorBoard sync.")
parser.add_argument("--wandb_project", type=str, default="ldm-quad-mbrl", help="Weights & Biases project name.")
parser.add_argument("--wandb_entity", type=str, default=None, help="Optional Weights & Biases entity or team.")
parser.add_argument("--wandb_name", type=str, default=None, help="Optional Weights & Biases run name. Defaults to the log directory name.")
parser.add_argument(
    "--wandb_mode",
    type=str,
    default="online",
    choices=["online", "offline", "disabled"],
    help="Weights & Biases logging mode.",
)
parser.add_argument("--wandb_alert", action="store_true", default=False, help="Send a W&B alert when training completes.")
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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import ldm_quad.tasks  # noqa: F401
from ldm_quad.mbrl import DynamicsEnsemble, LatentWorldModel, ReplayBuffer, build_planner, load_policy_prior


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
        command_cfg.ranges.lin_vel_x = (args_cli.wander_x_min, args_cli.wander_x_max)
        command_cfg.ranges.lin_vel_y = (args_cli.wander_y_min, args_cli.wander_y_max)
        command_cfg.ranges.ang_vel_z = (args_cli.wander_yaw_min, args_cli.wander_yaw_max)
        command_cfg.resampling_time_range = (args_cli.wander_resample_min, args_cli.wander_resample_max)
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


def init_wandb(log_dir: str) -> object | None:
    if not args_cli.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("[MBRL] W&B logging requested but wandb is not installed. Install with: pip install wandb")
        return None

    try:
        return wandb.init(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            name=args_cli.wandb_name or os.path.basename(log_dir),
            config=vars(args_cli),
            sync_tensorboard=True,
            dir=log_dir,
            mode=args_cli.wandb_mode,
        )
    except Exception as exc:
        print(f"[MBRL] Failed to initialize W&B logging: {exc}")
        return None


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    policy_optimizer: torch.optim.Optimizer | None,
    train_state: TrainState,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "policy_optimizer": policy_optimizer.state_dict() if policy_optimizer is not None else None,
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
    wandb_run = init_wandb(log_dir)
    episode_horizon_steps = infer_episode_horizon_steps(env, env_cfg)

    obs, _ = env.reset()
    obs = flatten_obs(obs, device)
    num_envs = obs.shape[0]
    action_shape = env.action_space.shape
    action_dim = int(action_shape[-1]) if len(action_shape) > 0 else int(np.prod(action_shape))
    obs_dim = obs.shape[-1]

    action_low, action_high, action_bounds_finite = get_action_bounds(env.action_space, device, action_dim)
    setattr(args_cli, "action_bounds_finite", action_bounds_finite)
    action_prior = None
    if args_cli.prior_checkpoint:
        action_prior = load_policy_prior(
            env=env,
            checkpoint_path=args_cli.prior_checkpoint,
            task_name=args_cli.prior_task or args_cli.task,
            prior_type=args_cli.prior_type,
            algorithm=args_cli.prior_algorithm,
            agent_cfg_entry_point=args_cli.prior_agent,
            obs_adapter=args_cli.prior_obs_adapter,
            action_adapter=args_cli.prior_action_adapter,
        )
        print(f"[MBRL] Loaded locomotion prior ({args_cli.prior_type}): {os.path.abspath(args_cli.prior_checkpoint)}")

    replay = ReplayBuffer(args_cli.buffer_capacity, obs_dim=obs_dim, action_dim=action_dim)
    if args_cli.model_type == "latent":
        model = LatentWorldModel(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=args_cli.latent_dim,
            hidden_dim=args_cli.hidden_dim,
            depth=args_cli.model_depth,
            num_q=args_cli.num_q,
            discount=args_cli.discount,
            tau=args_cli.target_tau,
            rho=args_cli.rho,
            entropy_coef=args_cli.entropy_coef,
        ).to(device)
        optimizer = torch.optim.Adam(model.model_parameters(), lr=args_cli.lr)
        policy_optimizer = torch.optim.Adam(model.policy_parameters(), lr=args_cli.policy_lr)
    else:
        model = DynamicsEnsemble(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=args_cli.ensemble_size,
            hidden_dim=args_cli.hidden_dim,
            depth=args_cli.model_depth,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args_cli.lr)
        policy_optimizer = None
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
        prior_acceptance_margin=args_cli.prior_acceptance_margin,
        prior_fallback=args_cli.prior_fallback,
        prior_candidate_fraction=args_cli.prior_candidate_fraction,
        prior_candidate_noise=args_cli.prior_candidate_noise,
        prior_command_candidate_fraction=args_cli.prior_command_candidate_fraction,
        prior_command_noise=args_cli.prior_command_noise,
        prior_command_start=args_cli.prior_command_start,
        prior_command_dim=args_cli.prior_command_dim,
        prior_control_mode=args_cli.prior_control_mode,
        action_bounds_finite=action_bounds_finite,
        planner_velocity_objective_weight=args_cli.planner_velocity_objective_weight,
        planner_velocity_target_x=args_cli.planner_velocity_target_x,
        planner_velocity_target_y=args_cli.planner_velocity_target_y,
        planner_velocity_target_yaw=args_cli.planner_velocity_target_yaw,
        use_best_candidate=args_cli.planner_use_best_candidate,
    )

    train_state = TrainState()
    episode_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
    episode_lengths = torch.zeros(num_envs, dtype=torch.float32, device=device)
    recent_returns: list[float] = []
    recent_lengths: list[float] = []
    recent_step_rewards: list[float] = []
    if args_cli.model_type == "latent":
        latest_losses = {
            "loss": 0.0,
            "consistency_loss": 0.0,
            "reward_loss": 0.0,
            "value_loss": 0.0,
            "continue_loss": 0.0,
            "policy_loss": 0.0,
            "policy_q": 0.0,
            "policy_entropy": 0.0,
        }
    else:
        latest_losses = {"loss": 0.0, "delta_loss": 0.0, "reward_loss": 0.0, "continue_loss": 0.0}
    latest_planner_diagnostics = {
        "planner_candidate_return_mean": 0.0,
        "planner_candidate_return_best": 0.0,
        "planner_candidate_return_std": 0.0,
        "planner_prior_candidate_fraction": 0.0,
        "planner_prior_command_candidate_fraction": 0.0,
        "planner_full_action_mode": 0.0,
        "planner_best_candidate_mode": 0.0,
        "planner_prior_action_norm_mean": 0.0,
        "planner_residual_norm_mean": 0.0,
        "planner_residual_abs_mean": 0.0,
        "planner_final_action_norm_mean": 0.0,
        "planner_residual_to_prior_norm": 0.0,
        "planner_selected_prior_fraction": 0.0,
        "planner_predicted_plan_return_mean": 0.0,
        "planner_predicted_prior_return_mean": 0.0,
        "planner_predicted_return_margin_mean": 0.0,
        "planner_predicted_return_margin_min": 0.0,
        "planner_prior_fallback_fraction": 0.0,
    }
    train_start_time = time.monotonic()
    early_stop_best_metric = float("-inf")
    early_stop_best_step = 0
    early_stop_reason: str | None = None
    planner_disabled_until = 0
    planner_start_steps = args_cli.planner_start_steps if args_cli.planner_start_steps is not None else args_cli.seed_steps
    planner_min_length = args_cli.planner_min_length_fraction * episode_horizon_steps
    planner_active = False

    with open(os.path.join(log_dir, "config.txt"), "w", encoding="utf-8") as f:
        for key, value in sorted(vars(args_cli).items()):
            f.write(f"{key}: {value}\n")

    def app_is_running() -> bool:
        return simulation_app is None or simulation_app.is_running()

    while app_is_running() and train_state.env_steps < args_cli.train_steps:
        with torch.inference_mode():
            recent_length_window = recent_lengths[-args_cli.planner_recent_episodes :]
            recent_mean_length = float(np.mean(recent_length_window)) if recent_length_window else 0.0
            prior_policy_available = action_prior is not None and args_cli.seed_with_prior
            replay_ready = (
                replay.can_sample_sequences(args_cli.batch_size, args_cli.horizon)
                if args_cli.model_type == "latent"
                else len(replay) >= args_cli.batch_size
            )
            planner_ready = (
                train_state.env_steps >= planner_start_steps
                and replay_ready
                and (not prior_policy_available or recent_mean_length >= planner_min_length)
                and train_state.env_steps >= planner_disabled_until
            )
            planner_active = planner_ready

            if not planner_ready:
                latest_planner_diagnostics = dict.fromkeys(latest_planner_diagnostics, 0.0)
                if action_prior is not None and args_cli.seed_with_prior:
                    actions = action_prior(obs)
                    if args_cli.seed_policy_noise > 0.0:
                        actions = actions + args_cli.seed_policy_noise * torch.randn_like(actions)
                    if action_bounds_finite:
                        actions = clip_actions(actions, action_low, action_high)
                else:
                    actions = random_actions(obs.shape[0], action_low, action_high)
            else:
                actions = planner.plan(obs)
                latest_planner_diagnostics.update(planner.last_diagnostics)

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
                planner.reset(done_mask)
                if planner_active:
                    recent_length_window = recent_lengths[-args_cli.planner_recent_episodes :]
                    recent_mean_length = float(np.mean(recent_length_window)) if recent_length_window else 0.0
                    if recent_mean_length < planner_min_length:
                        planner_disabled_until = max(
                            planner_disabled_until,
                            train_state.env_steps + args_cli.planner_recovery_steps,
                        )
                        planner.reset()

            obs = next_obs
            train_state.env_steps += 1

        train_ready = (
            replay.can_sample_sequences(args_cli.batch_size, args_cli.horizon)
            if args_cli.model_type == "latent"
            else len(replay) >= args_cli.batch_size
        )
        if train_ready:
            model.train()
            for _ in range(args_cli.updates_per_step):
                if args_cli.model_type == "latent":
                    batch = replay.sample_sequences(args_cli.batch_size, args_cli.horizon, device=device)
                    loss, metrics, rollout_zs = model.loss(batch)
                else:
                    batch = replay.sample(args_cli.batch_size, device=device)
                    loss, metrics = model.loss(batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.model_parameters() if args_cli.model_type == "latent" else model.parameters(),
                    10.0,
                )
                optimizer.step()
                if args_cli.model_type == "latent":
                    assert policy_optimizer is not None
                    policy_loss, policy_metrics = model.policy_loss(rollout_zs)
                    policy_optimizer.zero_grad(set_to_none=True)
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.policy_parameters(), 10.0)
                    policy_optimizer.step()
                    metrics.update(policy_metrics)
                if hasattr(model, "soft_update_targets"):
                    model.soft_update_targets()
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
                "planner_active": int(planner_active),
                "planner_disabled_until": planner_disabled_until,
                "action_bounds_finite": int(action_bounds_finite),
                "wall_time_s": elapsed_s,
                "steps_per_second": steps_per_second,
                "eta_s": eta_s,
                **latest_planner_diagnostics,
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
            writer.add_scalar("Train / planner_active", int(planner_active), train_state.env_steps)
            writer.add_scalar("Train / planner_disabled_until", planner_disabled_until, train_state.env_steps)
            writer.add_scalar("Train / action_bounds_finite", int(action_bounds_finite), train_state.env_steps)
            writer.add_scalar("Time / wall_time_s", elapsed_s, train_state.env_steps)
            writer.add_scalar("Time / steps_per_second", steps_per_second, train_state.env_steps)
            writer.add_scalar("Time / eta_s", eta_s, train_state.env_steps)
            for diagnostic_name, diagnostic_value in latest_planner_diagnostics.items():
                writer.add_scalar(f"Planner / {diagnostic_name}", diagnostic_value, train_state.env_steps)
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
                f"planner={'on' if planner_active else 'prior'} "
                f"residual_norm={latest_planner_diagnostics['planner_residual_norm_mean']:.3f} "
                f"fallback={latest_planner_diagnostics['planner_prior_fallback_fraction']:.2f} "
                f"model_margin={latest_planner_diagnostics['planner_predicted_return_margin_mean']:.3f} "
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
            save_checkpoint(checkpoint_path, model, optimizer, policy_optimizer, train_state, args_cli)

    final_path = os.path.join(log_dir, "checkpoints", "model_final.pt")
    save_checkpoint(final_path, model, optimizer, policy_optimizer, train_state, args_cli)
    if early_stop_reason is not None:
        early_stop_path = os.path.join(log_dir, "early_stop.txt")
        with open(early_stop_path, "w", encoding="utf-8") as f:
            f.write(f"step: {train_state.env_steps}\n")
            f.write(f"reason: {early_stop_reason}\n")
            f.write(f"checkpoint: {final_path}\n")
    writer.close()
    env.close()
    if wandb_run is not None:
        if args_cli.wandb_alert:
            try:
                wandb_run.alert(
                    title="MBRL training completed",
                    text=(
                        f"Run {os.path.basename(log_dir)} finished at step {train_state.env_steps}. "
                        f"best_mean_return={train_state.best_mean_return:.3f}. "
                        f"checkpoint={final_path}"
                    ),
                )
            except Exception as exc:
                print(f"[MBRL] Failed to send W&B completion alert: {exc}")
        wandb_run.finish()


if __name__ == "__main__":
    try:
        main()
    finally:
        if simulation_app is not None:
            simulation_app.close()
