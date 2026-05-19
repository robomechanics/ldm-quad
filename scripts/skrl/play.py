# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

# Example Usage:
# python scripts/skrl/play.py --task=Random-Agent-Unitree-Go2-Play-v0 --num_envs=16 --checkpoint=logs/skrl/go2_flat_ppo/2026-02-10_09-17-30_ppo_torch/checkpoints/best_agent.pt
# python scripts/skrl/play.py --task=Random-Agent-Unitree-Go2-Play-v0 --num_envs=16 --checkpoint=logs/skrl/go2_flat_ppo/2026-03-18_11-18-16_ppo_torch/checkpoints/best_agent.pt


import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "source", "ldm_quad")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_episodes", type=int, default=0, help="Number of completed episodes to evaluate before exiting.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum environment steps to run before exiting.")
parser.add_argument("--command_x", type=float, default=None, help="Fixed forward velocity command in m/s.")
parser.add_argument("--command_y", type=float, default=None, help="Fixed lateral velocity command in m/s.")
parser.add_argument("--command_yaw", type=float, default=None, help="Fixed yaw velocity command in rad/s.")
parser.add_argument(
    "--wander",
    action="store_true",
    default=False,
    help="Sample nonzero walking commands instead of the default heading-command mix.",
)
parser.add_argument("--wander_x_min", type=float, default=-0.8, help="Minimum wander forward velocity command.")
parser.add_argument("--wander_x_max", type=float, default=0.8, help="Maximum wander forward velocity command.")
parser.add_argument("--wander_y_min", type=float, default=-0.4, help="Minimum wander lateral velocity command.")
parser.add_argument("--wander_y_max", type=float, default=0.4, help="Maximum wander lateral velocity command.")
parser.add_argument("--wander_yaw_min", type=float, default=-0.8, help="Minimum wander yaw velocity command.")
parser.add_argument("--wander_yaw_max", type=float, default=0.8, help="Maximum wander yaw velocity command.")
parser.add_argument("--wander_resample_min", type=float, default=3.0, help="Minimum wander command resample time.")
parser.add_argument("--wander_resample_max", type=float, default=5.0, help="Maximum wander command resample time.")
# parser.add_argument(
#     "--use_pretrained_checkpoint",
#     action="store_true",
#     help="Use the pre-trained checkpoint from Nucleus.",
# )
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "SAC", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import random
import time

import gymnasium as gym
import numpy as np
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
# from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# heckpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import ldm_quad.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


def apply_velocity_command_overrides(env_cfg: object) -> None:
    if not args_cli.wander and args_cli.command_x is None and args_cli.command_y is None and args_cli.command_yaw is None:
        return

    commands = getattr(env_cfg, "commands", None)
    command_cfg = getattr(commands, "base_velocity", None)
    if command_cfg is None:
        raise AttributeError("This task config does not expose commands.base_velocity")

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
    command_cfg.debug_vis = True


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    apply_velocity_command_overrides(env_cfg)

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    # if args_cli.use_pretrained_checkpoint:
    #     resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
    #     if not resume_path:
    #         print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
    #         return
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    experiment_cfg["agent"]["random_timesteps"] = 0  # don't do random exploration during play
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    command_error_xy: list[float] = []
    command_error_yaw: list[float] = []
    episode_returns = torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
    episode_lengths = torch.zeros(obs.shape[0], dtype=torch.float32, device=obs.device)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            if isinstance(obs, torch.Tensor) and obs.shape[-1] >= 12:
                command = obs[:, 9:12]
                base_lin_vel = obs[:, 0:3]
                base_ang_vel = obs[:, 3:6]
                command_error_xy.append(torch.linalg.norm(base_lin_vel[:, :2] - command[:, :2], dim=-1).mean().item())
                command_error_yaw.append((base_ang_vel[:, 2] - command[:, 2]).abs().mean().item())
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, rewards, terminated, truncated, _ = env.step(actions)
            rewards = rewards.float().view(-1)
            done = terminated.bool().view(-1) | truncated.bool().view(-1)
            episode_returns += rewards
            episode_lengths += 1

            if done.any():
                done_returns = episode_returns[done].detach().cpu().tolist()
                done_lengths = episode_lengths[done].detach().cpu().tolist()
                completed_returns.extend(done_returns)
                completed_lengths.extend(done_lengths)
                episode_returns[done] = 0.0
                episode_lengths[done] = 0.0

        timestep += 1
        if args_cli.video:
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.num_episodes > 0 and len(completed_returns) >= args_cli.num_episodes:
            break
        if args_cli.max_steps > 0 and timestep >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.num_episodes > 0 or args_cli.max_steps > 0:
        eval_returns = completed_returns[: args_cli.num_episodes] if args_cli.num_episodes > 0 else completed_returns
        eval_lengths = completed_lengths[: args_cli.num_episodes] if args_cli.num_episodes > 0 else completed_lengths
        print(f"[EVAL] steps={timestep} completed_episodes={len(eval_returns)}")
        if eval_returns:
            print(
                "[EVAL] "
                f"mean_return={float(np.mean(eval_returns)):.3f} "
                f"std_return={float(np.std(eval_returns)):.3f} "
                f"mean_length={float(np.mean(eval_lengths)):.2f} "
                f"min_length={float(np.min(eval_lengths)):.2f}"
            )
        if command_error_xy:
            print(
                "[EVAL] "
                f"mean_cmd_xy_error={float(np.mean(command_error_xy)):.3f} "
                f"mean_cmd_yaw_error={float(np.mean(command_error_yaw)):.3f}"
            )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
