

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

# Example Script Usage:
# python scripts/skrl/train.py --task=Random-Agent-Unitree-Go2-v0 --num_envs=4096 --headless


import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_ROOT = os.path.join(PROJECT_ROOT, "source", "ldm_quad")
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default=None, help=("Name of the RL agent configuration entry point. Defaults to None, in which case the argument --algorithm is used to determine the default agent configuration entry point."),)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--ml_framework",type=str,default="torch",choices=["torch", "jax", "jax-numpy"],help="The ML framework used for training the skrl agent.",)
parser.add_argument("--algorithm",type=str,default="PPO",choices=["AMP", "PPO", "SAC", "IPPO", "MAPPO"],help="The RL algorithm used for training the skrl agent.",)
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

import gymnasium as gym
import random
from datetime import datetime

import omni
import skrl
from packaging import version

import ldm_quad # noqa: F401

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

from isaaclab.envs import (DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config



# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    apply_velocity_command_overrides(env_cfg)

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        rollouts = agent_cfg["agent"].get("rollouts")
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * rollouts if rollouts else args_cli.max_iterations
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = os.path.join(log_root_path, log_dir)
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Off-policy algorithms such as SAC sample random actions during exploration.
    # ManagerBasedRLEnv exposes unbounded action spaces, so clamp them before skrl wraps the env.
    action_clip = agent_cfg.get("action_clip", None)
    if action_clip is not None:
        import numpy as np

        clip_val = float(action_clip)
        unwrapped = env.unwrapped
        single_shape = (
            unwrapped.single_action_space.shape
            if hasattr(unwrapped, "single_action_space")
            else unwrapped.action_space.shape
        )
        bounded_space = gym.spaces.Box(low=-clip_val, high=clip_val, shape=single_shape, dtype=np.float32)
        unwrapped.single_action_space = bounded_space
        unwrapped.action_space = bounded_space
        print(f"[INFO] Action space overridden to [-{clip_val}, {clip_val}]: {bounded_space}")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
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
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)


    # run training
    runner.run()

    # close the simulator
    env.close()


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


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
