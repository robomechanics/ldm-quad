from __future__ import annotations

import os

import torch


class SkrlPolicyPrior:
    """Frozen skrl policy used as a locomotion prior for residual MPC."""

    def __init__(
        self,
        env: object,
        checkpoint_path: str,
        task_name: str,
        algorithm: str = "PPO",
        agent_cfg_entry_point: str | None = None,
    ):
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry
        from skrl.utils.runner.torch import Runner

        algorithm = algorithm.lower()
        if agent_cfg_entry_point is None:
            agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"

        agent_cfg = load_cfg_from_registry(task_name, agent_cfg_entry_point)
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        agent_cfg["agent"]["random_timesteps"] = 0

        wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")
        self.runner = Runner(wrapped_env, agent_cfg)
        self.agent = self.runner.agent
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.agent.load(self.checkpoint_path)
        self.agent.set_running_mode("eval")

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        outputs = self.agent.act(obs, timestep=0, timesteps=0)
        return outputs[-1].get("mean_actions", outputs[0])
