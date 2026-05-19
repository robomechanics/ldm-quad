from __future__ import annotations

import os
from collections.abc import Callable

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


class TorchScriptPolicyPrior:
    """Frozen TorchScript policy prior.

    The exported RSL-RL Go2 rough policy in this repo was trained with a 45-D observation:
    base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, and actions.
    The go2_rsl_rough observation adapter can also map the older 48-D MBRL observation
    layout to that 45-D policy layout.
    """

    def __init__(
        self,
        checkpoint_path: str,
        obs_adapter: str = "go2_rsl_rough",
        action_adapter: str = "none",
        device: torch.device | str | None = None,
    ):
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.policy = torch.jit.load(self.checkpoint_path, map_location=device or "cpu")
        self.policy.eval()
        self.obs_adapter = obs_adapter
        self.action_adapter = action_adapter
        self.input_dim = self._infer_input_dim()
        self._device = torch.device(device or "cpu")

    def _infer_input_dim(self) -> int | None:
        for name, parameter in self.policy.named_parameters():
            if name.endswith("weight") and parameter.ndim == 2:
                return int(parameter.shape[1])
        return None

    def _adapt_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.input_dim is None or obs.shape[-1] == self.input_dim:
            return obs

        if self.obs_adapter != "go2_rsl_rough":
            raise ValueError(
                f"TorchScript prior expects {self.input_dim} observations but received {obs.shape[-1]}; "
                f"unknown obs adapter: {self.obs_adapter}"
            )

        if obs.shape[-1] != 48 or self.input_dim != 45:
            raise ValueError(
                f"go2_rsl_rough adapter expects 48-D MBRL obs -> 45-D RSL obs, got "
                f"{obs.shape[-1]} -> {self.input_dim}"
            )

        base_ang_vel = obs[..., 3:6] * 0.25
        projected_gravity = obs[..., 6:9]
        velocity_commands = obs[..., 9:12]
        joint_pos = obs[..., 12:24]
        joint_vel = obs[..., 24:36] * 0.05
        actions = obs[..., 36:48]
        return torch.cat((base_ang_vel, projected_gravity, velocity_commands, joint_pos, joint_vel, actions), dim=-1)

    def _adapt_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.action_adapter in {"none", ""}:
            return actions

        if self.action_adapter != "go2_rsl_rough":
            raise ValueError(f"Unknown TorchScript prior action adapter: {self.action_adapter}")

        if actions.shape[-1] != 12:
            raise ValueError(f"go2_rsl_rough action adapter expects 12-D actions, got {actions.shape[-1]}")

        # The exported rough prior was trained with half-sized hip action scale
        # relative to the scalar 0.25 joint action scale used by the local MBRL env.
        # Convert raw policy actions so the resulting joint-position targets match.
        scale = torch.ones(actions.shape[-1], device=actions.device, dtype=actions.dtype)
        scale[[0, 3, 6, 9]] = 0.5
        return actions * scale

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.device != self._device:
            self.policy.to(obs.device)
            self._device = obs.device
        actions = self.policy(self._adapt_obs(obs))
        if isinstance(actions, tuple):
            actions = actions[0]
        return self._adapt_actions(actions)


def load_policy_prior(
    env: object,
    checkpoint_path: str,
    task_name: str,
    prior_type: str = "auto",
    algorithm: str = "PPO",
    agent_cfg_entry_point: str | None = None,
    obs_adapter: str = "go2_rsl_rough",
    action_adapter: str = "none",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Load either a skrl checkpoint or an exported TorchScript policy prior."""

    prior_type = prior_type.lower()
    if prior_type not in {"auto", "skrl", "torchscript", "rsl_jit"}:
        raise ValueError(f"Unsupported prior_type: {prior_type}")

    if prior_type in {"torchscript", "rsl_jit"}:
        return TorchScriptPolicyPrior(
            checkpoint_path=checkpoint_path,
            obs_adapter=obs_adapter,
            action_adapter=action_adapter,
        )

    if prior_type == "auto":
        try:
            return TorchScriptPolicyPrior(
                checkpoint_path=checkpoint_path,
                obs_adapter=obs_adapter,
                action_adapter=action_adapter,
            )
        except RuntimeError:
            pass

    return SkrlPolicyPrior(
        env=env,
        checkpoint_path=checkpoint_path,
        task_name=task_name,
        algorithm=algorithm,
        agent_cfg_entry_point=agent_cfg_entry_point,
    )
