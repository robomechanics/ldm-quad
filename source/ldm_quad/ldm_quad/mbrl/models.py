from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .world_model import DistributionalRegressionCfg, soft_ce, two_hot_inv


class EnsembleMLP(nn.Module):
    """Small MLP used as one member of the dynamics ensemble."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DynamicsPrediction:
    delta_obs: torch.Tensor
    rewards: torch.Tensor
    continue_logits: torch.Tensor


class DynamicsEnsemble(nn.Module):
    """Predicts observation deltas, rewards, and continuation logits."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dim: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        output_dim = obs_dim + 2
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.members = nn.ModuleList(
            EnsembleMLP(obs_dim + action_dim, output_dim, hidden_dim, depth) for _ in range(ensemble_size)
        )

    @property
    def ensemble_size(self) -> int:
        return len(self.members)

    def forward_members(self, obs: torch.Tensor, actions: torch.Tensor) -> DynamicsPrediction:
        inputs = torch.cat([obs, actions], dim=-1)
        preds = torch.stack([member(inputs) for member in self.members], dim=0)
        delta_obs = preds[..., : self.obs_dim]
        rewards = preds[..., self.obs_dim : self.obs_dim + 1]
        continue_logits = preds[..., self.obs_dim + 1 :]
        return DynamicsPrediction(delta_obs=delta_obs, rewards=rewards, continue_logits=continue_logits)

    def predict(self, obs: torch.Tensor, actions: torch.Tensor) -> DynamicsPrediction:
        member_preds = self.forward_members(obs, actions)
        return DynamicsPrediction(
            delta_obs=member_preds.delta_obs.mean(dim=0),
            rewards=member_preds.rewards.mean(dim=0),
            continue_logits=member_preds.continue_logits.mean(dim=0),
        )

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        obs = batch["obs"]
        actions = batch["actions"]
        target_delta = batch["next_obs"] - obs
        target_rewards = batch["rewards"]
        target_continue = batch["continues"]

        preds = self.forward_members(obs, actions)

        target_delta = target_delta.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        target_rewards = target_rewards.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        target_continue = target_continue.unsqueeze(0).expand(self.ensemble_size, -1, -1)

        delta_loss = F.mse_loss(preds.delta_obs, target_delta)
        reward_loss = F.mse_loss(preds.rewards, target_rewards)
        continue_loss = F.binary_cross_entropy_with_logits(preds.continue_logits, target_continue)
        total = delta_loss + reward_loss + 0.1 * continue_loss

        metrics = {
            "loss": float(total.detach().item()),
            "delta_loss": float(delta_loss.detach().item()),
            "reward_loss": float(reward_loss.detach().item()),
            "continue_loss": float(continue_loss.detach().item()),
        }
        return total, metrics


class StateWorldModel(DynamicsEnsemble):
    """Physical-state world model with TD-MPC-style value/policy helpers."""

    is_state_world_model = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        hidden_dim: int = 512,
        depth: int = 3,
        discount: float = 0.99,
        tau: float = 0.01,
        rho: float = 0.5,
        entropy_coef: float = 1e-4,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        value_coef: float = 0.1,
        reward_coef: float = 0.1,
        continue_coef: float = 1.0,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
    ):
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            ensemble_size=ensemble_size,
            hidden_dim=hidden_dim,
            depth=depth,
        )
        self.discount = discount
        self.tau = tau
        self.rho = rho
        self.entropy_coef = entropy_coef
        self.dreg = DistributionalRegressionCfg(num_bins=num_bins, vmin=vmin, vmax=vmax)
        self.value_coef = value_coef
        self.reward_coef = reward_coef
        self.continue_coef = continue_coef
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        head_dim = max(num_bins, 1)
        self.reward_head = EnsembleMLP(obs_dim + action_dim, head_dim, hidden_dim, depth)
        self.value_head = EnsembleMLP(obs_dim, head_dim, hidden_dim, depth)
        self.target_value_head = EnsembleMLP(obs_dim, head_dim, hidden_dim, depth)
        self.policy_head = EnsembleMLP(obs_dim, 2 * action_dim, hidden_dim, depth)
        self.target_value_head.load_state_dict(self.value_head.state_dict())
        for param in self.target_value_head.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_value_head.train(False)
        return self

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.target_value_head.parameters(), self.value_head.parameters(), strict=True):
            target_param.lerp_(param, self.tau)

    def reward_logits(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat([obs, actions], dim=-1))

    def reward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return two_hot_inv(self.reward_logits(obs, actions), self.dreg)

    def value_logits(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        head = self.target_value_head if target else self.value_head
        return head(obs)

    def value(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        return two_hot_inv(self.value_logits(obs, target=target), self.dreg)

    def _policy_stats(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.policy_head(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mean, log_std

    def pi(self, obs: torch.Tensor, deterministic: bool = True, return_info: bool = False):
        mean, log_std = self._policy_stats(obs)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        if not return_info:
            return action

        variance = (2.0 * log_std).exp()
        log_prob = -0.5 * (
            (pre_tanh - mean).square() / variance
            + 2.0 * log_std
            + torch.log(torch.tensor(2.0 * torch.pi, device=obs.device, dtype=obs.dtype))
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        correction = torch.log(1.0 - action.square() + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = log_prob - correction
        entropy = -log_prob
        return action, {"log_prob": log_prob, "entropy": entropy, "scaled_entropy": entropy * self.action_dim}

    def model_parameters(self):
        yield from self.members.parameters()
        yield from self.reward_head.parameters()
        yield from self.value_head.parameters()

    def policy_parameters(self):
        yield from self.policy_head.parameters()

    @torch.no_grad()
    def disagreement(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        preds = self.forward_members(obs, actions)
        delta_std = preds.delta_obs.std(dim=0, unbiased=False)
        reward_std = preds.rewards.std(dim=0, unbiased=False)
        return delta_std.square().mean(dim=-1) + reward_std.squeeze(-1).square()

    def predict(self, obs: torch.Tensor, actions: torch.Tensor) -> DynamicsPrediction:
        preds = super().predict(obs, actions)
        return DynamicsPrediction(
            delta_obs=preds.delta_obs,
            rewards=self.reward(obs, actions),
            continue_logits=preds.continue_logits,
        )

    def _one_step_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        obs = batch["obs"]
        actions = batch["actions"]
        target_delta = batch["next_obs"] - obs
        target_rewards = batch["rewards"]
        target_continue = batch["continues"]

        preds = self.forward_members(obs, actions)

        ensemble_target_delta = target_delta.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        ensemble_target_rewards = target_rewards.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        ensemble_target_continue = target_continue.unsqueeze(0).expand(self.ensemble_size, -1, -1)

        delta_loss = F.mse_loss(preds.delta_obs, ensemble_target_delta)
        continue_loss = F.binary_cross_entropy_with_logits(preds.continue_logits, ensemble_target_continue)
        reward_logits = self.reward_logits(obs, actions)
        reward_loss = soft_ce(reward_logits, target_rewards, self.dreg).mean()
        reward_loss = reward_loss + F.mse_loss(preds.rewards, ensemble_target_rewards)

        with torch.no_grad():
            target_value = target_rewards + self.discount * target_continue * self.value(batch["next_obs"], target=True)
        value_logits = self.value_logits(obs)
        value_loss = soft_ce(value_logits, target_value, self.dreg).mean()

        total = delta_loss + self.reward_coef * reward_loss + self.continue_coef * continue_loss + self.value_coef * value_loss

        metrics = {
            "loss": float(total.detach().item()),
            "delta_loss": float(delta_loss.detach().item()),
            "reward_loss": float(reward_loss.detach().item()),
            "continue_loss": float(continue_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "value_mean": float(self.value(obs).detach().mean().item()),
        }
        return total, metrics, torch.stack((obs, batch["next_obs"]), dim=0).detach()

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        if batch["obs"].ndim != 3:
            return self._one_step_loss(batch)

        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        continues = batch["continues"]
        horizon = actions.shape[0]
        state = obs[0]
        delta_loss = torch.zeros((), device=obs.device)
        reward_loss = torch.zeros((), device=obs.device)
        value_loss = torch.zeros((), device=obs.device)
        continue_loss = torch.zeros((), device=obs.device)
        rollout_states = [state]

        for t in range(horizon):
            weight = self.rho**t
            action_t = actions[t]
            target_delta = obs[t + 1] - state
            preds = self.forward_members(state, action_t)
            delta_target = target_delta.unsqueeze(0).expand(self.ensemble_size, -1, -1)
            continue_target = continues[t].unsqueeze(0).expand(self.ensemble_size, -1, -1)
            delta_loss = delta_loss + weight * F.mse_loss(preds.delta_obs, delta_target)
            continue_loss = continue_loss + weight * F.binary_cross_entropy_with_logits(
                preds.continue_logits,
                continue_target,
            )
            reward_loss = reward_loss + weight * soft_ce(self.reward_logits(state, action_t), rewards[t], self.dreg).mean()
            reward_target = rewards[t].unsqueeze(0).expand(self.ensemble_size, -1, -1)
            reward_loss = reward_loss + weight * F.mse_loss(preds.rewards, reward_target)
            next_state = state + preds.delta_obs.mean(dim=0)
            with torch.no_grad():
                target_value = rewards[t] + self.discount * continues[t] * self.value(obs[t + 1], target=True)
            value_loss = value_loss + weight * soft_ce(self.value_logits(state), target_value, self.dreg).mean()
            state = next_state
            rollout_states.append(state)

        normalizer = sum(self.rho**t for t in range(horizon))
        delta_loss = delta_loss / normalizer
        reward_loss = reward_loss / normalizer
        value_loss = value_loss / normalizer
        continue_loss = continue_loss / normalizer
        total = delta_loss + self.reward_coef * reward_loss + self.continue_coef * continue_loss + self.value_coef * value_loss
        metrics = {
            "loss": float(total.detach().item()),
            "delta_loss": float(delta_loss.detach().item()),
            "reward_loss": float(reward_loss.detach().item()),
            "continue_loss": float(continue_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "value_mean": float(self.value(obs[0]).detach().mean().item()),
        }
        return total, metrics, torch.stack(rollout_states, dim=0).detach()

    def policy_loss(self, states: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        flat_states = states.detach().reshape(-1, self.obs_dim)
        actions, info = self.pi(flat_states, deterministic=False, return_info=True)
        preds = super().predict(flat_states, actions)
        next_states = flat_states + preds.delta_obs
        objective = self.reward(flat_states, actions) + self.discount * preds.continue_logits.sigmoid() * self.value(next_states)
        loss = -(objective + self.entropy_coef * info["scaled_entropy"]).mean()
        metrics = {
            "policy_loss": float(loss.detach().item()),
            "policy_value": float(objective.detach().mean().item()),
            "policy_entropy": float(info["entropy"].detach().mean().item()),
        }
        return loss, metrics
