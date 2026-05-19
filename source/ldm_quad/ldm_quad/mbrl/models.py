from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


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
