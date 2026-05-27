from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class WorldModelLossWeights:
    consistency: float = 1.0
    reward: float = 1.0
    value: float = 0.5
    continue_: float = 0.1
    policy: float = 0.05


class LatentWorldModel(nn.Module):
    """TD-MPC-style decoder-free latent world model for proprioceptive observations."""

    is_latent_world_model = True

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        depth: int = 3,
        num_q: int = 2,
        discount: float = 0.99,
        tau: float = 0.01,
        rho: float = 0.5,
        loss_weights: WorldModelLossWeights | None = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_q = num_q
        self.discount = discount
        self.tau = tau
        self.rho = rho
        self.loss_weights = loss_weights or WorldModelLossWeights()

        self.encoder = mlp(obs_dim, hidden_dim, latent_dim, depth)
        self.dynamics = mlp(latent_dim + action_dim, hidden_dim, latent_dim, depth)
        self.reward_head = mlp(latent_dim + action_dim, hidden_dim, 1, depth)
        self.continue_head = mlp(latent_dim, hidden_dim, 1, depth)
        self.policy_head = mlp(latent_dim, hidden_dim, action_dim, depth)
        self.q_heads = nn.ModuleList(mlp(latent_dim + action_dim, hidden_dim, 1, depth) for _ in range(num_q))

        self.target_encoder = deepcopy(self.encoder)
        self.target_q_heads = deepcopy(self.q_heads)
        self._set_targets_requires_grad(False)

    def _set_targets_requires_grad(self, requires_grad: bool) -> None:
        for module in (self.target_encoder, self.target_q_heads):
            for param in module.parameters():
                param.requires_grad_(requires_grad)

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target_param.lerp_(param, self.tau)
        for target_param, param in zip(self.target_q_heads.parameters(), self.q_heads.parameters(), strict=True):
            target_param.lerp_(param, self.tau)

    def encode(self, obs: torch.Tensor, target: bool = False) -> torch.Tensor:
        encoder = self.target_encoder if target else self.encoder
        return encoder(obs)

    def next(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z, actions], dim=-1))

    def reward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat([z, actions], dim=-1))

    def continue_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.continue_head(z)

    def pi(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.policy_head(z))

    def Q(self, z: torch.Tensor, actions: torch.Tensor, target: bool = False, return_all: bool = False) -> torch.Tensor:
        heads = self.target_q_heads if target else self.q_heads
        inputs = torch.cat([z, actions], dim=-1)
        qs = torch.stack([head(inputs) for head in heads], dim=0)
        if return_all:
            return qs
        return qs.min(dim=0).values

    def _policy_loss(self, zs: torch.Tensor) -> torch.Tensor:
        q_requires_grad = [param.requires_grad for head in self.q_heads for param in head.parameters()]
        q_params = [param for head in self.q_heads for param in head.parameters()]
        for param in q_params:
            param.requires_grad_(False)
        actions = self.pi(zs.detach())
        loss = -self.Q(zs.detach(), actions).mean()
        for param, requires_grad in zip(q_params, q_requires_grad, strict=True):
            param.requires_grad_(requires_grad)
        return loss

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        continues = batch["continues"]
        if obs.ndim != 3:
            raise ValueError("LatentWorldModel.loss expects sequence batches shaped [H+1, B, dim].")

        horizon = actions.shape[0]
        with torch.no_grad():
            target_zs = self.encode(obs[1:].reshape(-1, self.obs_dim), target=True).view(horizon, obs.shape[1], -1)

        z = self.encode(obs[0])
        consistency_loss = torch.zeros((), device=obs.device)
        reward_loss = torch.zeros((), device=obs.device)
        value_loss = torch.zeros((), device=obs.device)
        continue_loss = torch.zeros((), device=obs.device)
        rollout_zs = []

        for t in range(horizon):
            weight = self.rho**t
            action_t = actions[t]
            reward_t = rewards[t]
            continue_t = continues[t]
            z_target = target_zs[t]

            q_pred = self.Q(z, action_t, return_all=True)
            reward_pred = self.reward(z, action_t)
            z_next = self.next(z, action_t)
            continue_pred = self.continue_logits(z_next)

            with torch.no_grad():
                target_action = self.pi(z_target)
                target_q = reward_t + self.discount * continue_t * self.Q(z_target, target_action, target=True)

            consistency_loss = consistency_loss + weight * F.mse_loss(z_next, z_target)
            reward_loss = reward_loss + weight * F.mse_loss(reward_pred, reward_t)
            value_loss = value_loss + weight * F.mse_loss(q_pred, target_q.unsqueeze(0).expand_as(q_pred))
            continue_loss = continue_loss + weight * F.binary_cross_entropy_with_logits(continue_pred, continue_t)
            rollout_zs.append(z)
            z = z_next

        normalizer = sum(self.rho**t for t in range(horizon))
        consistency_loss = consistency_loss / normalizer
        reward_loss = reward_loss / normalizer
        value_loss = value_loss / normalizer
        continue_loss = continue_loss / normalizer
        policy_loss = self._policy_loss(torch.stack(rollout_zs, dim=0))

        weights = self.loss_weights
        total = (
            weights.consistency * consistency_loss
            + weights.reward * reward_loss
            + weights.value * value_loss
            + weights.continue_ * continue_loss
            + weights.policy * policy_loss
        )
        metrics = {
            "loss": float(total.detach().item()),
            "consistency_loss": float(consistency_loss.detach().item()),
            "reward_loss": float(reward_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "continue_loss": float(continue_loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
        }
        return total, metrics
