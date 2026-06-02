from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DistributionalRegressionCfg:
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

    @property
    def bin_size(self) -> float:
        return (self.vmax - self.vmin) / max(self.num_bins - 1, 1)


def mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class SimNorm(nn.Module):
    """Simplex normalization over fixed-size latent groups."""

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 1:
            raise ValueError("SimNorm dim must be greater than 1.")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if shape[-1] % self.dim != 0:
            raise ValueError(f"Latent dimension {shape[-1]} must be divisible by simnorm_dim={self.dim}.")
        x = x.view(*shape[:-1], shape[-1] // self.dim, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shape)


def latent_mlp(input_dim: int, hidden_dim: int, latent_dim: int, depth: int, simnorm_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = [mlp(input_dim, hidden_dim, latent_dim, depth)]
    if simnorm_dim > 1:
        layers.append(SimNorm(simnorm_dim))
    return nn.Sequential(*layers)


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1.0)


def two_hot(x: torch.Tensor, cfg: DistributionalRegressionCfg) -> torch.Tensor:
    """Convert scalar targets to soft two-hot symlog targets."""

    if cfg.num_bins <= 1:
        return symlog(x)

    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(-1)
    bin_position = (x - cfg.vmin) / cfg.bin_size
    bin_idx = torch.floor(bin_position).long().clamp(0, cfg.num_bins - 1)
    bin_offset = (bin_position - bin_idx.to(bin_position.dtype)).unsqueeze(-1)

    target = torch.zeros(*x.shape, cfg.num_bins, device=x.device, dtype=x.dtype)
    target.scatter_(-1, bin_idx.unsqueeze(-1), 1.0 - bin_offset)
    target.scatter_add_(-1, ((bin_idx + 1) % cfg.num_bins).unsqueeze(-1), bin_offset)
    return target


def two_hot_inv(logits: torch.Tensor, cfg: DistributionalRegressionCfg) -> torch.Tensor:
    """Decode two-hot logits to scalar values."""

    if cfg.num_bins <= 1:
        return symexp(logits)

    bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=logits.device, dtype=logits.dtype)
    probs = F.softmax(logits, dim=-1)
    value = (probs * bins).sum(dim=-1, keepdim=True)
    return symexp(value)


def soft_ce(logits: torch.Tensor, target: torch.Tensor, cfg: DistributionalRegressionCfg) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(two_hot(target, cfg) * log_probs).sum(dim=-1, keepdim=True)


class RunningScale(nn.Module):
    """Running trimmed scale estimator used to normalize actor Q-values."""

    def __init__(self, tau: float):
        super().__init__()
        self.tau = tau
        self.register_buffer("value", torch.ones(1, dtype=torch.float32))
        self.register_buffer("_percentiles", torch.tensor([5.0, 95.0], dtype=torch.float32))

    def _positions(self, x_shape: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = self._percentiles.to(dtype=torch.float32, device=self.value.device) * (x_shape - 1) / 100.0
        floored = torch.floor(positions)
        ceiled = (floored + 1).clamp(max=x_shape - 1)
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        return floored.long(), ceiled.long(), weight_floored.unsqueeze(1), weight_ceiled.unsqueeze(1)

    def _percentile(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype, x_shape = x.dtype, x.shape
        x = x.flatten(1, x.ndim - 1)
        sorted_x = torch.sort(x, dim=0).values
        floored, ceiled, weight_floored, weight_ceiled = self._positions(sorted_x.shape[0])
        d0 = sorted_x[floored] * weight_floored
        d1 = sorted_x[ceiled] * weight_ceiled
        return (d0 + d1).reshape(-1, *x_shape[1:]).to(x_dtype)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
        self.value.data.lerp_(value.to(self.value.device), self.tau)

    def forward(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update:
            self.update(x)
        return x / self.value.to(device=x.device, dtype=x.dtype)


@dataclass
class WorldModelLossWeights:
    consistency: float = 20.0
    reward: float = 0.1
    value: float = 0.1
    continue_: float = 1.0


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
        entropy_coef: float = 1e-4,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        simnorm_dim: int = 8,
        q_dropout: float = 0.01,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
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
        self.entropy_coef = entropy_coef
        self.dreg = DistributionalRegressionCfg(num_bins=num_bins, vmin=vmin, vmax=vmax)
        self.simnorm_dim = simnorm_dim
        self.q_dropout = q_dropout
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.loss_weights = loss_weights or WorldModelLossWeights()
        head_dim = max(num_bins, 1)
        self.q_scale = RunningScale(tau)

        self.encoder = latent_mlp(obs_dim, hidden_dim, latent_dim, depth, simnorm_dim)
        self.dynamics = latent_mlp(latent_dim + action_dim, hidden_dim, latent_dim, depth, simnorm_dim)
        self.reward_head = mlp(latent_dim + action_dim, hidden_dim, head_dim, depth)
        self.continue_head = mlp(latent_dim, hidden_dim, 1, depth)
        self.policy_head = mlp(latent_dim, hidden_dim, 2 * action_dim, depth)
        self.q_heads = nn.ModuleList(
            mlp(latent_dim + action_dim, hidden_dim, head_dim, depth, dropout=q_dropout) for _ in range(num_q)
        )
        self._zero_init_distribution_heads()

        self.target_encoder = deepcopy(self.encoder)
        self.target_q_heads = deepcopy(self.q_heads)
        self._set_targets_requires_grad(False)

    def _zero_init_distribution_heads(self) -> None:
        for head in [self.reward_head, *self.q_heads]:
            final_layer = next((module for module in reversed(head) if isinstance(module, nn.Linear)), None)
            if final_layer is not None:
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)

    def _set_targets_requires_grad(self, requires_grad: bool) -> None:
        for module in (self.target_encoder, self.target_q_heads):
            for param in module.parameters():
                param.requires_grad_(requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.train(False)
        self.target_q_heads.train(False)
        return self

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
        logits = self.reward_logits(z, actions)
        return two_hot_inv(logits, self.dreg)

    def reward_logits(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat([z, actions], dim=-1))

    def continue_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.continue_head(z)

    def _policy_stats(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.policy_head(z).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def pi(self, z: torch.Tensor, deterministic: bool = True, return_info: bool = False):
        mean, log_std = self._policy_stats(z)
        if deterministic:
            pre_tanh = mean
        else:
            pre_tanh = mean + log_std.exp() * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        if not return_info:
            return action

        # Squashed Gaussian log-probability. The correction keeps entropy useful
        # near action bounds while matching the tanh action used by the planner.
        variance = (2.0 * log_std).exp()
        log_prob = -0.5 * ((pre_tanh - mean).square() / variance + 2.0 * log_std + torch.log(torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype)))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        correction = torch.log(1.0 - action.square() + 1e-6).sum(dim=-1, keepdim=True)
        log_prob = log_prob - correction
        entropy = -log_prob
        scaled_entropy = entropy * self.action_dim
        info = {
            "mean": torch.tanh(mean),
            "log_std": log_std,
            "log_prob": log_prob,
            "entropy": entropy,
            "scaled_entropy": scaled_entropy,
        }
        return action, info

    def Q(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        target: bool = False,
        return_all: bool = False,
        return_type: str = "min",
    ) -> torch.Tensor:
        qs = self.Q_logits(z, actions, target=target)
        if return_all:
            return two_hot_inv(qs, self.dreg)
        values = two_hot_inv(qs, self.dreg)
        if return_type == "all":
            return values
        if values.shape[0] >= 2:
            pair = torch.randperm(values.shape[0], device=values.device)[:2]
            values = values[pair]
        if return_type == "min":
            return values.min(dim=0).values
        if return_type == "avg":
            return values.mean(dim=0)
        raise ValueError(f"Unsupported Q return_type: {return_type}")

    def Q_logits(self, z: torch.Tensor, actions: torch.Tensor, target: bool = False) -> torch.Tensor:
        heads = self.target_q_heads if target else self.q_heads
        inputs = torch.cat([z, actions], dim=-1)
        return torch.stack([head(inputs) for head in heads], dim=0)

    def model_parameters(self):
        modules = (self.encoder, self.dynamics, self.reward_head, self.continue_head, self.q_heads)
        for module in modules:
            yield from module.parameters()

    def policy_parameters(self):
        yield from self.policy_head.parameters()

    def policy_loss(self, zs: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        q_requires_grad = [param.requires_grad for head in self.q_heads for param in head.parameters()]
        q_params = [param for head in self.q_heads for param in head.parameters()]
        for param in q_params:
            param.requires_grad_(False)
        actions, info = self.pi(zs.detach(), deterministic=False, return_info=True)
        q = self.Q(zs.detach(), actions, return_type="avg")
        self.q_scale.update(q[0])
        scaled_q = self.q_scale(q)
        rho = torch.pow(
            torch.as_tensor(self.rho, device=zs.device, dtype=zs.dtype),
            torch.arange(zs.shape[0], device=zs.device, dtype=zs.dtype),
        )
        per_step_loss = -(scaled_q + self.entropy_coef * info["scaled_entropy"]).mean(dim=(1, 2))
        loss = (per_step_loss * rho).mean()
        for param, requires_grad in zip(q_params, q_requires_grad, strict=True):
            param.requires_grad_(requires_grad)
        metrics = {
            "policy_loss": float(loss.detach().item()),
            "policy_q": float(q.detach().mean().item()),
            "policy_scaled_q": float(scaled_q.detach().mean().item()),
            "policy_entropy": float(info["entropy"].detach().mean().item()),
            "policy_scaled_entropy": float(info["scaled_entropy"].detach().mean().item()),
            "policy_q_scale": float(self.q_scale.value.detach().mean().item()),
        }
        return loss, metrics

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        continues = batch["continues"]
        if obs.ndim != 3:
            raise ValueError("LatentWorldModel.loss expects sequence batches shaped [H+1, B, dim].")

        horizon = actions.shape[0]
        with torch.no_grad():
            target_zs = self.encode(obs[1:].reshape(-1, self.obs_dim), target=False).view(horizon, obs.shape[1], -1)

        z = self.encode(obs[0])
        consistency_loss = torch.zeros((), device=obs.device)
        reward_loss = torch.zeros((), device=obs.device)
        value_loss = torch.zeros((), device=obs.device)
        continue_loss = torch.zeros((), device=obs.device)
        rollout_zs = [z]

        for t in range(horizon):
            weight = self.rho**t
            action_t = actions[t]
            reward_t = rewards[t]
            continue_t = continues[t]
            z_target = target_zs[t]

            q_logits = self.Q_logits(z, action_t)
            reward_logits = self.reward_logits(z, action_t)
            z_next = self.next(z, action_t)
            continue_pred = self.continue_logits(z_next)

            with torch.no_grad():
                target_action = self.pi(z_target, deterministic=False)
                target_q = reward_t + self.discount * continue_t * self.Q(
                    z_target,
                    target_action,
                    target=True,
                    return_type="min",
                )

            consistency_loss = consistency_loss + weight * F.mse_loss(z_next, z_target)
            reward_loss = reward_loss + weight * soft_ce(reward_logits, reward_t, self.dreg).mean()
            value_target = target_q.unsqueeze(0).expand(q_logits.shape[0], *target_q.shape)
            value_loss = value_loss + weight * soft_ce(q_logits.reshape(-1, q_logits.shape[-1]), value_target.reshape(-1, 1), self.dreg).mean()
            continue_loss = continue_loss + weight * F.binary_cross_entropy_with_logits(continue_pred, continue_t)
            z = z_next
            rollout_zs.append(z)

        normalizer = sum(self.rho**t for t in range(horizon))
        consistency_loss = consistency_loss / normalizer
        reward_loss = reward_loss / normalizer
        value_loss = value_loss / normalizer
        continue_loss = continue_loss / normalizer

        weights = self.loss_weights
        total = (
            weights.consistency * consistency_loss
            + weights.reward * reward_loss
            + weights.value * value_loss
            + weights.continue_ * continue_loss
        )
        metrics = {
            "loss": float(total.detach().item()),
            "consistency_loss": float(consistency_loss.detach().item()),
            "reward_loss": float(reward_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "continue_loss": float(continue_loss.detach().item()),
        }
        return total, metrics, torch.stack(rollout_zs, dim=0).detach()
