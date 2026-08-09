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
        layers.append(nn.Mish())
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
    target.scatter_add_(-1, (bin_idx + 1).clamp(max=cfg.num_bins - 1).unsqueeze(-1), bin_offset)
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


class HistoryEncoder(nn.Module):
    """Permutation-invariant set-transformer over recent transitions.

    Ported from robomechanics/payAttentionDrift's system-identification transformer.
    Each history token concatenates the applied action with the *observed* one-step
    transition ``next_obs - obs`` (the sysid cue: "this action in this regime produced
    this change"). Tokens are embedded, passed through a small TransformerEncoder with
    no causal mask, mean-pooled over valid steps, and compressed to a fixed context
    vector. Empty histories fall back to a learned ``null_context``.
    """

    def __init__(
        self,
        action_dim: int,
        transition_dim: int,
        context_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_fc = nn.Linear(action_dim + transition_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, nn.LayerNorm(d_model))
        self.final_fc = nn.Linear(d_model, context_dim)
        self.null_context = nn.Parameter(torch.zeros(context_dim))

    def forward(
        self,
        actions: torch.Tensor,
        transitions: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """actions ``[B, K, action_dim]``, transitions ``[B, K, transition_dim]``,
        pad_mask ``[B, K]`` with ``True`` marking padded (invalid) history steps.
        Returns the context vector ``[B, context_dim]``."""
        tokens = self.input_fc(torch.cat([actions, transitions], dim=-1))
        if pad_mask is None:
            encoded = self.encoder(tokens)
            pooled = encoded.mean(dim=1)
            return self._project(self.final_fc(pooled))

        # Attend only over valid steps. A fully padded row masks every key and makes
        # TransformerEncoder emit NaNs, so feed those rows an all-valid mask and
        # overwrite their output with the learned null context afterwards.
        all_pad = pad_mask.all(dim=1, keepdim=True)
        safe_mask = pad_mask & ~all_pad
        encoded = self.encoder(tokens, src_key_padding_mask=safe_mask)
        keep = (~pad_mask).unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)
        context = self.final_fc(pooled)
        context = torch.where(all_pad, self.null_context.to(context.dtype), context)
        return self._project(context)

    @staticmethod
    def _project(context: torch.Tensor) -> torch.Tensor:
        """Project onto the unit ball (TD-MPC2 constrains the conditioning vector to
        ``||c|| <= 1`` for stable conditioning)."""
        return context / context.norm(dim=-1, keepdim=True).clamp_min(1.0)

    def null(self) -> torch.Tensor:
        """Normalized empty-history context, used when no history is available."""
        return self._project(self.null_context)


@dataclass
class WorldModelLossWeights:
    consistency: float = 20.0
    reward: float = 0.1
    value: float = 0.1
    continue_: float = 1.0
    physical: float = 0.0


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
        physical_feature_indices: list[int] | tuple[int, ...] | None = None,
        loss_weights: WorldModelLossWeights | None = None,
        context_dim: int = 0,
        history_len: int = 0,
        history_d_model: int = 64,
        history_nhead: int = 4,
        history_layers: int = 1,
        history_ff: int = 256,
        history_dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.context_dim = max(0, int(context_dim))
        self.history_len = max(0, int(history_len))
        if self.context_dim > 0 and self.history_len <= 0:
            raise ValueError("context_dim > 0 requires history_len > 0.")
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
        self.physical_feature_indices = tuple(physical_feature_indices or ())
        self.loss_weights = loss_weights or WorldModelLossWeights()
        self._last_context: torch.Tensor | None = None
        head_dim = max(num_bins, 1)
        self.q_scale = RunningScale(tau)

        self.history_encoder = (
            HistoryEncoder(
                action_dim=action_dim,
                transition_dim=obs_dim,
                context_dim=self.context_dim,
                d_model=history_d_model,
                nhead=history_nhead,
                dim_feedforward=history_ff,
                num_layers=history_layers,
                dropout=history_dropout,
            )
            if self.context_dim > 0
            else None
        )
        # The system-id context is concatenated to every conditioned component, mirroring
        # TD-MPC2's learnable task embedding e = h(s,e), d(z,a,e), R(z,a,e), Q(z,a,e), p(z,e).
        # continue/physical heads read the (already context-aware) latent directly.
        ctx = self.context_dim
        self.encoder = latent_mlp(obs_dim + ctx, hidden_dim, latent_dim, depth, simnorm_dim)
        self.dynamics = latent_mlp(latent_dim + action_dim + ctx, hidden_dim, latent_dim, depth, simnorm_dim)
        self.reward_head = mlp(latent_dim + action_dim + ctx, hidden_dim, head_dim, depth)
        self.continue_head = mlp(latent_dim, hidden_dim, 1, depth)
        self.policy_head = mlp(latent_dim + ctx, hidden_dim, 2 * action_dim, depth)
        self.physical_head = (
            mlp(latent_dim, hidden_dim, len(self.physical_feature_indices), depth)
            if self.physical_feature_indices
            else None
        )
        self.q_heads = nn.ModuleList(
            mlp(latent_dim + action_dim + ctx, hidden_dim, head_dim, depth, dropout=q_dropout) for _ in range(num_q)
        )
        self._zero_init_distribution_heads()

        self.target_encoder = deepcopy(self.encoder)
        self.target_q_heads = deepcopy(self.q_heads)
        self.detach_q_heads = deepcopy(self.q_heads)
        self._set_targets_requires_grad(False)

    def _zero_init_distribution_heads(self) -> None:
        for head in [self.reward_head, *self.q_heads]:
            final_layer = next((module for module in reversed(head) if isinstance(module, nn.Linear)), None)
            if final_layer is not None:
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)

    def _set_targets_requires_grad(self, requires_grad: bool) -> None:
        for module in (self.target_encoder, self.target_q_heads, self.detach_q_heads):
            for param in module.parameters():
                param.requires_grad_(requires_grad)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.train(False)
        self.target_q_heads.train(False)
        self.detach_q_heads.train(False)
        return self

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target_param.lerp_(param, self.tau)
        for target_param, param in zip(self.target_q_heads.parameters(), self.q_heads.parameters(), strict=True):
            target_param.lerp_(param, self.tau)

    @torch.no_grad()
    def sync_detached_qs(self) -> None:
        for detach_param, param in zip(self.detach_q_heads.parameters(), self.q_heads.parameters(), strict=True):
            detach_param.copy_(param)

    def _apply_context(self, x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        """Concatenate the system-id context onto a component input.

        No-op when the history encoder is disabled. When ``context`` is ``None`` the
        learned (normalized) null context is broadcast over ``x``'s leading dims;
        otherwise ``context`` must already match those leading dims (callers that
        expand the latent per candidate expand the context the same way)."""
        if self.context_dim == 0:
            return x
        if context is None:
            assert self.history_encoder is not None
            null = self.history_encoder.null().to(dtype=x.dtype, device=x.device)
            context = null.expand(*x.shape[:-1], self.context_dim)
        return torch.cat([x, context], dim=-1)

    def encode(self, obs: torch.Tensor, context: torch.Tensor | None = None, target: bool = False) -> torch.Tensor:
        encoder = self.target_encoder if target else self.encoder
        return encoder(self._apply_context(obs, context))

    def encode_context(
        self,
        history_actions: torch.Tensor,
        history_transitions: torch.Tensor,
        history_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Encode a window of recent transitions into the system-id context vector.

        Returns ``None`` when the history encoder is disabled (``context_dim == 0``)."""
        if self.history_encoder is None:
            return None
        return self.history_encoder(history_actions, history_transitions, history_pad_mask)

    def next(self, z: torch.Tensor, actions: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return self.dynamics(self._apply_context(torch.cat([z, actions], dim=-1), context))

    def reward(self, z: torch.Tensor, actions: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.reward_logits(z, actions, context=context)
        return two_hot_inv(logits, self.dreg)

    def reward_logits(self, z: torch.Tensor, actions: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return self.reward_head(self._apply_context(torch.cat([z, actions], dim=-1), context))

    def continue_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.continue_head(z)

    def physical_features(self, z: torch.Tensor) -> torch.Tensor:
        if self.physical_head is None:
            raise RuntimeError("LatentWorldModel was created without physical feature prediction.")
        return self.physical_head(z)

    def _policy_stats(self, z: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.policy_head(self._apply_context(z, context)).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1.0)
        return mean, log_std

    def pi(self, z: torch.Tensor, deterministic: bool = True, return_info: bool = False, context: torch.Tensor | None = None):
        mean, log_std = self._policy_stats(z, context=context)
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
        detach: bool = False,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qs = self.Q_logits(z, actions, target=target, detach=detach, context=context)
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

    def Q_logits(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        target: bool = False,
        detach: bool = False,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target:
            heads = self.target_q_heads
        elif detach:
            heads = self.detach_q_heads
        else:
            heads = self.q_heads
        inputs = self._apply_context(torch.cat([z, actions], dim=-1), context)
        return torch.stack([head(inputs) for head in heads], dim=0)

    def encoder_parameters(self):
        yield from self.encoder.parameters()
        if self.history_encoder is not None:
            yield from self.history_encoder.parameters()

    def non_encoder_model_parameters(self):
        modules = (self.dynamics, self.reward_head, self.continue_head, self.q_heads)
        for module in modules:
            yield from module.parameters()
        if self.physical_head is not None:
            yield from self.physical_head.parameters()

    def model_parameters(self):
        modules = (self.encoder, self.dynamics, self.reward_head, self.continue_head, self.q_heads)
        for module in modules:
            yield from module.parameters()
        if self.history_encoder is not None:
            yield from self.history_encoder.parameters()
        if self.physical_head is not None:
            yield from self.physical_head.parameters()

    def policy_parameters(self):
        yield from self.policy_head.parameters()

    def policy_loss(self, zs: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        if context is None:
            context = getattr(self, "_last_context", None)
        if context is not None:
            # zs is [H+1, B, latent]; broadcast the per-segment context over the horizon.
            context = context.unsqueeze(0).expand(zs.shape[0], -1, -1)
        actions, info = self.pi(zs.detach(), deterministic=False, return_info=True, context=context)
        q = self.Q(zs.detach(), actions, return_type="avg", detach=True, context=context)
        self.q_scale.update(q[0])
        scaled_q = self.q_scale(q)
        rho = torch.pow(
            torch.as_tensor(self.rho, device=zs.device, dtype=zs.dtype),
            torch.arange(zs.shape[0], device=zs.device, dtype=zs.dtype),
        )
        per_step_loss = -(scaled_q + self.entropy_coef * info["scaled_entropy"]).mean(dim=(1, 2))
        loss = (per_step_loss * rho).mean()
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
        batch_dim = obs.shape[1]

        # One system-id context per sampled segment, held fixed across the horizon
        # (payAttentionDrift assumes a locally constant dynamics regime).
        context = None
        context_seq = None
        if self.context_dim > 0:
            context = self.encode_context(
                batch["history_actions"],
                batch["history_transitions"],
                batch.get("history_pad_mask"),
            )
            assert context is not None
            context_seq = context.unsqueeze(0).expand(horizon, -1, -1).reshape(-1, self.context_dim)

        with torch.no_grad():
            target_zs = self.encode(
                obs[1:].reshape(-1, self.obs_dim), context=context_seq, target=False
            ).view(horizon, batch_dim, -1)

        z = self.encode(obs[0], context=context)
        consistency_loss = torch.zeros((), device=obs.device)
        reward_loss = torch.zeros((), device=obs.device)
        value_loss = torch.zeros((), device=obs.device)
        continue_loss = torch.zeros((), device=obs.device)
        physical_loss = torch.zeros((), device=obs.device)
        rollout_zs = [z]
        physical_indices = None
        if self.physical_head is not None and self.loss_weights.physical > 0.0:
            physical_indices = torch.as_tensor(self.physical_feature_indices, device=obs.device, dtype=torch.long)

        for t in range(horizon):
            weight = self.rho**t
            action_t = actions[t]
            reward_t = rewards[t]
            continue_t = continues[t]
            z_target = target_zs[t]

            q_logits = self.Q_logits(z, action_t, context=context)
            reward_logits = self.reward_logits(z, action_t, context=context)
            z_next = self.next(z, action_t, context=context)
            continue_pred = self.continue_logits(z_next)

            with torch.no_grad():
                target_action = self.pi(z_target, deterministic=False, context=context)
                target_q = reward_t + self.discount * continue_t * self.Q(
                    z_target,
                    target_action,
                    target=True,
                    return_type="min",
                    context=context,
                )

            consistency_loss = consistency_loss + weight * F.mse_loss(z_next, z_target)
            reward_loss = reward_loss + weight * soft_ce(reward_logits, reward_t, self.dreg).mean()
            value_target = target_q.unsqueeze(0).expand(q_logits.shape[0], *target_q.shape)
            value_loss = value_loss + weight * soft_ce(q_logits.reshape(-1, q_logits.shape[-1]), value_target.reshape(-1, 1), self.dreg).mean()
            continue_loss = continue_loss + weight * F.binary_cross_entropy_with_logits(continue_pred, continue_t)
            if physical_indices is not None:
                physical_target = obs[t + 1].index_select(-1, physical_indices)
                physical_loss = physical_loss + weight * F.mse_loss(self.physical_features(z_next), physical_target)
            z = z_next
            rollout_zs.append(z)

        normalizer = sum(self.rho**t for t in range(horizon))
        consistency_loss = consistency_loss / normalizer
        reward_loss = reward_loss / normalizer
        value_loss = value_loss / normalizer
        continue_loss = continue_loss / normalizer
        physical_loss = physical_loss / normalizer

        weights = self.loss_weights
        total = (
            weights.consistency * consistency_loss
            + weights.reward * reward_loss
            + weights.value * value_loss
            + weights.continue_ * continue_loss
            + weights.physical * physical_loss
        )
        metrics = {
            "loss": float(total.detach().item()),
            "consistency_loss": float(consistency_loss.detach().item()),
            "reward_loss": float(reward_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "continue_loss": float(continue_loss.detach().item()),
            "physical_loss": float(physical_loss.detach().item()),
        }
        # Cached so the immediately following policy_loss() call conditions the actor/critic
        # on the same segment context (the training update runs loss -> policy_loss serially).
        self._last_context = context.detach() if context is not None else None
        return total, metrics, torch.stack(rollout_zs, dim=0).detach()
