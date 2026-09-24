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


class ContextSequential(nn.Sequential):
    """``nn.Sequential`` whose first Linear also takes an additive context term.

    ``first(x) + context @ context_weight.T`` is formed before the first LayerNorm /
    activation. ``context_weight`` is zero-initialised, so a freshly built context model
    is bit-identical to the context-free one for ANY context, and a context-free
    checkpoint loads with unchanged key names (only ``*.context_weight`` is new).
    It is a Parameter rather than a child module so indexing/iterating the Sequential
    (e.g. "last Linear" lookups) still sees only the original layers. When the first
    child is itself a ContextSequential (latent_mlp) the context is forwarded to it.
    """

    def __init__(self, *layers: nn.Module, context_dim: int = 0):
        super().__init__(*layers)
        first = self[0]
        if context_dim > 0 and not isinstance(first, ContextSequential):
            self.context_weight = nn.Parameter(torch.zeros(first.out_features, context_dim))
        else:
            self.register_parameter("context_weight", None)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        layers = iter(self)
        first = next(layers)
        if isinstance(first, ContextSequential):
            h = first(x, context)
        else:
            h = first(x)
            if self.context_weight is not None and context is not None:
                h = h + F.linear(context, self.context_weight)
        for layer in layers:
            h = layer(h)
        return h


def mlp(
    input_dim: int, hidden_dim: int, output_dim: int, depth: int, dropout: float = 0.0, context_dim: int = 0
) -> nn.Sequential:
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
    return ContextSequential(*layers, context_dim=context_dim)


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


def latent_mlp(
    input_dim: int, hidden_dim: int, latent_dim: int, depth: int, simnorm_dim: int, context_dim: int = 0
) -> nn.Sequential:
    layers: list[nn.Module] = [mlp(input_dim, hidden_dim, latent_dim, depth, context_dim=context_dim)]
    if simnorm_dim > 1:
        layers.append(SimNorm(simnorm_dim))
    return ContextSequential(*layers)


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
        # Frozen (SIT adapter mode): keep the loaded scale. The EMA tracks the Q spread of the
        # current buffer, and a narrow buffer would otherwise rescale the actor loss.
        self.frozen = False

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
        if self.frozen:
            return
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
    vector. Empty histories fall back to ``null_context``, a fixed zero buffer (not learned):
    with the additive context projections, an empty history therefore reproduces the
    context-free model exactly, which is what defines the null-context arm.
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
        self.register_buffer("null_context", torch.zeros(context_dim))

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


def expand_state_dict_for_command_skip(old_sd: dict, model: "LatentWorldModel") -> dict:
    """Fit a no-skip checkpoint into a command-skip model WITHOUT changing its outputs.

    Widening a head's first Linear shifts the action columns, so a plain
    load_state_dict(strict=False) is not merely lossy here -- the shapes mismatch, the layer is
    skipped entirely, and the head silently reverts to random init. Every first layer is
    therefore rebuilt column-by-column:

        old [z | a]        ->  new [z | cmd | a]
             ^^^ copied          ^^^   ^^^   ^^^ copied, shifted right by command_dim
                                       zero-init

    Zeroing the command columns makes the grafted model numerically identical to the original
    at load time, so an A/B measures what the skip-connection LEARNS rather than the shock of
    re-initialised weights.
    """
    cd = model.command_dim
    if cd == 0:
        return dict(old_sd)
    lat, act = model.latent_dim, model.action_dim
    new_sd = model.state_dict()
    # first-layer key -> does its input carry the action after z?
    specs: dict[str, bool] = {
        "dynamics.0.0.weight": True,
        "reward_head.0.weight": True,
        "continue_head.0.weight": False,
        "policy_head.0.weight": False,
        "physical_head.0.weight": False,
    }
    for i in range(model.num_q):
        for prefix in ("q_heads", "target_q_heads", "detach_q_heads"):
            specs[f"{prefix}.{i}.0.weight"] = True

    out, grafted = {}, []
    for key, value in old_sd.items():
        if key in specs and key in new_sd and torch.is_tensor(value) and value.ndim == 2:
            has_action = specs[key]
            expected_old = lat + (act if has_action else 0)
            if value.shape[-1] != expected_old:
                raise ValueError(
                    f"{key}: expected old in_features {expected_old}, found {value.shape[-1]}. "
                    "Refusing to graft -- the column layout is not what this function assumes."
                )
            target = new_sd[key]
            if target.shape[-1] != expected_old + cd:
                raise ValueError(
                    f"{key}: new in_features {target.shape[-1]} != {expected_old + cd}."
                )
            fresh = torch.zeros_like(target)
            fresh[:, :lat] = value[:, :lat]
            if has_action:
                fresh[:, lat + cd : lat + cd + act] = value[:, lat : lat + act]
            out[key] = fresh
            grafted.append(key)
        else:
            out[key] = value
    # History-context tensors are ADDED, not widened; the context graft
    # (mbrl/checkpoint.py) validates and initialises them, so they may be absent here.
    missing = [
        k for k in new_sd
        if k not in out and not (k.endswith("context_weight") or k.startswith("history_encoder."))
    ]
    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} key(s), e.g. {missing[:4]}")
    print(f"[#6a] grafted {len(grafted)} first-layer(s) for command_dim={cd}: {sorted(grafted)}", flush=True)
    return out


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
        bc_coef: float = 0.0,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        simnorm_dim: int = 8,
        q_dropout: float = 0.01,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        physical_feature_indices: list[int] | tuple[int, ...] | None = None,
        command_indices: list[int] | tuple[int, ...] | None = None,
        loss_weights: WorldModelLossWeights | None = None,
        context_dim: int = 0,
        history_len: int = 0,
        history_d_model: int = 64,
        history_nhead: int = 4,
        history_layers: int = 1,
        history_ff: int = 256,
        history_dropout: float = 0.1,
        context_components: str = "all",
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
        # TD-M(PC)^2 (arXiv 2502.03550) behavior-cloning coefficient (beta). When
        # 0.0 the actor loss is identical to vanilla TD-MPC2: the BC term below is
        # skipped entirely, so this is a no-op unless explicitly enabled.
        self.bc_coef = bc_coef
        self.dreg = DistributionalRegressionCfg(num_bins=num_bins, vmin=vmin, vmax=vmax)
        self.simnorm_dim = simnorm_dim
        self.q_dropout = q_dropout
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.physical_feature_indices = tuple(physical_feature_indices or ())
        # #6a COMMAND SKIP-CONNECTION.
        # Every head reads only z, and z is a 256-d SimNorm bottleneck of the 48-d obs. If the
        # command (obs[9:12]) is not preserved faithfully through that bottleneck, the reward
        # head cannot tell WHICH axis is being commanded and fits an axis-averaged reward --
        # a concrete mechanism for axis trading living in the world model rather than in pi.
        # This routes the raw command around the encoder and into every head. Default () keeps
        # the architecture bit-identical to every existing checkpoint.
        self.command_indices = tuple(command_indices or ())
        self.command_dim = len(self.command_indices)
        if self.command_dim:
            self.register_buffer(
                "_command_index_t",
                torch.as_tensor(self.command_indices, dtype=torch.long),
                persistent=False,
            )
        # Width of what encode()/next() hand to the heads.
        self.z_dim = latent_dim + self.command_dim
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
        # The system-id context conditions every component TD-MPC2 conditions on its task
        # embedding e: h(s,e), d(z,a,e), R(z,a,e), Q(z,a,e), p(z,e). It is ADDED at each first
        # layer through a zero-initialised projection (ContextSequential) rather than
        # concatenated, so base-MLP shapes and key names match a context-free checkpoint and
        # a warm-started model reproduces it exactly until the projections learn.
        # continue/physical heads read the (already context-aware) latent directly.
        ctx = self.context_dim
        # Where the context is wired in at CONSTRUCTION. "all": every conditioned component (above).
        # "dynamics_only" (payAttentionDrift's structure): only d(z, a, c) sees it; the encoder,
        # reward, Q and policy are the context-free model's, so no other context_weight exists.
        if context_components not in ("all", "dynamics_only"):
            raise ValueError(f"context_components must be 'all' or 'dynamics_only', got {context_components!r}")
        self.context_components = context_components
        ctx_heads = ctx if context_components == "all" else 0
        z_dim = self.z_dim
        self.encoder = latent_mlp(obs_dim, hidden_dim, latent_dim, depth, simnorm_dim, context_dim=ctx_heads)
        self.dynamics = latent_mlp(z_dim + action_dim, hidden_dim, latent_dim, depth, simnorm_dim, context_dim=ctx)
        self.reward_head = mlp(z_dim + action_dim, hidden_dim, head_dim, depth, context_dim=ctx_heads)
        self.continue_head = mlp(z_dim, hidden_dim, 1, depth)
        self.policy_head = mlp(z_dim, hidden_dim, 2 * action_dim, depth, context_dim=ctx_heads)
        self.physical_head = (
            mlp(z_dim, hidden_dim, len(self.physical_feature_indices), depth)
            if self.physical_feature_indices
            else None
        )
        self.q_heads = nn.ModuleList(
            mlp(z_dim + action_dim, hidden_dim, head_dim, depth, dropout=q_dropout, context_dim=ctx_heads)
            for _ in range(num_q)
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
        # Only tensors that are being trained are tracked. With every online tensor trainable
        # (the default) this is the plain EMA; in SIT adapter mode the frozen base targets keep
        # their loaded values exactly instead of creeping toward the frozen online weights.
        for target_param, param in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            if param.requires_grad:
                target_param.lerp_(param, self.tau)
        for target_param, param in zip(self.target_q_heads.parameters(), self.q_heads.parameters(), strict=True):
            if param.requires_grad:
                target_param.lerp_(param, self.tau)

    @torch.no_grad()
    def sync_detached_qs(self) -> None:
        for detach_param, param in zip(self.detach_q_heads.parameters(), self.q_heads.parameters(), strict=True):
            detach_param.copy_(param)

    def _resolve_context(self, x: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor | None:
        """System-id context for a component whose input is ``x``.

        ``None`` when the history encoder is disabled. When ``context`` is ``None`` the
        learned (normalized) null context is broadcast over ``x``'s leading dims;
        otherwise ``context`` must already match those leading dims (callers that
        expand the latent per candidate expand the context the same way). The context
        enters each conditioned MLP additively at its first layer (ContextSequential)."""
        if self.context_dim == 0:
            return None
        if context is None:
            assert self.history_encoder is not None
            null = self.history_encoder.null().to(dtype=x.dtype, device=x.device)
            context = null.expand(*x.shape[:-1], self.context_dim)
        return context

    def encode(self, obs: torch.Tensor, context: torch.Tensor | None = None, target: bool = False) -> torch.Tensor:
        encoder = self.target_encoder if target else self.encoder
        z = encoder(obs, self._resolve_context(obs, context))
        if self.command_dim:
            z = torch.cat([z, obs.index_select(-1, self._command_index_t)], dim=-1)
        return z

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
        inputs = torch.cat([z, actions], dim=-1)
        z_next = self.dynamics(inputs, self._resolve_context(inputs, context))
        if self.command_dim:
            # The command is exogenous; keep it fixed during a model rollout.
            z_next = torch.cat([z_next, z[..., -self.command_dim:]], dim=-1)
        return z_next

    def reward(self, z: torch.Tensor, actions: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.reward_logits(z, actions, context=context)
        return two_hot_inv(logits, self.dreg)

    def reward_logits(self, z: torch.Tensor, actions: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        inputs = torch.cat([z, actions], dim=-1)
        return self.reward_head(inputs, self._resolve_context(inputs, context))

    def continue_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.continue_head(z)

    def physical_features(self, z: torch.Tensor) -> torch.Tensor:
        if self.physical_head is None:
            raise RuntimeError("LatentWorldModel was created without physical feature prediction.")
        return self.physical_head(z)

    def _policy_stats(self, z: torch.Tensor, context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.policy_head(z, self._resolve_context(z, context)).chunk(2, dim=-1)
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
        inputs = torch.cat([z, actions], dim=-1)
        context = self._resolve_context(inputs, context)
        return torch.stack([head(inputs, context) for head in heads], dim=0)

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

    def adapter_parameters(self):
        """SIT adapter tensors trained by the world-model optimizer: the history encoder and the
        online context projections of encoder, dynamics, reward and Q heads (not target/detach
        copies, not the policy's)."""
        if self.history_encoder is not None:
            yield from self.history_encoder.parameters()
        for module in (self.encoder, self.dynamics, self.reward_head, *self.q_heads):
            for name, param in module.named_parameters():
                if name.endswith("context_weight"):
                    yield param

    def policy_adapter_parameters(self):
        if self.policy_head.context_weight is not None:
            yield self.policy_head.context_weight

    CONTEXT_MASKS = ("none", "dynamics_only", "dynamics_strict")
    _MASKED_HEADS = {
        "dynamics_only": ("reward_head", "q_heads", "target_q_heads", "detach_q_heads", "policy_head"),
        "dynamics_strict": ("encoder", "target_encoder", "reward_head", "q_heads", "target_q_heads",
                            "detach_q_heads", "policy_head"),
    }

    @torch.no_grad()
    def restrict_context(self, mask: str = "none") -> list[str]:
        """Evaluation-time ablation of where the system-id context acts (zeroes projections).

        ``dynamics_only`` zeroes the reward head, every Q-head copy and the policy head: the
        planner's objective is the context-free model's, but the ENCODER still sees the context,
        so the latent those heads read is context-shifted. ``dynamics_strict`` also zeroes the
        encoder (and its target), so the context acts only inside the dynamics MLP. Heads that were
        co-adapted with the context through z are not recovered by masking either way.
        Returns the zeroed parameter names.
        """
        if mask not in self.CONTEXT_MASKS:
            raise ValueError(f"mask must be one of {self.CONTEXT_MASKS}, got {mask!r}")
        if mask == "none" or self.context_dim == 0:
            return []
        zeroed = []
        for name, param in self.named_parameters():
            if name.endswith("context_weight") and name.split(".")[0] in self._MASKED_HEADS[mask]:
                param.zero_()
                zeroed.append(name)
        return zeroed

    def freeze_base_for_adapter(self) -> tuple[int, int]:
        """Freeze every online weight except the SIT adapter, and the actor-loss Q scale.

        Returns (trainable, frozen) parameter counts."""
        if self.context_dim == 0:
            raise ValueError("adapter-only training needs a history-context model (context_dim > 0)")
        adapter = {id(p) for p in self.adapter_parameters()} | {id(p) for p in self.policy_adapter_parameters()}
        trainable = frozen = 0
        for param in self.parameters():
            if id(param) in adapter:
                param.requires_grad_(True)
                trainable += param.numel()
            else:
                if param.requires_grad:
                    frozen += param.numel()
                param.requires_grad_(False)
        self.q_scale.frozen = True
        return trainable, frozen

    def policy_loss(
        self,
        zs: torch.Tensor,
        planner_mean: torch.Tensor | None = None,
        planner_std: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
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

        # TD-M(PC)^2 behavior-cloning term: pull the freshly-sampled actor action
        # toward the MPC planner's stored Gaussian mu ~ pi_H (arXiv 2502.03550).
        # mu is a Gaussian defined directly over the (already tanh-squashed) action
        # space, so log_mu is a plain diagonal-Gaussian log-density -- no tanh
        # Jacobian correction (that correction only applies to pi's own density).
        # rollout state t aligns 1:1 with real transition t, so planner_mean/std
        # (H steps) match actions[:H]. Steps with no stored planner Gaussian (seed
        # phase, model-bootstrap, or legacy buffers) carry a NaN sentinel and are
        # masked out. bc_coef == 0.0 makes this a no-op.
        bc_logmu_mean = 0.0
        if self.bc_coef > 0.0 and planner_mean is not None and planner_std is not None:
            horizon = planner_mean.shape[0]
            a_bc = actions[:horizon]
            valid = torch.isfinite(planner_mean).all(dim=-1, keepdim=True) & torch.isfinite(planner_std).all(dim=-1, keepdim=True)
            valid_f = valid.to(a_bc.dtype)
            # CRITICAL: sanitize the sentinel NaNs BEFORE any math. Masking the
            # forward with torch.where(valid, log_mu, 0) is NOT enough -- autograd
            # backprops 0*NaN = NaN through the invalid (seed/bootstrap/legacy)
            # steps, poisoning policy_head and, via self.pi(), the model loss and
            # planner. nan_to_num keeps NaN out of the graph entirely; the
            # multiplicative valid_f mask then zeroes invalid steps cleanly
            # (finite*0 = 0, with 0 gradient).
            safe_mean = torch.nan_to_num(planner_mean, nan=0.0)
            safe_std = torch.nan_to_num(planner_std, nan=1.0).clamp_min(1e-3)
            two_pi = torch.log(torch.tensor(2.0 * torch.pi, device=zs.device, dtype=zs.dtype))
            log_mu = -0.5 * (((a_bc - safe_mean) / safe_std).square() + 2.0 * safe_std.log() + two_pi)
            log_mu = log_mu.sum(dim=-1, keepdim=True) * valid_f
            # Normalize by S_q (same running scale as the Q term) without updating it.
            scaled_log_mu = self.q_scale(log_mu)
            denom = valid_f.sum(dim=(1, 2)).clamp_min(1.0)
            bc_per_step = (scaled_log_mu * valid_f).sum(dim=(1, 2)) / denom
            bc_loss = -(self.bc_coef * bc_per_step * rho[:horizon]).mean()
            loss = loss + bc_loss
            bc_logmu_mean = float((log_mu * valid_f).sum().detach().item() / denom.sum().clamp_min(1.0).item())

        metrics = {
            "policy_loss": float(loss.detach().item()),
            "policy_q": float(q.detach().mean().item()),
            "policy_scaled_q": float(scaled_q.detach().mean().item()),
            "policy_entropy": float(info["entropy"].detach().mean().item()),
            "policy_scaled_entropy": float(info["scaled_entropy"].detach().mean().item()),
            "policy_q_scale": float(self.q_scale.value.detach().mean().item()),
            "policy_bc_logmu": bc_logmu_mean,
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

            # Compare the LEARNED latent only. next() carries the command forward from step t,
            # while z_target holds the command at t+1; on a command-resample step those differ,
            # and penalising that would charge the dynamics for an unpredictable exogenous
            # change. Slicing is a no-op when the skip-connection is off.
            consistency_loss = consistency_loss + weight * F.mse_loss(
                z_next[..., : self.latent_dim], z_target[..., : self.latent_dim]
            )
            reward_loss = reward_loss + weight * soft_ce(reward_logits, reward_t, self.dreg).mean()
            value_target = target_q.unsqueeze(0).expand(q_logits.shape[0], *target_q.shape)
            value_loss = value_loss + weight * soft_ce(q_logits.reshape(-1, q_logits.shape[-1]), value_target.reshape(-1, 1), self.dreg).mean()
            # Class-balance the continuation/termination BCE. Terminations
            # (continue_t == 0) are only ~1% of transitions, so unweighted BCE
            # collapses to "always alive": the continue head saturates near 1 and
            # the planner never sees a predicted fall, feeding the over-optimism.
            # Up-weight the rare terminal transitions so falls are actually learned.
            continue_bce_weight = torch.where(
                continue_t < 0.5,
                continue_t.new_tensor(10.0),
                continue_t.new_tensor(1.0),
            )
            continue_loss = continue_loss + weight * F.binary_cross_entropy_with_logits(
                continue_pred, continue_t, weight=continue_bce_weight
            )
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
