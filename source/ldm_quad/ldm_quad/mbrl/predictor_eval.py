"""Passive-predictor evaluation (Level 3, prediction level).

A fixed acting policy drives the robot; one or more context-update schemes ("arms") of a
history-context world model are scored as PASSIVE predictors on the executed trajectory, so every
arm sees exactly the same states and actions. Torch-only (tests load it by path).

Per env step and arm (context from the history BEFORE the step, as in closed loop):
  phys1   one-step physical-head MSE: physical(next(encode(obs_t, c), a_t)) vs next_obs features
          (identical formula to play.py's model_physical_mse, so the null arm of a context model
          whose null context is zero reproduces a context-free model's number exactly)
  lat1    one-step latent consistency MSE vs encode(next_obs, c)
  phys4/8 open-loop physical MSE after k executed actions: from encode(obs_{t-k+1}, c_{t-k+1}),
          roll the dynamics forward on the actions actually executed, compare with the observed
          features of obs_{t+1}. Computed with a k-step delay, only for envs whose last k
          transitions lie in one episode (NaN when none qualify)
  ctx_norm, ctx_drift  of the arm's context
Arm names: null | rolling<K> | frozen<step> | truncated<step>   (K may exceed the model's
history_len; the set encoder takes any window length, but >48 is outside its training).
"""

from __future__ import annotations

import re

import torch

try:
    from .history import ContextController
    from .world_model import LatentWorldModel
except ImportError:  # loaded by file path (tests): import the sibling files the same way
    import importlib.util as _ilu
    import os as _os

    import sys as _sys

    def _sibling(name):
        key = f"_ldmq_{name}_pe"
        if key in _sys.modules:
            return _sys.modules[key]
        spec = _ilu.spec_from_file_location(key, _os.path.join(_os.path.dirname(__file__), f"{name}.py"))
        mod = _ilu.module_from_spec(spec)
        _sys.modules[key] = mod  # dataclasses resolve their module through sys.modules
        spec.loader.exec_module(mod)
        return mod

    ContextController = _sibling("history").ContextController
    LatentWorldModel = _sibling("world_model").LatentWorldModel

ARM_RE = re.compile(r"^(null|rolling(\d+)|frozen(\d+)|truncated(\d+))$")
KS = (4, 8)
METRICS = ("phys1", "lat1", "phys4", "phys8", "ctx_norm", "ctx_drift")


def _indices(value) -> list[int]:
    return [int(i) for i in str(value or "").split(",") if str(i).strip()]


def build_predictor_model(path: str, obs_dim: int, action_dim: int, device) -> "LatentWorldModel":
    """Rebuild a latent world model exactly as its checkpoint was trained (structure from its own
    args: latent/hidden/depth, physical indices, command skip, history context and
    context_components), load it strictly, and return it in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a, sd = ckpt["args"], ckpt["model"]
    if a.get("model_type", "latent") != "latent":
        raise ValueError(f"{path}: passive prediction needs a latent world model, got {a.get('model_type')!r}")
    model = LatentWorldModel(
        obs_dim=obs_dim, action_dim=action_dim, latent_dim=a.get("latent_dim", 128), hidden_dim=a["hidden_dim"],
        depth=a["model_depth"], num_q=a.get("num_q", 5), discount=a["discount"], tau=a.get("target_tau", 0.01),
        rho=a.get("rho", 0.5), entropy_coef=a.get("entropy_coef", 1e-4), num_bins=a.get("num_bins", 101),
        vmin=a.get("vmin", -10.0), vmax=a.get("vmax", 10.0), simnorm_dim=a.get("simnorm_dim", 8),
        q_dropout=a.get("q_dropout", 0.01), physical_feature_indices=_indices(a.get("latent_physical_indices")),
        command_indices=_indices(a.get("command_skip_indices")),
        context_dim=int(a.get("history_context_dim") or 0), history_len=int(a.get("history_len") or 48),
        history_d_model=a.get("history_d_model", 64), history_nhead=a.get("history_nhead", 4),
        history_layers=a.get("history_layers", 1), history_ff=a.get("history_ff", 256),
        history_dropout=a.get("history_dropout", 0.1), context_components=a.get("context_components") or "all",
    )
    model.load_state_dict(sd, strict=True)
    if not model.physical_feature_indices:
        raise ValueError(f"{path}: model has no physical head; passive prediction scores physical features")
    return model.to(device).eval()


def parse_arm(name: str, history_len: int) -> dict:
    m = ARM_RE.match(name)
    if not m:
        raise ValueError(f"bad predictor arm {name!r}: use null | rolling<K> | frozen<step> | truncated<step>")
    if name == "null":
        return {"mode": "null", "history_len": history_len}
    if m.group(2):
        k = int(m.group(2))
        return {"mode": "rolling", "history_len": max(k, 1), "context_len": k}
    if m.group(3):
        return {"mode": "frozen", "history_len": history_len, "freeze_step": int(m.group(3))}
    return {"mode": "truncated", "history_len": history_len, "truncate_step": int(m.group(4))}


class PredictorEvaluator:
    def __init__(self, model: torch.nn.Module, arm_names: list[str], num_envs: int, obs_dim: int,
                 action_dim: int, device: torch.device):
        self.model = model
        self.device = device
        self.phys_idx = torch.as_tensor(model.physical_feature_indices, dtype=torch.long, device=device)
        self.lat = int(model.latent_dim)
        has_ctx = getattr(model, "context_dim", 0) > 0
        self.arms: dict[str, ContextController | None] = {}
        for name in arm_names:
            spec = parse_arm(name, int(getattr(model, "history_len", 48) or 48))
            if not has_ctx:
                self.arms[name] = None  # context-free predictor: every arm is the plain model
                continue
            self.arms[name] = ContextController(
                num_envs, spec["history_len"], action_dim, obs_dim, device, mode=spec["mode"],
                context_len=spec.get("context_len"), freeze_step=spec.get("freeze_step"),
                truncate_step=spec.get("truncate_step"),
            )
        kmax = max(KS)
        self._obs = torch.zeros(kmax, num_envs, obs_dim, device=device)       # obs at each origin step
        self._act = torch.zeros(kmax, num_envs, action_dim, device=device)    # action executed at that step
        self._ctx: dict[str, list] = {name: [None] * kmax for name in self.arms}
        self._since_reset = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.t = 0

    @torch.no_grad()
    def step(self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor, done: torch.Tensor) -> dict[str, float]:
        """Score every arm on (obs_t, a_t, next_obs_t), then advance their histories."""
        kmax = max(KS)
        slot = self.t % kmax
        self._obs[slot] = obs
        self._act[slot] = actions
        self._since_reset += 1  # transitions in the current episode, including this one
        target1 = next_obs.index_select(-1, self.phys_idx)
        out: dict[str, float] = {}
        for name, ctrl in self.arms.items():
            c = ctrl.context(self.model) if ctrl is not None else None
            self._ctx[name][slot] = c
            z = self.model.encode(obs, context=c)
            z1 = self.model.next(z, actions, context=c)
            out[f"{name}_phys1"] = float((self.model.physical_features(z1) - target1).square().mean())
            zt = self.model.encode(next_obs, context=c)
            out[f"{name}_lat1"] = float((z1[..., : self.lat] - zt[..., : self.lat]).square().mean())
            for k in KS:
                # last k transitions in one episode, and this step's target is not a post-reset obs
                valid = (self._since_reset >= k) & ~done.view(-1).bool()
                if self.t + 1 < k or not bool(valid.any()):
                    out[f"{name}_phys{k}"] = float("nan")
                    continue
                o = (self.t - k + 1) % kmax
                c0 = self._ctx[name][o]
                zk = self.model.encode(self._obs[o], context=c0)
                for j in range(k):
                    zk = self.model.next(zk, self._act[(o + j) % kmax], context=c0)
                err = (self.model.physical_features(zk) - target1).square().mean(-1)
                out[f"{name}_phys{k}"] = float(err[valid].mean())
            if ctrl is not None:
                ms = ctrl.metrics()
                out[f"{name}_ctx_norm"] = ms.get("context_norm_mean", float("nan"))
                out[f"{name}_ctx_drift"] = ms.get("context_drift_mean", float("nan"))
                ctrl.append(actions, next_obs - obs, done)
            else:
                out[f"{name}_ctx_norm"] = 0.0
                out[f"{name}_ctx_drift"] = 0.0
        self._since_reset[done.view(-1).bool()] = 0
        self.t += 1
        return {f"predictor_{k}": v for k, v in out.items()}

    def field_names(self) -> list[str]:
        return [f"predictor_{name}_{m}" for name in self.arms for m in METRICS]
