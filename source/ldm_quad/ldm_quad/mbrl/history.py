"""Recent action and observation transitions for system identification."""

import torch


class RollingHistory:
    """Per-env ring buffer of recent (action, obs-transition) tokens for the history encoder.

    Fed one vectorized step at a time during rollout/eval, cleared per-env on episode
    reset. Provides the ``(actions, transitions, pad_mask)`` window the history encoder
    consumes. Order within the window does not matter (the encoder is permutation
    invariant and mean-pools valid steps).
    """

    def __init__(self, num_envs: int, history_len: int, action_dim: int, obs_dim: int, device: torch.device):
        self.history_len = history_len
        self.actions = torch.zeros((num_envs, history_len, action_dim), device=device)
        self.transitions = torch.zeros((num_envs, history_len, obs_dim), device=device)
        self.valid = torch.zeros((num_envs, history_len), dtype=torch.bool, device=device)
        self.ptr = 0

    def clear(self) -> None:
        self.valid.zero_()
        self.ptr = 0

    def append(self, actions: torch.Tensor, transitions: torch.Tensor, done: torch.Tensor | None = None) -> None:
        self.actions[:, self.ptr] = actions.detach()
        self.transitions[:, self.ptr] = transitions.detach()
        self.valid[:, self.ptr] = True
        self.ptr = (self.ptr + 1) % self.history_len
        if done is not None and done.any():
            # A finished episode starts fresh; drop its (now cross-boundary) history.
            self.valid[done.view(-1)] = False

    def context(self, model: torch.nn.Module) -> torch.Tensor | None:
        if not hasattr(model, "encode_context"):
            return None
        return model.encode_context(self.actions, self.transitions, ~self.valid)


CONTEXT_MODES = ("null", "rolling", "frozen", "truncated")


class ContextController:
    """Evaluation-time context policy over a :class:`RollingHistory` (play.py ``--context_mode``).

    Same interface as RollingHistory (``context(model)``, ``append(...)``); ``append`` is called
    once per env step and advances the step counter ``t`` that the modes key on.

    null      zero null context every step (with a pinned-zero null_context this is exactly the
              context-free walker).
    rolling   the last ``context_len`` valid transitions of each env (default mode).
    frozen    rolling until global step ``freeze_step``; from then on each env keeps the context
              it had at that step. An env that resets after the freeze gets the null context for
              the rest of the run (its pre-freeze identity belongs to a finished episode).
    truncated rolling, but at global step ``truncate_step`` every env's window is cleared, so the
              context is rebuilt from post-truncation transitions only (the oracle arm when the
              truncation coincides with a dynamics switch).

    ``context_len`` <= history_len keeps only the most recent K' slots; older ones are padded.

    ``ema_tau`` < 1 smooths the context actually used: c_used_t = (1 - tau) c_used_(t-1) + tau c_raw_t,
    per env, reset to c_raw when the env resets. Left unnormalised on purpose: every raw context is
    in the unit ball and a convex combination stays there, while rescaling to norm 1 would inflate
    the averaged direction exactly when successive contexts disagree (the jitter being damped).
    tau = 1 (default) is the raw context, bit for bit.
    """

    def __init__(
        self,
        num_envs: int,
        history_len: int,
        action_dim: int,
        obs_dim: int,
        device: torch.device,
        mode: str = "rolling",
        context_len: int | None = None,
        freeze_step: int | None = None,
        truncate_step: int | None = None,
        ema_tau: float = 1.0,
    ):
        if mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}, got {mode!r}")
        if mode == "frozen" and freeze_step is None:
            raise ValueError("context_mode=frozen needs a freeze step")
        if mode == "truncated" and truncate_step is None:
            raise ValueError("context_mode=truncated needs a truncate step")
        context_len = history_len if context_len is None else int(context_len)
        if not 1 <= context_len <= history_len:
            raise ValueError(f"context_len must be in [1, {history_len}], got {context_len}")
        self.history = RollingHistory(num_envs, history_len, action_dim, obs_dim, device)
        self.mode = mode
        self.context_len = context_len
        self.freeze_step = freeze_step
        self.truncate_step = truncate_step
        self.t = 0
        self._frozen: torch.Tensor | None = None
        self._null_after_freeze = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._prev: torch.Tensor | None = None
        self._last: torch.Tensor | None = None
        if not 0.0 < float(ema_tau) <= 1.0:
            raise ValueError(f"ema_tau must be in (0, 1], got {ema_tau}")
        self.ema_tau = float(ema_tau)
        self._ema: torch.Tensor | None = None
        self._ema_reset = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def pad_mask(self) -> torch.Tensor:
        """``[N, K]`` True for padded slots: invalid, or older than the ``context_len`` newest."""
        k = self.history.history_len
        slots = torch.arange(k, device=self.history.valid.device)
        age = (self.history.ptr - 1 - slots) % k  # 0 = most recent write
        return ~self.history.valid | (age >= self.context_len).unsqueeze(0)

    def _null(self, model: torch.nn.Module, n: int) -> torch.Tensor:
        return model.history_encoder.null().unsqueeze(0).expand(n, -1)

    def context(self, model: torch.nn.Module) -> torch.Tensor | None:
        if not hasattr(model, "encode_context") or getattr(model, "history_encoder", None) is None:
            return None
        n = self.history.valid.shape[0]
        if self.mode == "null":
            ctx = self._null(model, n).clone()
        elif self.mode == "frozen" and self._frozen is not None:
            ctx = torch.where(self._null_after_freeze.unsqueeze(-1), self._null(model, n), self._frozen)
        else:
            ctx = model.encode_context(self.history.actions, self.history.transitions, self.pad_mask())
            if self.mode == "frozen" and self.t >= self.freeze_step:
                self._frozen = ctx.detach().clone()
        if self.ema_tau < 1.0:
            raw = ctx.detach()
            if self._ema is None:
                self._ema = raw.clone()
            else:
                smooth = (1.0 - self.ema_tau) * self._ema + self.ema_tau * raw
                self._ema = torch.where(self._ema_reset.unsqueeze(-1), raw, smooth)
            self._ema_reset.zero_()
            ctx = self._ema.clone()
        self._prev, self._last = self._last, ctx.detach()
        return ctx

    def append(self, actions: torch.Tensor, transitions: torch.Tensor, done: torch.Tensor | None = None) -> None:
        self.history.append(actions, transitions, done)
        if done is not None:
            self._ema_reset |= done.view(-1).bool()
        if done is not None and self.mode == "frozen" and self._frozen is not None:
            self._null_after_freeze |= done.view(-1).bool()
        self.t += 1
        if self.mode == "truncated" and self.t == self.truncate_step:
            self.history.valid.zero_()  # the NEXT context sees post-truncation transitions only

    def metrics(self) -> dict[str, float]:
        if self._last is None:
            return {}
        drift = 0.0 if self._prev is None else float((self._last - self._prev).norm(dim=-1).mean())
        return {"context_norm_mean": float(self._last.norm(dim=-1).mean()), "context_drift_mean": drift}

    def describe(self) -> str:
        extra = {
            "frozen": f" freeze_step={self.freeze_step} (global step; post-freeze resets -> null context)",
            "truncated": f" truncate_step={self.truncate_step} (every env's window cleared at that step)",
        }.get(self.mode, "")
        ema = f" ema_tau={self.ema_tau}" if self.ema_tau < 1.0 else ""
        return f"context_mode={self.mode} context_len={self.context_len}/{self.history.history_len}{extra}{ema}"
