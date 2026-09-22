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
