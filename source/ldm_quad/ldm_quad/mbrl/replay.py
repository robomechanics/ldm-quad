from __future__ import annotations

import torch


class ReplayBuffer:
    """Simple replay buffer storing vectorized transitions on CPU."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.continues = torch.zeros((capacity, 1), dtype=torch.float32)
        self.ptr = 0
        self.size = 0
        self._last_batch_size = 1

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        continues: torch.Tensor,
    ) -> None:
        batch_size = obs.shape[0]
        self._last_batch_size = batch_size
        if batch_size > self.capacity:
            obs = obs[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            continues = continues[-self.capacity :]
            batch_size = self.capacity

        start = self.ptr
        end = start + batch_size

        if end <= self.capacity:
            self.obs[start:end] = obs
            self.actions[start:end] = actions
            self.rewards[start:end] = rewards
            self.next_obs[start:end] = next_obs
            self.continues[start:end] = continues
        else:
            first = self.capacity - start
            second = end - self.capacity
            self.obs[start:] = obs[:first]
            self.actions[start:] = actions[:first]
            self.rewards[start:] = rewards[:first]
            self.next_obs[start:] = next_obs[:first]
            self.continues[start:] = continues[:first]
            self.obs[:second] = obs[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.next_obs[:second] = next_obs[first:]
            self.continues[:second] = continues[first:]

        self.ptr = end % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,))
        return {
            "obs": self.obs[indices].to(device),
            "actions": self.actions[indices].to(device),
            "rewards": self.rewards[indices].to(device),
            "next_obs": self.next_obs[indices].to(device),
            "continues": self.continues[indices].to(device),
        }

    def can_sample_sequences(self, batch_size: int, horizon: int) -> bool:
        stride = max(int(self._last_batch_size), 1)
        return self.size >= batch_size and self.size >= horizon * stride + 1

    def sample_sequences(
        self,
        batch_size: int,
        horizon: int,
        device: torch.device | str,
        max_attempts: int = 10000,
    ) -> dict[str, torch.Tensor]:
        """Sample same-env contiguous transition sequences.

        The buffer is filled by vectorized env steps, so transition `i + num_envs`
        is the next transition for the same environment as transition `i`.
        """

        stride = max(int(self._last_batch_size), 1)
        if not self.can_sample_sequences(batch_size, horizon):
            raise ValueError(
                f"Cannot sample horizon={horizon} sequences from buffer size={self.size} "
                f"with vectorized stride={stride}."
            )

        starts: list[int] = []
        attempts = 0
        valid_limit = self.capacity if self.size == self.capacity else self.size
        while len(starts) < batch_size and attempts < max_attempts:
            attempts += 1
            start = int(torch.randint(0, valid_limit, ()).item())
            indices = (start + torch.arange(horizon + 1) * stride) % self.capacity
            if self.size < self.capacity and int(indices[-1].item()) >= self.size:
                continue
            if horizon > 1 and self.continues[indices[:-2]].min().item() <= 0.0:
                continue
            starts.append(start)

        if len(starts) < batch_size:
            raise RuntimeError(
                f"Only found {len(starts)} valid horizon={horizon} sequences after {max_attempts} attempts. "
                "Collect more non-terminal data or reduce the sequence horizon."
            )

        start_t = torch.as_tensor(starts, dtype=torch.long)
        indices = (start_t.unsqueeze(0) + torch.arange(horizon + 1).unsqueeze(1) * stride) % self.capacity
        transition_indices = indices[:-1]
        return {
            "obs": self.obs[indices].to(device),
            "actions": self.actions[transition_indices].to(device),
            "rewards": self.rewards[transition_indices].to(device),
            "continues": self.continues[transition_indices].to(device),
        }
