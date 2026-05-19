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
