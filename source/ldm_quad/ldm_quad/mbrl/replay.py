from __future__ import annotations

import torch


class ReplayBuffer:
    """Vectorized transition replay buffer.

    Storage lives on ``device`` (default CPU). Keeping it on the training GPU
    avoids a host<->device copy on every sampled batch and removes the swap
    pressure of a large CPU-resident buffer, at the cost of a few hundred MB of
    VRAM. Checkpoints are always written on CPU for portability.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device | str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=self.device)
        self.next_obs = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.continues = torch.zeros((capacity, 1), dtype=torch.float32, device=self.device)
        self.env_ids = -torch.ones(capacity, dtype=torch.long, device=self.device)
        self.episode_ids = -torch.ones(capacity, dtype=torch.long, device=self.device)
        self.step_ids = -torch.ones(capacity, dtype=torch.long, device=self.device)
        self.ptr = 0
        self.size = 0
        self._last_batch_size = 1
        self._env_episode_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        self._env_step_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        self._valid_start_cache: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict[str, object]:
        """Return a CPU checkpoint payload for exact replay-buffer resume."""
        return {
            "capacity": self.capacity,
            "obs_dim": int(self.obs.shape[-1]),
            "action_dim": int(self.actions.shape[-1]),
            "obs": self.obs.cpu(),
            "actions": self.actions.cpu(),
            "next_obs": self.next_obs.cpu(),
            "rewards": self.rewards.cpu(),
            "continues": self.continues.cpu(),
            "env_ids": self.env_ids.cpu(),
            "episode_ids": self.episode_ids.cpu(),
            "step_ids": self.step_ids.cpu(),
            "ptr": self.ptr,
            "size": self.size,
            "last_batch_size": self._last_batch_size,
            "env_episode_ids": self._env_episode_ids.cpu(),
            "env_step_ids": self._env_step_ids.cpu(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore a replay buffer saved by :meth:`state_dict`.

        ``copy_`` handles the CPU-checkpoint -> current-device transfer, so a
        buffer restored onto the GPU loads transparently from a CPU checkpoint.
        """
        capacity = int(state_dict.get("capacity", -1))
        obs_dim = int(state_dict.get("obs_dim", -1))
        action_dim = int(state_dict.get("action_dim", -1))
        if capacity != self.capacity or obs_dim != self.obs.shape[-1] or action_dim != self.actions.shape[-1]:
            raise ValueError(
                "Replay checkpoint shape mismatch: "
                f"checkpoint capacity={capacity} obs_dim={obs_dim} action_dim={action_dim}, "
                f"current capacity={self.capacity} obs_dim={self.obs.shape[-1]} action_dim={self.actions.shape[-1]}"
            )

        self.obs.copy_(state_dict["obs"])
        self.actions.copy_(state_dict["actions"])
        self.next_obs.copy_(state_dict["next_obs"])
        self.rewards.copy_(state_dict["rewards"])
        self.continues.copy_(state_dict["continues"])
        self.env_ids.copy_(state_dict["env_ids"])
        self.episode_ids.copy_(state_dict["episode_ids"])
        self.step_ids.copy_(state_dict["step_ids"])
        self.ptr = int(state_dict["ptr"])
        self.size = int(state_dict["size"])
        self._last_batch_size = int(state_dict.get("last_batch_size", 1))
        self._env_episode_ids = state_dict["env_episode_ids"].clone().to(device=self.device, dtype=torch.long)
        self._env_step_ids = state_dict["env_step_ids"].clone().to(device=self.device, dtype=torch.long)
        self._valid_start_cache.clear()

    def add_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        continues: torch.Tensor,
        resets: torch.Tensor | None = None,
    ) -> None:
        obs = obs.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True)
        next_obs = next_obs.to(self.device, non_blocking=True)
        continues = continues.to(self.device, non_blocking=True)
        if resets is not None:
            resets = resets.to(self.device, non_blocking=True)

        batch_size = obs.shape[0]
        self._last_batch_size = batch_size
        if self._env_episode_ids.numel() != batch_size:
            self._env_episode_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            self._env_step_ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        if batch_size > self.capacity:
            obs = obs[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            continues = continues[-self.capacity :]
            batch_size = self.capacity

        env_ids = torch.arange(batch_size, dtype=torch.long, device=self.device)
        episode_ids = self._env_episode_ids[:batch_size].clone()
        step_ids = self._env_step_ids[:batch_size].clone()
        start = self.ptr
        end = start + batch_size

        if end <= self.capacity:
            self.obs[start:end] = obs
            self.actions[start:end] = actions
            self.rewards[start:end] = rewards
            self.next_obs[start:end] = next_obs
            self.continues[start:end] = continues
            self.env_ids[start:end] = env_ids
            self.episode_ids[start:end] = episode_ids
            self.step_ids[start:end] = step_ids
        else:
            first = self.capacity - start
            second = end - self.capacity
            self.obs[start:] = obs[:first]
            self.actions[start:] = actions[:first]
            self.rewards[start:] = rewards[:first]
            self.next_obs[start:] = next_obs[:first]
            self.continues[start:] = continues[:first]
            self.env_ids[start:] = env_ids[:first]
            self.episode_ids[start:] = episode_ids[:first]
            self.step_ids[start:] = step_ids[:first]
            self.obs[:second] = obs[first:]
            self.actions[:second] = actions[first:]
            self.rewards[:second] = rewards[first:]
            self.next_obs[:second] = next_obs[first:]
            self.continues[:second] = continues[first:]
            self.env_ids[:second] = env_ids[first:]
            self.episode_ids[:second] = episode_ids[first:]
            self.step_ids[:second] = step_ids[first:]

        self.ptr = end % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
        self._valid_start_cache.clear()
        done = resets.view(-1).bool() if resets is not None else continues.view(-1) <= 0.0
        self._env_step_ids[:batch_size] += 1
        done_indices = done.nonzero(as_tuple=False).view(-1)
        if done_indices.numel() > 0:
            self._env_episode_ids[done_indices] += 1
            self._env_step_ids[done_indices] = 0

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, torch.Tensor]:
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[indices].to(device),
            "actions": self.actions[indices].to(device),
            "rewards": self.rewards[indices].to(device),
            "next_obs": self.next_obs[indices].to(device),
            "continues": self.continues[indices].to(device),
        }

    def _valid_sequence_starts(self, horizon: int) -> torch.Tensor:
        stride = max(int(self._last_batch_size), 1)
        horizon = int(horizon)
        cached = self._valid_start_cache.get(horizon)
        if cached is not None:
            return cached

        if horizon <= 0 or self.size < horizon * stride + 1 or self.episode_ids[: self.size].min().item() < 0:
            starts = torch.empty(0, dtype=torch.long, device=self.device)
            self._valid_start_cache[horizon] = starts
            return starts

        valid_limit = self.capacity if self.size == self.capacity else self.size
        starts = torch.arange(valid_limit, dtype=torch.long, device=self.device)
        offsets = torch.arange(horizon + 1, dtype=torch.long, device=self.device) * stride
        indices = (starts.unsqueeze(1) + offsets.unsqueeze(0)) % self.capacity

        valid = torch.ones(starts.shape[0], dtype=torch.bool, device=self.device)
        if self.size < self.capacity:
            valid &= (indices < self.size).all(dim=1)

        first = indices[:, 0]
        valid &= (self.env_ids[indices] == self.env_ids[first].unsqueeze(1)).all(dim=1)
        valid &= (self.episode_ids[indices] == self.episode_ids[first].unsqueeze(1)).all(dim=1)
        expected_steps = self.step_ids[first].unsqueeze(1) + torch.arange(horizon + 1, device=self.device)
        valid &= (self.step_ids[indices] == expected_steps).all(dim=1)
        if horizon > 1:
            valid &= (self.continues[indices[:, :-2]].squeeze(-1) > 0.0).all(dim=1)

        starts = starts[valid]
        self._valid_start_cache[horizon] = starts
        return starts

    def valid_sequence_count(self, horizon: int) -> int:
        return int(self._valid_sequence_starts(horizon).numel())

    def can_sample_sequences(self, batch_size: int, horizon: int) -> bool:
        return self.size >= batch_size and self.valid_sequence_count(horizon) > 0

    def sample_sequences(
        self,
        batch_size: int,
        horizon: int,
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        """Sample strict same-env, same-episode contiguous transition sequences.

        The buffer is filled by vectorized env steps, so transition `i + num_envs`
        is the next transition for the same environment as transition `i`.
        """

        stride = max(int(self._last_batch_size), 1)
        starts = self._valid_sequence_starts(horizon)
        if self.size < batch_size or starts.numel() == 0:
            raise ValueError(
                f"Cannot sample horizon={horizon} sequences from buffer size={self.size} "
                f"with vectorized stride={stride}. valid_sequences={starts.numel()}."
            )

        start_t = starts[torch.randint(0, starts.numel(), (batch_size,), device=self.device)]
        offsets = torch.arange(horizon + 1, device=self.device).unsqueeze(1) * stride
        indices = (start_t.unsqueeze(0) + offsets) % self.capacity
        transition_indices = indices[:-1]
        return {
            "obs": self.obs[indices].to(device),
            "actions": self.actions[transition_indices].to(device),
            "rewards": self.rewards[transition_indices].to(device),
            "continues": self.continues[transition_indices].to(device),
        }
