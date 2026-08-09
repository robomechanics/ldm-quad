# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


# Cache of resolved foot-body indices, keyed by id(articulation), so the regex
# body lookup only runs once per environment instead of every reward step.
_FOOT_BODY_CACHE: dict[int, list[int]] = {}


def feet_swing_gait(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    foot_pattern: str = ".*_foot",
    target_height: float = 0.08,
    command_threshold: float = 0.1,
    asset_name: str = "robot",
) -> torch.Tensor:
    """Contact-sensor-free gait shaping reward.

    Rewards the *spread* of foot clearances (std across feet) while a motion command
    is active. A spread is only large when some feet swing while others are planted,
    so this rewards an alternating gait and gives ~zero for standing (all feet low),
    sliding (all feet low), or hopping (all feet high). It is gated by command speed,
    so it is inert during the zero-command standing stage.

    Uses only ``body_pos_w`` (kinematics), so it works on both the PhysX and Newton
    backends of Isaac Lab -- no contact sensor required. If the foot bodies cannot be
    resolved on a given backend it returns zeros, so it can never crash env startup.

    Note: foot height is world-frame z, which equals height-above-ground only on flat
    terrain (this task's ``terrain_type = "plane"``). Revisit for rough terrain.
    """
    asset: Articulation = env.scene[asset_name]
    cache_key = id(asset)
    foot_ids = _FOOT_BODY_CACHE.get(cache_key)
    if foot_ids is None:
        try:
            foot_ids = list(asset.find_bodies(foot_pattern)[0])
        except Exception:  # noqa: BLE001 - degrade gracefully on any backend
            foot_ids = []
        _FOOT_BODY_CACHE[cache_key] = foot_ids
    if not foot_ids:
        return torch.zeros(env.num_envs, device=env.device)

    foot_height = asset.data.body_pos_w[:, foot_ids, 2]
    clearance = (foot_height.clamp(min=0.0) / max(float(target_height), 1e-6)).clamp(max=1.0)
    spread = clearance.std(dim=1, unbiased=False)

    command = env.command_manager.get_command(command_name)
    moving = (torch.linalg.norm(command[:, :2], dim=1) > float(command_threshold)).float()
    return spread * moving
