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


def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track ONLY the forward/backward velocity command, with its own exponential kernel.

    Split out from IsaacLab's ``track_lin_vel_xy_exp``, which exponentiates the SUMMED squared
    xy error: ``exp(-(ex^2 + ey^2)/s^2)`` factorises into ``exp(-ex^2/s^2) * exp(-ey^2/s^2)``,
    so each axis MULTIPLICATIVELY gates the other's gradient. Measured at std=0.20: an x error
    of 0.30 leaves only 10.5% of the y gradient alive. Worse, it is super-additive -- from both
    axes at 0.30 error, fixing x pays +0.754 and then fixing y pays +7.157, i.e. the reward pays
    ~9x more for COMPLETING a specialisation than for starting one. That is a direct incentive
    to trade one axis away for the other, and it matches the observed Stage M behaviour
    (backward 38%->49% while lateral collapsed 98%->63%).

    Two additive per-axis terms at half weight each give the same total at zero error and pay
    the same for either axis regardless of the other, removing the trading incentive.
    NOTE: the additive form is SOFTER at moderate error (at ex=ey=0.10, joint=4.85 vs
    additive=6.23), so step reward rises on the switch for structural reasons, not skill.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    err = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    return torch.exp(-err / std**2)


def track_lin_vel_y_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track ONLY the lateral velocity command. See :func:`track_lin_vel_x_exp` for why split."""
    asset: Articulation = env.scene[asset_cfg.name]
    err = torch.square(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1])
    return torch.exp(-err / std**2)
