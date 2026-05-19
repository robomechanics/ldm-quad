# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .flat_env_cfg import UnitreeGo2RandFlatEnvCfg


@configclass
class UnitreeGo2SacFlatEnvCfg(UnitreeGo2RandFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Keep a SAC-specific config hook for reward or termination tuning.
        # self.terminations.base_contact = None


@configclass
class UnitreeGo2SacFlatEnvCfg_PLAY(UnitreeGo2SacFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
