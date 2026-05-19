# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2RandFlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # disable startup randomization terms that are incompatible with the installed Isaac Lab event API
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        # disable contact-sensor-dependent terms for current Isaac Lab/Newton compatibility
        self.scene.contact_forces = None
        self.rewards.feet_air_time = None
        self.rewards.undesired_contacts = None
        self.terminations.base_contact = None
        self.terminations.base_height = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": 0.18},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=base_mdp.bad_orientation,
            params={"limit_angle": math.radians(60.0)},
        )
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None




class UnitreeGo2RandFlatEnvCfg_PLAY(UnitreeGo2RandFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
