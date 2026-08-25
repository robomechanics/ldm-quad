# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import mdp
from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2RandFlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Action scale: baked to 0.40 (action-scale sweep winner) so the FROZEN benchmark
        # task drives the faster gait that clears the reward-only ~0.33 ceiling to ~0.39 m/s
        # at command 0.4. Parent rough_env_cfg sets 0.25; the older x0.2/x0.3/x0.4-v0p31
        # curriculum checkpoints were trained at 0.25 and must be replayed with
        # --action_scale 0.25 (run.sh play does this per stage).
        self.actions.joint_pos.scale = 0.40

        # override rewards
        self.rewards.alive.weight = 0.10  # tuned (x=0.4 sweep best): lower alive so speed-tracking dominates
        # Soften the fall cliff and the anti-motion penalties so a faster gait (needed
        # for higher command_x, e.g. 0.4 m/s) is not catastrophically punished. The
        # old -8.0 terminating + -2.5 tilt + -2.0 vertical penalties made every
        # speed-induced stumble a large negative, collapsing the policy at x=0.4.
        self.rewards.terminating.weight = -2.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        # Sharpen + strengthen forward-speed tracking so the policy actually pursues
        # the commanded velocity instead of standing still. The original std=0.5 was so
        # forgiving it paid ~79% reward at near-zero speed (vs a 0.3 command); a lower
        # std tightens the tracking gradient and a higher weight makes speed matter
        # more relative to the +0.5/step alive reward.
        self.rewards.track_lin_vel_xy_exp.weight = 8.0   # tuned (x=0.4 sweep best): peak vel ~0.31 m/s
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.11  # tuned: sharper tracking
        self.rewards.feet_air_time.weight = 0.25

        # proprioceptive gait shaping: contact-sensor-free so it runs on both the
        # PhysX and Newton backends. Command-gated, so it is inert while standing.
        self.rewards.feet_swing_gait = RewTerm(
            func=mdp.feet_swing_gait,
            weight=0.5,
            params={
                "command_name": "base_velocity",
                "foot_pattern": ".*_foot",
                "target_height": 0.08,
                "command_threshold": 0.1,
            },
        )

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
        # Loosen the early-fall thresholds: the 0.40 m spawn left only 0.15 m of
        # height margin and 35 deg of tilt, which a dynamic 0.4 m/s gait crosses on
        # normal stride dips/pitch, ending episodes at ~90/1000 steps. Give the
        # faster gait room before it counts as a fall.
        self.terminations.base_height = DoneTerm(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": 0.20},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=base_mdp.bad_orientation,
            params={"limit_angle": math.radians(45.0)},
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
