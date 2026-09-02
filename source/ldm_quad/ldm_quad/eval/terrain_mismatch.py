"""Terrain mismatches shared by the MBRL and PPO evaluation scripts.

Both controllers must see the *same* perturbation for a baseline comparison to
mean anything, so the definitions live here rather than being duplicated per
script. Rigid<->deformable transitions are not covered: they run under Newton
MPM, which owns its own simulator.
"""
from __future__ import annotations

from copy import deepcopy

# Terrain axes of the planned evaluation. Each takes a single severity parameter.
TERRAIN_MISMATCHES = ["nominal", "low_friction", "compliant", "rough", "slope"]


def _terrain_material(env_cfg: object):
    return getattr(getattr(env_cfg, "scene", None), "terrain", None).physics_material


def _publish_material(env_cfg: object, material) -> None:
    if getattr(env_cfg, "sim", None) is not None:
        env_cfg.sim.physics_material = material


def set_friction(env_cfg: object, friction: float) -> None:
    material = _terrain_material(env_cfg)
    material.static_friction = friction
    material.dynamic_friction = friction
    _publish_material(env_cfg, material)
    events = getattr(env_cfg, "events", None)
    if getattr(events, "physics_material", None) is not None:
        events.physics_material.params["static_friction_range"] = (friction, friction)
        events.physics_material.params["dynamic_friction_range"] = (friction, friction)


def set_compliance(env_cfg: object, stiffness: float, damping: float) -> None:
    material = _terrain_material(env_cfg)
    material.compliant_contact_stiffness = stiffness
    material.compliant_contact_damping = damping
    _publish_material(env_cfg, material)


def use_generated_terrain(env_cfg: object):
    """Switch the scene onto a fresh, uncurriculumed 5x5 terrain generator."""
    from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = deepcopy(ROUGH_TERRAINS_CFG)
    env_cfg.scene.terrain.max_init_terrain_level = None
    if getattr(env_cfg, "curriculum", None) is not None:
        env_cfg.curriculum.terrain_levels = None
    generator = env_cfg.scene.terrain.terrain_generator
    if generator is not None:
        generator.num_rows = 5
        generator.num_cols = 5
        generator.curriculum = False
    return generator


def apply_terrain_mismatch(
    env_cfg: object,
    mismatch: str,
    *,
    friction: float = 0.35,
    compliant_stiffness: float = 5000.0,
    compliant_damping: float = 100.0,
    rough_noise: float = 0.025,
    slope: float = 0.3,
) -> None:
    """Apply one terrain mismatch to an env config, in place."""
    if mismatch == "nominal":
        return

    if mismatch == "low_friction":
        set_friction(env_cfg, friction)
        return

    if mismatch == "compliant":
        set_compliance(env_cfg, compliant_stiffness, compliant_damping)
        return

    if mismatch == "rough":
        # Roughness only: isolating random_rough keeps this a single-parameter axis,
        # distinct from the discrete contact geometry of "slope".
        generator = use_generated_terrain(env_cfg)
        if generator is not None:
            sub = generator.sub_terrains.get("random_rough")
            if sub is None:
                raise ValueError("ROUGH_TERRAINS_CFG has no random_rough sub-terrain.")
            sub.proportion = 1.0
            sub.noise_range = (0.005, rough_noise)
            sub.noise_step = 0.01
            generator.sub_terrains = {"random_rough": sub}
        return

    if mismatch == "slope":
        generator = use_generated_terrain(env_cfg)
        if generator is not None:
            slopes = {k: v for k, v in generator.sub_terrains.items() if "slope" in k}
            if not slopes:
                raise ValueError("ROUGH_TERRAINS_CFG has no sloped sub-terrain.")
            for sub in slopes.values():
                sub.proportion = 1.0 / len(slopes)
                sub.slope_range = (0.0, slope)
            generator.sub_terrains = slopes
        return

    raise ValueError(f"Unsupported terrain mismatch: {mismatch}")
