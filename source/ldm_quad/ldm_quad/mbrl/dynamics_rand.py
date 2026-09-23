"""Per-env dynamics randomisation with the true parameters exposed for logging and probes.

Torch-only on purpose (Isaac objects are used duck-typed through the env handed to
``attach``), so tests can load this file by path without starting Isaac Sim.

Axes (column order of :meth:`DynamicsRandomizer.params`, see ``PARAM_NAMES``):

* ``motor_gain``: per-env multiplier on the joint-position action scale, applied INSIDE the
  action term (``processed = raw * scale * g + offset``). The raw command the policy produced
  is what the observation's last-action term, the replay buffer and the history encoder see,
  so the gain is a genuine actuator mismatch rather than something readable from the obs.
* ``foot_friction``: per-env friction written to every collision shape of the robot through
  the PhysX view (static = dynamic = f). The terrain uses friction_combine_mode="multiply",
  so the effective foot-ground coefficient is terrain_mu * f.

Both axes resample per env at every episode reset from a uniform range. A disabled axis
leaves the simulator untouched and reports NaN in ``params``.
"""

from __future__ import annotations

import torch

PARAM_NAMES = ("motor_gain", "foot_friction")


class DynamicsRandomizer:
    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        motor_gain_range: tuple[float, float] | None = None,
        friction_range: tuple[float, float] | None = None,
        seed: int | None = None,
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.motor_gain_range = self._check_range("motor_gain", motor_gain_range)
        self.friction_range = self._check_range("foot_friction", friction_range)
        self._gen = torch.Generator(device="cpu")
        if seed is not None:
            self._gen.manual_seed(int(seed))
        nan = torch.full((self.num_envs,), float("nan"), device=self.device)
        self.motor_gain = nan.clone()
        self.foot_friction = nan.clone()
        self._action_term = None
        self._base_scale = None
        self._robot_view = None
        self._materials: torch.Tensor | None = None
        self.default_materials: torch.Tensor | None = None

    @staticmethod
    def _check_range(name: str, rng) -> tuple[float, float] | None:
        if rng is None:
            return None
        lo, hi = float(rng[0]), float(rng[1])
        if not (0.0 < lo <= hi):
            raise ValueError(f"{name} range must satisfy 0 < lo <= hi, got {rng}")
        return lo, hi

    @property
    def enabled(self) -> bool:
        return self.motor_gain_range is not None or self.friction_range is not None

    def attach(self, env, action_term: str = "joint_pos", robot: str = "robot") -> None:
        """Bind to a live Isaac Lab env and draw every env's parameters once."""
        if not self.enabled:
            return
        unwrapped = getattr(env, "unwrapped", env)
        if self.motor_gain_range is not None:
            term = unwrapped.action_manager.get_term(action_term)
            base = term._scale
            action_dim = term.action_dim
            if isinstance(base, torch.Tensor):
                base = base.detach().clone().to(self.device)
            else:
                base = torch.full((self.num_envs, action_dim), float(base), device=self.device)
            self._action_term = term
            self._base_scale = base
            term._scale = base.clone()
        if self.friction_range is not None:
            view = unwrapped.scene[robot].root_physx_view
            self._robot_view = view
            self._materials = view.get_material_properties().clone()  # CPU [N, shapes, 3]
            self.default_materials = self._materials.clone()
        self.resample(torch.arange(self.num_envs, device=self.device))

    def _draw(self, rng: tuple[float, float], n: int) -> torch.Tensor:
        lo, hi = rng
        return (lo + (hi - lo) * torch.rand(n, generator=self._gen)).to(self.device)

    def resample(self, env_ids: torch.Tensor) -> None:
        """Redraw the parameters of ``env_ids`` (index tensor or bool mask) and push them to sim."""
        if not self.enabled:
            return
        if env_ids.dtype == torch.bool:
            env_ids = env_ids.view(-1).nonzero(as_tuple=False).view(-1)
        env_ids = env_ids.to(self.device, torch.long)
        if env_ids.numel() == 0:
            return
        if self.motor_gain_range is not None:
            self._resample_gain(env_ids)
        if self.friction_range is not None:
            self._resample_friction(env_ids)

    def _resample_gain(self, env_ids: torch.Tensor) -> None:
        self.motor_gain[env_ids] = self._draw(self.motor_gain_range, env_ids.numel())
        if self._action_term is not None:
            self._action_term._scale[env_ids] = self._base_scale[env_ids] * self.motor_gain[env_ids].unsqueeze(-1)

    def _resample_friction(self, env_ids: torch.Tensor) -> None:
        self.foot_friction[env_ids] = self._draw(self.friction_range, env_ids.numel())
        if self._robot_view is not None:
            ids_cpu = env_ids.cpu()
            f = self.foot_friction[env_ids].cpu().unsqueeze(-1)
            self._materials[ids_cpu, :, 0] = f
            self._materials[ids_cpu, :, 1] = f
            self._robot_view.set_material_properties(self._materials, ids_cpu)

    def switch(self, motor_gain: float | None = None, friction: float | None = None) -> None:
        """Within-episode dynamics switch: set every env's gain and/or friction NOW (no reset;
        robot state and history untouched). The axis range is pinned to the new value, so envs
        that reset afterwards keep it. An axis must be enabled (attached) to be switched."""
        all_ids = torch.arange(self.num_envs, device=self.device)
        if motor_gain is not None:
            if self.motor_gain_range is None:
                raise ValueError("switching motor_gain needs --dyn_motor_gain_range (pin the pre-switch value)")
            self.motor_gain_range = self._check_range("motor_gain", (motor_gain, motor_gain))
            self._resample_gain(all_ids)
        if friction is not None:
            if self.friction_range is None:
                raise ValueError("switching friction needs --dyn_friction_range (pin the pre-switch value)")
            self.friction_range = self._check_range("foot_friction", (friction, friction))
            self._resample_friction(all_ids)

    def params(self) -> torch.Tensor:
        """True parameters active in each env right now, ``[num_envs, len(PARAM_NAMES)]``."""
        return torch.stack([self.motor_gain, self.foot_friction], dim=-1)

    def metrics(self) -> dict[str, float]:
        """Per-env mean/min/max of the ENABLED axes (column names stable for metrics.csv)."""
        out: dict[str, float] = {}
        for name, values, rng in (
            ("motor_gain", self.motor_gain, self.motor_gain_range),
            ("foot_friction", self.foot_friction, self.friction_range),
        ):
            if rng is None:
                continue
            out[f"dyn_{name}_mean"] = float(values.mean())
            out[f"dyn_{name}_min"] = float(values.min())
            out[f"dyn_{name}_max"] = float(values.max())
        return out

    def read_back_friction(self) -> torch.Tensor | None:
        """Per-env static friction as the simulator reports it (mean over shapes)."""
        if self._robot_view is None:
            return None
        return self._robot_view.get_material_properties()[:, :, 0].mean(dim=1)
