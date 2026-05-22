from __future__ import annotations

from collections.abc import Callable

import torch

from .models import DynamicsEnsemble


class TrajectoryPlanner:
    """Shared rollout evaluation utilities for action-sequence planners."""

    def __init__(
        self,
        model: DynamicsEnsemble,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        horizon: int = 30,
        candidates: int = 512,
        discount: float = 0.99,
        temperature: float = 0.5,
        action_spline_knots: int = 0,
        action_prior: Callable[[torch.Tensor], torch.Tensor] | None = None,
        prior_residual_scale: float = 0.3,
        prior_residual_penalty: float = 0.0,
        prior_acceptance_margin: float = 0.0,
        prior_fallback: bool = True,
        action_bounds_finite: bool = True,
        planner_velocity_objective_weight: float = 0.0,
        planner_velocity_target_x: float = 0.0,
        planner_velocity_target_y: float = 0.0,
        planner_velocity_target_yaw: float = 0.0,
    ):
        self.model = model
        self.action_low = action_low
        self.action_high = action_high
        self.horizon = horizon
        self.candidates = candidates
        self.discount = discount
        self.temperature = temperature
        self.action_dim = action_low.numel()
        self.action_spline_knots = action_spline_knots if 1 < action_spline_knots < horizon else 0
        self.action_prior = action_prior
        self.prior_residual_scale = prior_residual_scale
        self.prior_residual_penalty = prior_residual_penalty
        self.prior_acceptance_margin = prior_acceptance_margin
        self.prior_fallback = prior_fallback
        self.action_bounds_finite = action_bounds_finite
        self.planner_velocity_objective_weight = planner_velocity_objective_weight
        self.planner_velocity_target_x = planner_velocity_target_x
        self.planner_velocity_target_y = planner_velocity_target_y
        self.planner_velocity_target_yaw = planner_velocity_target_yaw
        self._prev_mean: torch.Tensor | None = None
        self.last_diagnostics: dict[str, float] = {}

    def reset(self, done: torch.Tensor | None = None) -> None:
        if self._prev_mean is None:
            return
        if done is None:
            self._prev_mean = None
            return
        done = done.to(device=self._prev_mean.device, dtype=torch.bool).view(-1)
        if done.numel() != self._prev_mean.shape[0]:
            self._prev_mean = None
            return
        self._prev_mean[done] = 0.0

    @property
    def control_horizon(self) -> int:
        return self.action_spline_knots or self.horizon

    def _warm_start_mean(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        shape = (batch_size, self.control_horizon, self.action_dim)
        if self._prev_mean is None or self._prev_mean.shape != shape or self._prev_mean.device != obs.device:
            return torch.zeros(shape, device=obs.device, dtype=obs.dtype)

        if not self.action_spline_knots:
            shifted = torch.zeros_like(self._prev_mean)
            shifted[:, :-1, :] = self._prev_mean[:, 1:, :]
            return shifted

        shifted_actions = torch.zeros((batch_size, self.horizon, self.action_dim), device=obs.device, dtype=obs.dtype)
        previous_actions = self._interpolate_action_spline(self._prev_mean)
        shifted_actions[:, :-1, :] = previous_actions[:, 1:, :]
        return self._sample_actions_at_knots(shifted_actions)

    def _sample_actions_at_knots(self, action_sequences: torch.Tensor) -> torch.Tensor:
        knot_positions = torch.linspace(0, self.horizon - 1, self.control_horizon, device=action_sequences.device)
        knot_indices = knot_positions.round().long().clamp_(0, self.horizon - 1)
        return action_sequences[:, knot_indices, :]

    def _interpolate_action_spline(self, action_knots: torch.Tensor) -> torch.Tensor:
        """Expand uniformly spaced action knots with cubic Catmull-Rom interpolation."""
        knot_count = action_knots.shape[-2]
        if knot_count == self.horizon:
            return action_knots

        positions = torch.linspace(0, knot_count - 1, self.horizon, device=action_knots.device)
        left = positions.floor().long().clamp_(0, knot_count - 1)
        right = (left + 1).clamp_(0, knot_count - 1)
        before = (left - 1).clamp_(0, knot_count - 1)
        after = (right + 1).clamp_(0, knot_count - 1)
        frac = (positions - left.to(positions.dtype)).view(*([1] * (action_knots.ndim - 2)), self.horizon, 1)

        p0 = action_knots[..., before, :]
        p1 = action_knots[..., left, :]
        p2 = action_knots[..., right, :]
        p3 = action_knots[..., after, :]
        frac2 = frac.square()
        frac3 = frac2 * frac
        return 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * frac
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * frac2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * frac3
        )

    def _expand_controls(self, controls: torch.Tensor) -> torch.Tensor:
        if not self.action_spline_knots:
            return controls
        return self._interpolate_action_spline(controls)

    def _zero_controls(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((obs.shape[0], self.control_horizon, self.action_dim), device=obs.device, dtype=obs.dtype)

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.action_bounds_finite:
            return actions
        return torch.max(torch.min(actions, self.action_high.view(1, -1)), self.action_low.view(1, -1))

    def _clip_action_sequences(self, action_sequences: torch.Tensor) -> torch.Tensor:
        return torch.max(
            torch.min(action_sequences, self.action_high.view(1, 1, 1, -1)),
            self.action_low.view(1, 1, 1, -1),
        )

    def _clip_controls(self, controls: torch.Tensor) -> torch.Tensor:
        if self.action_prior is None:
            return self._clip_action_sequences(controls)

        residual_scale = max(float(self.prior_residual_scale), 1e-6)
        return controls.clamp(-residual_scale, residual_scale)

    def _actions_from_controls(self, states: torch.Tensor, controls_t: torch.Tensor) -> torch.Tensor:
        if self.action_prior is None:
            return controls_t

        prior_actions = self.action_prior(states)
        return self._clip_actions(prior_actions + controls_t)

    def _first_action_from_controls(self, obs: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        controls = self._expand_controls(mean)
        controls_t = controls[:, 0, :]
        if self.action_prior is None:
            return self._clip_actions(controls_t)
        return self._actions_from_controls(obs, controls_t)

    def _maybe_fallback_to_prior(self, obs: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        self._record_solution_diagnostics(obs, mean)
        if self.action_prior is None or not self.prior_fallback:
            return mean

        candidate_return = self.evaluate_sequences(obs, self._expand_controls(mean).unsqueeze(1)).squeeze(1)
        zero_mean = self._zero_controls(obs)
        prior_return = self.evaluate_sequences(obs, self._expand_controls(zero_mean).unsqueeze(1)).squeeze(1)
        accept = candidate_return >= prior_return + self.prior_acceptance_margin
        selected_mean = torch.where(accept.view(-1, 1, 1), mean, zero_mean)
        self._record_solution_diagnostics(
            obs,
            selected_mean,
            candidate_return=candidate_return,
            prior_return=prior_return,
            accept=accept,
        )
        return selected_mean

    def _record_candidate_diagnostics(self, returns: torch.Tensor) -> None:
        returns = returns.detach()
        self.last_diagnostics.update(
            {
                "planner_candidate_return_mean": float(returns.mean().item()),
                "planner_candidate_return_best": float(returns.max(dim=1).values.mean().item()),
                "planner_candidate_return_std": float(returns.std(unbiased=False).item()),
            }
        )

    def _record_solution_diagnostics(
        self,
        obs: torch.Tensor,
        mean: torch.Tensor,
        candidate_return: torch.Tensor | None = None,
        prior_return: torch.Tensor | None = None,
        accept: torch.Tensor | None = None,
    ) -> None:
        controls = self._expand_controls(mean)
        first_residual = controls[:, 0, :].detach()
        residual_norm = first_residual.norm(dim=-1)
        diagnostics = {
            "planner_residual_norm_mean": float(residual_norm.mean().item()),
            "planner_residual_abs_mean": float(first_residual.abs().mean().item()),
            "planner_selected_prior_fraction": float((residual_norm <= 1e-6).float().mean().item()),
        }

        if self.action_prior is None:
            first_action = self._clip_actions(first_residual)
            diagnostics.update(
                {
                    "planner_prior_action_norm_mean": 0.0,
                    "planner_final_action_norm_mean": float(first_action.detach().norm(dim=-1).mean().item()),
                    "planner_residual_to_prior_norm": 0.0,
                }
            )
        else:
            prior_actions = self.action_prior(obs).detach()
            final_actions = self._clip_actions(prior_actions + first_residual).detach()
            prior_norm = prior_actions.norm(dim=-1)
            diagnostics.update(
                {
                    "planner_prior_action_norm_mean": float(prior_norm.mean().item()),
                    "planner_final_action_norm_mean": float(final_actions.norm(dim=-1).mean().item()),
                    "planner_residual_to_prior_norm": float((residual_norm / prior_norm.clamp_min(1e-6)).mean().item()),
                }
            )

        if candidate_return is not None and prior_return is not None and accept is not None:
            margin = (candidate_return - prior_return).detach()
            diagnostics.update(
                {
                    "planner_predicted_plan_return_mean": float(candidate_return.detach().mean().item()),
                    "planner_predicted_prior_return_mean": float(prior_return.detach().mean().item()),
                    "planner_predicted_return_margin_mean": float(margin.mean().item()),
                    "planner_predicted_return_margin_min": float(margin.min().item()),
                    "planner_prior_fallback_fraction": float((~accept).float().mean().item()),
                }
            )
        else:
            diagnostics.update(
                {
                    "planner_predicted_plan_return_mean": 0.0,
                    "planner_predicted_prior_return_mean": 0.0,
                    "planner_predicted_return_margin_mean": 0.0,
                    "planner_predicted_return_margin_min": 0.0,
                    "planner_prior_fallback_fraction": 0.0,
                }
            )

        self.last_diagnostics.update(diagnostics)

    def _planner_velocity_objective_reward(self, states: torch.Tensor) -> torch.Tensor:
        if self.planner_velocity_objective_weight <= 0.0 or states.shape[-1] < 6:
            return torch.zeros(states.shape[0], device=states.device, dtype=states.dtype)

        target_x = torch.as_tensor(self.planner_velocity_target_x, device=states.device, dtype=states.dtype)
        target_y = torch.as_tensor(self.planner_velocity_target_y, device=states.device, dtype=states.dtype)
        target_yaw = torch.as_tensor(self.planner_velocity_target_yaw, device=states.device, dtype=states.dtype)
        error = (states[:, 0] - target_x).square()
        error = error + (states[:, 1] - target_y).square()
        error = error + (states[:, 5] - target_yaw).square()
        return -float(self.planner_velocity_objective_weight) * error

    @torch.no_grad()
    def evaluate_sequences(self, obs: torch.Tensor, control_sequences: torch.Tensor) -> torch.Tensor:
        batch_size, candidates, _, action_dim = control_sequences.shape
        states = obs.unsqueeze(1).expand(-1, candidates, -1).reshape(batch_size * candidates, -1)
        returns = torch.zeros(batch_size * candidates, device=obs.device, dtype=obs.dtype)
        discounts = torch.ones_like(returns)
        alive = torch.ones_like(returns)

        for t in range(self.horizon):
            controls_t = control_sequences[:, :, t, :].reshape(batch_size * candidates, action_dim)
            actions_t = self._actions_from_controls(states, controls_t)
            preds = self.model.predict(states, actions_t)
            reward = preds.rewards.squeeze(-1)
            reward = reward + self._planner_velocity_objective_reward(states)
            if self.action_prior is not None and self.prior_residual_penalty > 0.0:
                reward = reward - self.prior_residual_penalty * controls_t.square().sum(dim=-1)
            continue_prob = preds.continue_logits.sigmoid().squeeze(-1)
            returns = returns + discounts * alive * reward
            alive = alive * continue_prob
            discounts = discounts * self.discount
            states = states + preds.delta_obs

        return returns.view(batch_size, candidates)


class CEMPlanner(TrajectoryPlanner):
    """Cross-entropy planner over learned dynamics."""

    def __init__(
        self,
        model: DynamicsEnsemble,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        horizon: int = 30,
        candidates: int = 512,
        elites: int = 64,
        iterations: int = 5,
        discount: float = 0.99,
        temperature: float = 0.5,
        action_spline_knots: int = 0,
        action_prior: Callable[[torch.Tensor], torch.Tensor] | None = None,
        prior_residual_scale: float = 0.3,
        prior_residual_penalty: float = 0.0,
        prior_acceptance_margin: float = 0.0,
        prior_fallback: bool = True,
        action_bounds_finite: bool = True,
        planner_velocity_objective_weight: float = 0.0,
        planner_velocity_target_x: float = 0.0,
        planner_velocity_target_y: float = 0.0,
        planner_velocity_target_yaw: float = 0.0,
    ):
        super().__init__(
            model=model,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
            candidates=candidates,
            discount=discount,
            temperature=temperature,
            action_spline_knots=action_spline_knots,
            action_prior=action_prior,
            prior_residual_scale=prior_residual_scale,
            prior_residual_penalty=prior_residual_penalty,
            prior_acceptance_margin=prior_acceptance_margin,
            prior_fallback=prior_fallback,
            action_bounds_finite=action_bounds_finite,
            planner_velocity_objective_weight=planner_velocity_objective_weight,
            planner_velocity_target_x=planner_velocity_target_x,
            planner_velocity_target_y=planner_velocity_target_y,
            planner_velocity_target_yaw=planner_velocity_target_yaw,
        )
        self.elites = elites
        self.iterations = iterations

    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self._warm_start_mean(obs)
        std = torch.full_like(mean, self.temperature)

        for _ in range(self.iterations):
            noise = torch.randn(
                (obs.shape[0], self.candidates, self.control_horizon, self.action_dim),
                device=obs.device,
                dtype=obs.dtype,
            )
            controls = mean.unsqueeze(1) + std.unsqueeze(1) * noise
            controls = self._clip_controls(controls)
            if self.action_prior is not None and controls.shape[1] > 0:
                controls[:, 0, :, :] = 0.0
            action_sequences = self._expand_controls(controls)

            returns = self.evaluate_sequences(obs, action_sequences)
            self._record_candidate_diagnostics(returns)
            elite_indices = returns.topk(self.elites, dim=1).indices
            elite_controls = controls.gather(
                1, elite_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.control_horizon, self.action_dim)
            )
            mean = elite_controls.mean(dim=1)
            std = elite_controls.std(dim=1).clamp_min(1e-3)

        mean = self._maybe_fallback_to_prior(obs, mean)
        self._prev_mean = mean.detach()
        return self._first_action_from_controls(obs, mean)


class MPPIPlanner(TrajectoryPlanner):
    """Model predictive path integral planner over learned dynamics."""

    def __init__(
        self,
        model: DynamicsEnsemble,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        horizon: int = 30,
        candidates: int = 512,
        iterations: int = 5,
        discount: float = 0.99,
        temperature: float = 0.5,
        lambda_: float = 1.0,
        action_spline_knots: int = 0,
        action_prior: Callable[[torch.Tensor], torch.Tensor] | None = None,
        prior_residual_scale: float = 0.3,
        prior_residual_penalty: float = 0.0,
        prior_acceptance_margin: float = 0.0,
        prior_fallback: bool = True,
        action_bounds_finite: bool = True,
        planner_velocity_objective_weight: float = 0.0,
        planner_velocity_target_x: float = 0.0,
        planner_velocity_target_y: float = 0.0,
        planner_velocity_target_yaw: float = 0.0,
    ):
        super().__init__(
            model=model,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
            candidates=candidates,
            discount=discount,
            temperature=temperature,
            action_spline_knots=action_spline_knots,
            action_prior=action_prior,
            prior_residual_scale=prior_residual_scale,
            prior_residual_penalty=prior_residual_penalty,
            prior_acceptance_margin=prior_acceptance_margin,
            prior_fallback=prior_fallback,
            action_bounds_finite=action_bounds_finite,
            planner_velocity_objective_weight=planner_velocity_objective_weight,
            planner_velocity_target_x=planner_velocity_target_x,
            planner_velocity_target_y=planner_velocity_target_y,
            planner_velocity_target_yaw=planner_velocity_target_yaw,
        )
        self.iterations = iterations
        self.lambda_ = lambda_

    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self._warm_start_mean(obs)
        std = torch.full_like(mean, self.temperature)

        for _ in range(self.iterations):
            noise = torch.randn(
                (obs.shape[0], self.candidates, self.control_horizon, self.action_dim),
                device=obs.device,
                dtype=obs.dtype,
            )
            controls = mean.unsqueeze(1) + std.unsqueeze(1) * noise
            controls = self._clip_controls(controls)
            if self.action_prior is not None and controls.shape[1] > 0:
                controls[:, 0, :, :] = 0.0
            action_sequences = self._expand_controls(controls)
            returns = self.evaluate_sequences(obs, action_sequences)
            self._record_candidate_diagnostics(returns)

            shifted_returns = returns - returns.max(dim=1, keepdim=True).values
            weights = torch.softmax(shifted_returns / max(self.lambda_, 1e-6), dim=1)
            mean = (weights.unsqueeze(-1).unsqueeze(-1) * controls).sum(dim=1)
            centered = controls - mean.unsqueeze(1)
            variance = (weights.unsqueeze(-1).unsqueeze(-1) * centered.square()).sum(dim=1)
            std = variance.sqrt().clamp_min(1e-3)

        mean = self._maybe_fallback_to_prior(obs, mean)
        self._prev_mean = mean.detach()
        return self._first_action_from_controls(obs, mean)


def build_planner(
    planner_name: str,
    model: DynamicsEnsemble,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    horizon: int,
    candidates: int,
    elites: int,
    iterations: int,
    discount: float,
    temperature: float,
    lambda_: float,
    action_spline_knots: int = 0,
    action_prior: Callable[[torch.Tensor], torch.Tensor] | None = None,
    prior_residual_scale: float = 0.3,
    prior_residual_penalty: float = 0.0,
    prior_acceptance_margin: float = 0.0,
    prior_fallback: bool = True,
    action_bounds_finite: bool = True,
        planner_velocity_objective_weight: float = 0.0,
        planner_velocity_target_x: float = 0.0,
        planner_velocity_target_y: float = 0.0,
        planner_velocity_target_yaw: float = 0.0,
) -> TrajectoryPlanner:
    if planner_name == "mppi":
        return MPPIPlanner(
            model=model,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
            candidates=candidates,
            iterations=iterations,
            discount=discount,
            temperature=temperature,
            lambda_=lambda_,
            action_spline_knots=action_spline_knots,
            action_prior=action_prior,
            prior_residual_scale=prior_residual_scale,
            prior_residual_penalty=prior_residual_penalty,
            prior_acceptance_margin=prior_acceptance_margin,
            prior_fallback=prior_fallback,
            action_bounds_finite=action_bounds_finite,
            planner_velocity_objective_weight=planner_velocity_objective_weight,
            planner_velocity_target_x=planner_velocity_target_x,
            planner_velocity_target_y=planner_velocity_target_y,
            planner_velocity_target_yaw=planner_velocity_target_yaw,
        )
    if planner_name == "cem":
        return CEMPlanner(
            model=model,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
            candidates=candidates,
            elites=elites,
            iterations=iterations,
            discount=discount,
            temperature=temperature,
            action_spline_knots=action_spline_knots,
            action_prior=action_prior,
            prior_residual_scale=prior_residual_scale,
            prior_residual_penalty=prior_residual_penalty,
            prior_acceptance_margin=prior_acceptance_margin,
            prior_fallback=prior_fallback,
            action_bounds_finite=action_bounds_finite,
            planner_velocity_objective_weight=planner_velocity_objective_weight,
            planner_velocity_target_x=planner_velocity_target_x,
            planner_velocity_target_y=planner_velocity_target_y,
            planner_velocity_target_yaw=planner_velocity_target_yaw,
        )
    raise ValueError(f"Unsupported planner: {planner_name}")
