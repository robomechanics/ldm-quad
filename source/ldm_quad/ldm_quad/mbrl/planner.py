from __future__ import annotations

from collections.abc import Callable

import torch

from .models import DynamicsEnsemble, StateWorldModel
from .world_model import LatentWorldModel


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
        prior_candidate_fraction: float = 0.1,
        prior_candidate_noise: float = 0.02,
        prior_command_candidate_fraction: float = 0.0,
        prior_command_noise: float = 0.1,
        prior_command_start: int = -1,
        prior_command_dim: int = 3,
        prior_control_mode: str = "residual",
        action_bounds_finite: bool = True,
        planner_velocity_objective_weight: float = 0.0,
        planner_velocity_target_x: float = 0.0,
        planner_velocity_target_y: float = 0.0,
        planner_velocity_target_yaw: float = 0.0,
        use_best_candidate: bool = False,
        terminal_value: bool = False,
        disagreement_penalty: float = 0.0,
        model_policy_candidate_count: int = 0,
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
        self.prior_candidate_fraction = prior_candidate_fraction
        self.prior_candidate_noise = prior_candidate_noise
        self.prior_command_candidate_fraction = prior_command_candidate_fraction
        self.prior_command_noise = prior_command_noise
        self.prior_command_start = prior_command_start
        self.prior_command_dim = prior_command_dim
        if prior_control_mode not in {"residual", "full_action"}:
            raise ValueError(f"Unsupported prior_control_mode: {prior_control_mode}")
        self.prior_control_mode = prior_control_mode
        self.action_bounds_finite = action_bounds_finite
        self.planner_velocity_objective_weight = planner_velocity_objective_weight
        self.planner_velocity_target_x = planner_velocity_target_x
        self.planner_velocity_target_y = planner_velocity_target_y
        self.planner_velocity_target_yaw = planner_velocity_target_yaw
        self.use_best_candidate = use_best_candidate
        self.terminal_value = terminal_value
        self.disagreement_penalty = disagreement_penalty
        self.model_policy_candidate_count = max(0, int(model_policy_candidate_count))
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

    @property
    def _uses_full_action_prior(self) -> bool:
        return self.action_prior is not None and self.prior_control_mode == "full_action"

    def _warm_start_mean(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        shape = (batch_size, self.control_horizon, self.action_dim)
        if self._prev_mean is None or self._prev_mean.shape != shape or self._prev_mean.device != obs.device:
            if self._uses_full_action_prior:
                return self._prior_control_mean(obs)
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
        return action_sequences[..., knot_indices, :]

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
        if self.action_prior is None or self._uses_full_action_prior:
            return self._clip_action_sequences(controls)

        residual_scale = max(float(self.prior_residual_scale), 1e-6)
        return controls.clamp(-residual_scale, residual_scale)

    def _prior_candidate_count(self, num_candidates: int) -> int:
        if self.action_prior is None or num_candidates <= 0:
            return 0
        fraction = min(max(float(self.prior_candidate_fraction), 0.0), 1.0)
        return min(num_candidates, max(1, int(round(fraction * num_candidates))))

    def _model_policy_candidate_count(self, num_candidates: int) -> int:
        if num_candidates <= 0 or self.model_policy_candidate_count <= 0 or not hasattr(self.model, "pi"):
            return 0
        if self.action_prior is not None and not self._uses_full_action_prior:
            return 0
        prior_count = self._prior_candidate_count(num_candidates)
        return min(max(num_candidates - prior_count, 0), self.model_policy_candidate_count)

    def _prior_command_candidate_count(self, num_candidates: int) -> int:
        if not self._uses_full_action_prior or num_candidates <= 1:
            return 0
        prior_count = self._prior_candidate_count(num_candidates)
        if prior_count <= 1:
            return 0
        fraction = min(max(float(self.prior_command_candidate_fraction), 0.0), 1.0)
        return min(prior_count - 1, int(round(fraction * num_candidates)))

    def _command_slice(self, obs_dim: int) -> slice | None:
        command_dim = int(self.prior_command_dim)
        if command_dim <= 0:
            return None
        if self.prior_command_start >= 0:
            start = int(self.prior_command_start)
        elif obs_dim == 45:
            start = 6
        else:
            start = 9
        end = start + command_dim
        if start < 0 or end > obs_dim:
            return None
        return slice(start, end)

    def _perturb_prior_commands(self, obs: torch.Tensor, candidate_count: int) -> torch.Tensor:
        obs_candidates = obs.unsqueeze(1).expand(-1, candidate_count, -1).clone()
        command_slice = self._command_slice(obs.shape[-1])
        if command_slice is None or candidate_count <= 0:
            return obs_candidates

        noise_scale = max(float(self.prior_command_noise), 0.0)
        if noise_scale <= 0.0:
            return obs_candidates

        command_noise = noise_scale * torch.randn_like(obs_candidates[..., command_slice])
        obs_candidates[..., command_slice] = obs_candidates[..., command_slice] + command_noise
        return obs_candidates

    def _rollout_prior_actions(self, obs: torch.Tensor) -> torch.Tensor:
        if self.action_prior is None:
            return self._expand_controls(self._zero_controls(obs))

        leading_shape = obs.shape[:-1]
        states = obs.reshape(-1, obs.shape[-1])
        actions = []
        for _ in range(self.horizon):
            action = self._clip_actions(self.action_prior(states))
            actions.append(action)
            preds = self.model.predict(states, action)
            states = states + preds.delta_obs
        return torch.stack(actions, dim=1).view(*leading_shape, self.horizon, self.action_dim)

    def _prior_control_mean(self, obs: torch.Tensor) -> torch.Tensor:
        prior_actions = self._rollout_prior_actions(obs)
        if self.action_spline_knots:
            return self._sample_actions_at_knots(prior_actions)
        return prior_actions

    def _rollout_model_policy_actions(self, obs: torch.Tensor, candidate_count: int) -> torch.Tensor:
        leading_shape = (obs.shape[0], candidate_count)
        states = obs.unsqueeze(1).expand(-1, candidate_count, -1).reshape(-1, obs.shape[-1])
        actions = []
        for _ in range(self.horizon):
            action = self._clip_actions(self.model.pi(states, deterministic=False))
            actions.append(action.view(*leading_shape, self.action_dim))
            preds = self.model.predict(states, action)
            states = states + preds.delta_obs
        return torch.stack(actions, dim=2)

    def _prior_baseline_controls(self, obs: torch.Tensor) -> torch.Tensor:
        if self._uses_full_action_prior:
            return self._prior_control_mean(obs)
        return self._zero_controls(obs)

    def _policy_candidate_controls(self, obs: torch.Tensor) -> torch.Tensor | None:
        policy_count = self._model_policy_candidate_count(self.candidates)
        if policy_count == 0:
            return None
        policy_actions = self._rollout_model_policy_actions(obs, policy_count)
        return self._sample_actions_at_knots(policy_actions) if self.action_spline_knots else policy_actions

    def _inject_prior_candidates(
        self,
        obs: torch.Tensor,
        controls: torch.Tensor,
        policy_controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reserve initial candidates for pure or near-prior closed-loop rollouts."""
        count = self._prior_candidate_count(controls.shape[1])
        if count == 0:
            policy_start = 0
        else:
            policy_start = count

        if count > 0 and self._uses_full_action_prior:
            prior_controls = self._prior_control_mean(obs)
            controls[:, :count, :, :] = prior_controls.unsqueeze(1)
            command_count = self._prior_command_candidate_count(controls.shape[1])
            if command_count > 0:
                command_obs = self._perturb_prior_commands(obs, command_count)
                command_prior_actions = self._rollout_prior_actions(command_obs)
                if self.action_spline_knots:
                    command_prior_controls = self._sample_actions_at_knots(command_prior_actions)
                else:
                    command_prior_controls = command_prior_actions
                controls[:, 1 : 1 + command_count, :, :] = command_prior_controls
        elif count > 0:
            controls[:, :count, :, :] = 0.0

        noise_scale = max(float(self.prior_candidate_noise), 0.0)
        if count > 1 and noise_scale > 0.0:
            noise_start = 1 + self._prior_command_candidate_count(controls.shape[1]) if self._uses_full_action_prior else 1
            noise_start = min(max(noise_start, 1), count)
            prior_noise = noise_scale * torch.randn_like(controls[:, noise_start:count, :, :])
            if self._uses_full_action_prior:
                controls[:, noise_start:count, :, :] = self._clip_controls(
                    controls[:, noise_start:count, :, :] + prior_noise
                )
            else:
                controls[:, noise_start:count, :, :] = self._clip_controls(prior_noise)

        policy_count = 0 if policy_controls is None else min(policy_controls.shape[1], controls.shape[1] - policy_start)
        if policy_count > 0:
            controls[:, policy_start : policy_start + policy_count, :, :] = policy_controls[:, :policy_count]
        return controls

    def _actions_from_controls(self, states: torch.Tensor, controls_t: torch.Tensor) -> torch.Tensor:
        if self.action_prior is None or self._uses_full_action_prior:
            return self._clip_actions(controls_t)

        prior_actions = self.action_prior(states)
        return self._clip_actions(prior_actions + controls_t)

    def _first_action_from_controls(self, obs: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        controls = self._expand_controls(mean)
        controls_t = controls[:, 0, :]
        if self.action_prior is None or self._uses_full_action_prior:
            return self._clip_actions(controls_t)
        return self._actions_from_controls(obs, controls_t)

    def _maybe_fallback_to_prior(self, obs: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        self._record_solution_diagnostics(obs, mean)
        if self.action_prior is None or not self.prior_fallback:
            return mean

        candidate_return = self.evaluate_sequences(obs, self._expand_controls(mean).unsqueeze(1)).squeeze(1)
        prior_mean = self._prior_baseline_controls(obs)
        prior_return = self.evaluate_sequences(obs, self._expand_controls(prior_mean).unsqueeze(1)).squeeze(1)
        accept = candidate_return >= prior_return + self.prior_acceptance_margin
        selected_mean = torch.where(accept.view(-1, 1, 1), mean, prior_mean)
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
        first_control = controls[:, 0, :].detach()
        first_delta = first_control
        prior_actions = None
        if self.action_prior is not None:
            prior_actions = self.action_prior(obs).detach()
            if self._uses_full_action_prior:
                first_delta = first_control - prior_actions
        residual_norm = first_delta.norm(dim=-1)
        diagnostics = {
            "planner_residual_norm_mean": float(residual_norm.mean().item()),
            "planner_residual_abs_mean": float(first_delta.abs().mean().item()),
            "planner_selected_prior_fraction": float((residual_norm <= 1e-6).float().mean().item()),
            "planner_prior_candidate_fraction": float(self._prior_candidate_count(self.candidates) / max(self.candidates, 1)),
            "planner_model_policy_candidate_fraction": float(
                self._model_policy_candidate_count(self.candidates) / max(self.candidates, 1)
            ),
            "planner_prior_command_candidate_fraction": float(
                self._prior_command_candidate_count(self.candidates) / max(self.candidates, 1)
            ),
            "planner_full_action_mode": float(self._uses_full_action_prior),
            "planner_best_candidate_mode": float(self.use_best_candidate),
        }

        if self.action_prior is None:
            first_action = self._clip_actions(first_control)
            diagnostics.update(
                {
                    "planner_prior_action_norm_mean": 0.0,
                    "planner_final_action_norm_mean": float(first_action.detach().norm(dim=-1).mean().item()),
                    "planner_residual_to_prior_norm": 0.0,
                }
            )
        else:
            assert prior_actions is not None
            if self._uses_full_action_prior:
                final_actions = self._clip_actions(first_control).detach()
            else:
                final_actions = self._clip_actions(prior_actions + first_delta).detach()
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
            if self.disagreement_penalty > 0.0 and hasattr(self.model, "disagreement"):
                reward = reward - self.disagreement_penalty * self.model.disagreement(states, actions_t)
            if self.action_prior is not None and self.prior_residual_penalty > 0.0:
                if self._uses_full_action_prior:
                    prior_actions = self._clip_actions(self.action_prior(states))
                    residual = actions_t - prior_actions
                else:
                    residual = controls_t
                reward = reward - self.prior_residual_penalty * residual.square().sum(dim=-1)
            continue_prob = preds.continue_logits.sigmoid().squeeze(-1)
            returns = returns + discounts * alive * reward
            alive = alive * continue_prob
            discounts = discounts * self.discount
            states = states + preds.delta_obs

        if self.terminal_value and hasattr(self.model, "value"):
            returns = returns + discounts * alive * self.model.value(states).squeeze(-1)

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
        prior_candidate_fraction: float = 0.1,
        prior_candidate_noise: float = 0.02,
        prior_command_candidate_fraction: float = 0.0,
        prior_command_noise: float = 0.1,
        prior_command_start: int = -1,
        prior_command_dim: int = 3,
        prior_control_mode: str = "residual",
        action_bounds_finite: bool = True,
        planner_velocity_objective_weight: float = 0.0,
        planner_velocity_target_x: float = 0.0,
        planner_velocity_target_y: float = 0.0,
        planner_velocity_target_yaw: float = 0.0,
        use_best_candidate: bool = False,
        terminal_value: bool = False,
        disagreement_penalty: float = 0.0,
        model_policy_candidate_count: int = 0,
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
            prior_candidate_fraction=prior_candidate_fraction,
            prior_candidate_noise=prior_candidate_noise,
            prior_command_candidate_fraction=prior_command_candidate_fraction,
            prior_command_noise=prior_command_noise,
            prior_command_start=prior_command_start,
            prior_command_dim=prior_command_dim,
            prior_control_mode=prior_control_mode,
            action_bounds_finite=action_bounds_finite,
            planner_velocity_objective_weight=planner_velocity_objective_weight,
            planner_velocity_target_x=planner_velocity_target_x,
            planner_velocity_target_y=planner_velocity_target_y,
            planner_velocity_target_yaw=planner_velocity_target_yaw,
            use_best_candidate=use_best_candidate,
            terminal_value=terminal_value,
            disagreement_penalty=disagreement_penalty,
            model_policy_candidate_count=model_policy_candidate_count,
        )
        self.elites = elites
        self.iterations = iterations

    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self._warm_start_mean(obs)
        std = torch.full_like(mean, self.temperature)
        policy_controls = self._policy_candidate_controls(obs)

        for _ in range(self.iterations):
            noise = torch.randn(
                (obs.shape[0], self.candidates, self.control_horizon, self.action_dim),
                device=obs.device,
                dtype=obs.dtype,
            )
            controls = mean.unsqueeze(1) + std.unsqueeze(1) * noise
            controls = self._clip_controls(controls)
            controls = self._inject_prior_candidates(obs, controls, policy_controls)
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
        prior_candidate_fraction: float = 0.1,
        prior_candidate_noise: float = 0.02,
        prior_command_candidate_fraction: float = 0.0,
        prior_command_noise: float = 0.1,
        prior_command_start: int = -1,
        prior_command_dim: int = 3,
        prior_control_mode: str = "residual",
        action_bounds_finite: bool = True,
        planner_velocity_objective_weight: float = 0.0,
        planner_velocity_target_x: float = 0.0,
        planner_velocity_target_y: float = 0.0,
        planner_velocity_target_yaw: float = 0.0,
        use_best_candidate: bool = False,
        terminal_value: bool = False,
        disagreement_penalty: float = 0.0,
        model_policy_candidate_count: int = 0,
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
            prior_candidate_fraction=prior_candidate_fraction,
            prior_candidate_noise=prior_candidate_noise,
            prior_command_candidate_fraction=prior_command_candidate_fraction,
            prior_command_noise=prior_command_noise,
            prior_command_start=prior_command_start,
            prior_command_dim=prior_command_dim,
            prior_control_mode=prior_control_mode,
            action_bounds_finite=action_bounds_finite,
            planner_velocity_objective_weight=planner_velocity_objective_weight,
            planner_velocity_target_x=planner_velocity_target_x,
            planner_velocity_target_y=planner_velocity_target_y,
            planner_velocity_target_yaw=planner_velocity_target_yaw,
            use_best_candidate=use_best_candidate,
            terminal_value=terminal_value,
            disagreement_penalty=disagreement_penalty,
            model_policy_candidate_count=model_policy_candidate_count,
        )
        self.iterations = iterations
        self.lambda_ = lambda_

    def plan(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self._warm_start_mean(obs)
        std = torch.full_like(mean, self.temperature)
        best_controls = mean
        policy_controls = self._policy_candidate_controls(obs)

        for _ in range(self.iterations):
            noise = torch.randn(
                (obs.shape[0], self.candidates, self.control_horizon, self.action_dim),
                device=obs.device,
                dtype=obs.dtype,
            )
            controls = mean.unsqueeze(1) + std.unsqueeze(1) * noise
            controls = self._clip_controls(controls)
            controls = self._inject_prior_candidates(obs, controls, policy_controls)
            action_sequences = self._expand_controls(controls)
            returns = self.evaluate_sequences(obs, action_sequences)
            self._record_candidate_diagnostics(returns)
            best_indices = returns.argmax(dim=1)
            best_controls = controls.gather(
                1,
                best_indices.view(-1, 1, 1, 1).expand(-1, 1, self.control_horizon, self.action_dim),
            ).squeeze(1)

            shifted_returns = returns - returns.max(dim=1, keepdim=True).values
            weights = torch.softmax(shifted_returns / max(self.lambda_, 1e-6), dim=1)
            mean = (weights.unsqueeze(-1).unsqueeze(-1) * controls).sum(dim=1)
            centered = controls - mean.unsqueeze(1)
            variance = (weights.unsqueeze(-1).unsqueeze(-1) * centered.square()).sum(dim=1)
            std = variance.sqrt().clamp_min(1e-3)

        if self.use_best_candidate:
            mean = best_controls
        mean = self._maybe_fallback_to_prior(obs, mean)
        self._prev_mean = mean.detach()
        return self._first_action_from_controls(obs, mean)


class LatentMPPIPlanner:
    """TD-MPC2-style planner over a latent world model."""

    def __init__(
        self,
        model: LatentWorldModel,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        horizon: int = 30,
        candidates: int = 512,
        elites: int = 64,
        iterations: int = 5,
        discount: float = 0.99,
        temperature: float = 0.5,
        lambda_: float = 1.0,
        min_std: float = 0.05,
        max_std: float = 2.0,
        num_pi_trajs: int = 24,
        action_spline_knots: int = 0,
        action_prior: Callable[[torch.Tensor], torch.Tensor] | None = None,
        prior_candidate_fraction: float = 0.1,
        prior_candidate_noise: float = 0.02,
        action_bounds_finite: bool = True,
        use_best_candidate: bool = False,
        action_noise: bool = True,
        use_continue_model: bool = False,
        hard_continue_model: bool = False,
        continue_threshold: float = 0.5,
        **_: object,
    ):
        self.model = model
        self.action_low = action_low
        self.action_high = action_high
        self.horizon = horizon
        self.candidates = candidates
        self.elites = min(max(1, elites), candidates)
        self.iterations = iterations
        self.discount = discount
        self.temperature = temperature
        self.lambda_ = lambda_
        self.min_std = min_std
        self.max_std = max(max_std, min_std)
        self.num_pi_trajs = min(max(0, num_pi_trajs), candidates)
        self.action_dim = action_low.numel()
        self.action_spline_knots = action_spline_knots if 1 < action_spline_knots < horizon else 0
        self.action_prior = action_prior
        self.prior_candidate_fraction = prior_candidate_fraction
        self.prior_candidate_noise = prior_candidate_noise
        self.action_bounds_finite = action_bounds_finite
        self.use_best_candidate = use_best_candidate
        self.action_noise = action_noise
        self.use_continue_model = use_continue_model
        self.hard_continue_model = hard_continue_model
        self.continue_threshold = continue_threshold
        self._prev_mean: torch.Tensor | None = None
        self.last_diagnostics: dict[str, float] = {}

    @property
    def control_horizon(self) -> int:
        return self.action_spline_knots or self.horizon

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

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.action_bounds_finite:
            return actions.clamp(-1.0, 1.0)
        return torch.max(
            torch.min(actions, self.action_high.view(*([1] * (actions.ndim - 1)), -1)),
            self.action_low.view(*([1] * (actions.ndim - 1)), -1),
        )

    def _sample_actions_at_knots(self, action_sequences: torch.Tensor) -> torch.Tensor:
        knot_positions = torch.linspace(0, self.horizon - 1, self.control_horizon, device=action_sequences.device)
        knot_indices = knot_positions.round().long().clamp_(0, self.horizon - 1)
        return action_sequences[..., knot_indices, :]

    def _interpolate_action_spline(self, action_knots: torch.Tensor) -> torch.Tensor:
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

    def _controls_from_actions(self, action_sequences: torch.Tensor) -> torch.Tensor:
        if not self.action_spline_knots:
            return action_sequences
        return self._sample_actions_at_knots(action_sequences)

    def _warm_start_mean(self, obs: torch.Tensor) -> torch.Tensor:
        shape = (obs.shape[0], self.control_horizon, self.action_dim)
        if self._prev_mean is None or self._prev_mean.shape != shape or self._prev_mean.device != obs.device:
            return torch.zeros(shape, device=obs.device, dtype=obs.dtype)
        shifted = torch.zeros_like(self._prev_mean)
        if not self.action_spline_knots:
            shifted[:, :-1] = self._prev_mean[:, 1:]
            return shifted

        previous_actions = self._expand_controls(self._prev_mean)
        shifted_actions = torch.zeros((obs.shape[0], self.horizon, self.action_dim), device=obs.device, dtype=obs.dtype)
        shifted_actions[:, :-1] = previous_actions[:, 1:]
        return self._controls_from_actions(shifted_actions)

    @torch.no_grad()
    def _rollout_policy_actions(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        z = self.model.encode(obs)
        actions = []
        for _ in range(self.horizon):
            action = self._clip_actions(self.model.pi(z, deterministic=deterministic))
            actions.append(action)
            z = self.model.next(z, action)
        return torch.stack(actions, dim=1)

    def _policy_candidate_controls(self, obs: torch.Tensor, z0: torch.Tensor | None = None) -> torch.Tensor:
        controls = torch.empty(
            (obs.shape[0], self.num_pi_trajs, self.control_horizon, self.action_dim),
            device=obs.device,
            dtype=obs.dtype,
        )
        if self.num_pi_trajs == 0:
            return controls

        next_index = 0

        if self.action_prior is not None and next_index < self.num_pi_trajs:
            external_prior = self._clip_actions(self.action_prior(obs))
            external_actions = external_prior.unsqueeze(1).expand(-1, self.horizon, -1)
            controls[:, next_index] = self._controls_from_actions(external_actions)
            next_index += 1

        stochastic_count = self.num_pi_trajs - next_index
        if stochastic_count > 0:
            z = self.model.encode(obs) if z0 is None else z0
            z = z.unsqueeze(1).expand(-1, stochastic_count, -1).reshape(obs.shape[0] * stochastic_count, -1)
            actions = []
            for _ in range(self.horizon):
                action = self._clip_actions(self.model.pi(z, deterministic=False))
                actions.append(action.view(obs.shape[0], stochastic_count, self.action_dim))
                z = self.model.next(z, action)
            action_sequences = torch.stack(actions, dim=2)
            controls[:, next_index:] = self._controls_from_actions(action_sequences)
        return controls

    def _elite_score_weights(self, elite_returns: torch.Tensor) -> torch.Tensor:
        centered = elite_returns - elite_returns.max(dim=1, keepdim=True).values
        weights = torch.softmax(float(self.temperature) * centered, dim=1)
        return weights

    @torch.no_grad()
    def evaluate_sequences(
        self,
        obs: torch.Tensor,
        action_sequences: torch.Tensor,
        z0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, candidates, _, action_dim = action_sequences.shape
        z = self.model.encode(obs) if z0 is None else z0
        z = z.unsqueeze(1).expand(-1, candidates, -1).reshape(batch_size * candidates, -1)
        returns = torch.zeros(batch_size * candidates, device=obs.device, dtype=obs.dtype)
        discounts = torch.ones_like(returns)
        alive = torch.ones_like(returns)

        for t in range(self.horizon):
            actions_t = action_sequences[:, :, t, :].reshape(batch_size * candidates, action_dim)
            reward = self.model.reward(z, actions_t).squeeze(-1)
            z = self.model.next(z, actions_t)
            returns = returns + discounts * alive * reward
            if self.use_continue_model:
                continue_prob = self.model.continue_logits(z).sigmoid().squeeze(-1)
                if self.hard_continue_model:
                    alive = alive * (continue_prob > self.continue_threshold).to(alive.dtype)
                else:
                    alive = alive * continue_prob
            discounts = discounts * self.discount

        terminal_action = self.model.pi(z, deterministic=False)
        terminal_value = self.model.Q(z, terminal_action, return_type="avg").squeeze(-1)
        returns = returns + discounts * alive * terminal_value
        return returns.view(batch_size, candidates)

    def _record_diagnostics(self, returns: torch.Tensor, action: torch.Tensor) -> None:
        returns = returns.detach()
        self.last_diagnostics = {
            "planner_candidate_return_mean": float(returns.mean().item()),
            "planner_candidate_return_best": float(returns.max(dim=1).values.mean().item()),
            "planner_candidate_return_std": float(returns.std(unbiased=False).item()),
            "planner_prior_candidate_fraction": float(self.num_pi_trajs / self.candidates),
            "planner_prior_command_candidate_fraction": 0.0,
            "planner_full_action_mode": 1.0,
            "planner_best_candidate_mode": float(self.use_best_candidate),
            "planner_action_noise_mode": float(self.action_noise),
            "planner_continue_model_mode": float(self.use_continue_model),
            "planner_hard_continue_model_mode": float(self.hard_continue_model),
            "planner_prior_action_norm_mean": 0.0,
            "planner_residual_norm_mean": 0.0,
            "planner_residual_abs_mean": 0.0,
            "planner_final_action_norm_mean": float(action.detach().norm(dim=-1).mean().item()),
            "planner_residual_to_prior_norm": 0.0,
            "planner_selected_prior_fraction": 0.0,
            "planner_predicted_plan_return_mean": 0.0,
            "planner_predicted_prior_return_mean": 0.0,
            "planner_predicted_return_margin_mean": 0.0,
            "planner_predicted_return_margin_min": 0.0,
            "planner_prior_fallback_fraction": 0.0,
        }

    def plan(self, obs: torch.Tensor, eval_mode: bool = False, t0: bool = False) -> torch.Tensor:
        if t0:
            self.reset()
        z0 = self.model.encode(obs)
        mean = self._warm_start_mean(obs)
        std = torch.full_like(mean, self.max_std).clamp_(self.min_std, self.max_std)
        policy_controls = self._policy_candidate_controls(obs, z0)
        random_candidates = self.candidates - self.num_pi_trajs
        final_elite_controls = mean.unsqueeze(1)
        final_elite_returns = torch.zeros((obs.shape[0], 1), device=obs.device, dtype=obs.dtype)
        last_returns = torch.zeros((obs.shape[0], self.candidates), device=obs.device, dtype=obs.dtype)

        for _ in range(self.iterations):
            if random_candidates > 0:
                noise = torch.randn(
                    (obs.shape[0], random_candidates, self.control_horizon, self.action_dim),
                    device=obs.device,
                    dtype=obs.dtype,
                )
                sampled_controls = self._clip_actions(mean.unsqueeze(1) + std.unsqueeze(1) * noise)
                controls = torch.cat((policy_controls, sampled_controls), dim=1)
            else:
                controls = policy_controls

            actions = self._expand_controls(controls)
            returns = self.evaluate_sequences(obs, actions, z0)
            returns = returns.nan_to_num(0.0, posinf=0.0, neginf=0.0)
            last_returns = returns

            final_elite_returns, elite_indices = returns.topk(self.elites, dim=1)
            final_elite_controls = controls.gather(
                1,
                elite_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.control_horizon, self.action_dim),
            )
            weights = self._elite_score_weights(final_elite_returns)
            mean = (weights.unsqueeze(-1).unsqueeze(-1) * final_elite_controls).sum(dim=1)
            variance = (weights.unsqueeze(-1).unsqueeze(-1) * (final_elite_controls - mean.unsqueeze(1)).square()).sum(dim=1)
            std = variance.sqrt().clamp_(self.min_std, self.max_std)

        if self.use_best_candidate:
            selected_indices = final_elite_returns.argmax(dim=1)
        else:
            gumbel = -torch.empty_like(final_elite_returns).exponential_().log()
            selected_indices = (float(self.temperature) * final_elite_returns + gumbel).argmax(dim=1)
        selected_controls = final_elite_controls.gather(
            1,
            selected_indices.view(-1, 1, 1, 1).expand(-1, 1, self.control_horizon, self.action_dim),
        ).squeeze(1)
        self._prev_mean = mean.detach()
        action = self._clip_actions(self._expand_controls(selected_controls)[:, 0])
        if self.action_noise and not eval_mode:
            action = self._clip_actions(action + std[:, 0] * torch.randn_like(action))
        self._record_diagnostics(last_returns, action)
        return action


def build_planner(
    planner_name: str,
    model: DynamicsEnsemble | StateWorldModel | LatentWorldModel,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    horizon: int,
    candidates: int,
    elites: int,
    iterations: int,
    discount: float,
    temperature: float,
    lambda_: float,
    min_std: float = 0.05,
    max_std: float = 2.0,
    num_pi_trajs: int = 24,
    action_spline_knots: int = 0,
    action_prior: Callable[[torch.Tensor], torch.Tensor] | None = None,
    prior_residual_scale: float = 0.3,
    prior_residual_penalty: float = 0.0,
    prior_acceptance_margin: float = 0.0,
    prior_fallback: bool = True,
    prior_candidate_fraction: float = 0.1,
    prior_candidate_noise: float = 0.02,
    prior_command_candidate_fraction: float = 0.0,
    prior_command_noise: float = 0.1,
    prior_command_start: int = -1,
    prior_command_dim: int = 3,
    prior_control_mode: str = "residual",
    action_bounds_finite: bool = True,
    action_noise: bool = True,
    use_continue_model: bool = False,
    hard_continue_model: bool = False,
    continue_threshold: float = 0.5,
    planner_velocity_objective_weight: float = 0.0,
    planner_velocity_target_x: float = 0.0,
    planner_velocity_target_y: float = 0.0,
    planner_velocity_target_yaw: float = 0.0,
    use_best_candidate: bool = False,
    terminal_value: bool = False,
    disagreement_penalty: float = 0.0,
    model_policy_candidate_count: int = 0,
) -> TrajectoryPlanner | LatentMPPIPlanner:
    if getattr(model, "is_latent_world_model", False):
        if planner_name != "mppi":
            raise ValueError("LatentWorldModel currently supports only the mppi planner.")
        return LatentMPPIPlanner(
            model=model,
            action_low=action_low,
            action_high=action_high,
            horizon=horizon,
            candidates=candidates,
            elites=elites,
            iterations=iterations,
            discount=discount,
            temperature=temperature,
            lambda_=lambda_,
            min_std=min_std,
            max_std=max_std,
            num_pi_trajs=num_pi_trajs,
            action_spline_knots=action_spline_knots,
            action_prior=action_prior,
            prior_candidate_fraction=prior_candidate_fraction,
            prior_candidate_noise=prior_candidate_noise,
            action_bounds_finite=action_bounds_finite,
            use_best_candidate=use_best_candidate,
            action_noise=action_noise,
            use_continue_model=use_continue_model,
            hard_continue_model=hard_continue_model,
            continue_threshold=continue_threshold,
        )
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
            prior_candidate_fraction=prior_candidate_fraction,
            prior_candidate_noise=prior_candidate_noise,
            prior_command_candidate_fraction=prior_command_candidate_fraction,
            prior_command_noise=prior_command_noise,
            prior_command_start=prior_command_start,
            prior_command_dim=prior_command_dim,
            prior_control_mode=prior_control_mode,
            action_bounds_finite=action_bounds_finite,
            planner_velocity_objective_weight=planner_velocity_objective_weight,
            planner_velocity_target_x=planner_velocity_target_x,
            planner_velocity_target_y=planner_velocity_target_y,
            planner_velocity_target_yaw=planner_velocity_target_yaw,
            use_best_candidate=use_best_candidate,
            terminal_value=terminal_value,
            disagreement_penalty=disagreement_penalty,
            model_policy_candidate_count=model_policy_candidate_count,
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
            prior_candidate_fraction=prior_candidate_fraction,
            prior_candidate_noise=prior_candidate_noise,
            prior_command_candidate_fraction=prior_command_candidate_fraction,
            prior_command_noise=prior_command_noise,
            prior_command_start=prior_command_start,
            prior_command_dim=prior_command_dim,
            prior_control_mode=prior_control_mode,
            action_bounds_finite=action_bounds_finite,
            planner_velocity_objective_weight=planner_velocity_objective_weight,
            planner_velocity_target_x=planner_velocity_target_x,
            planner_velocity_target_y=planner_velocity_target_y,
            planner_velocity_target_yaw=planner_velocity_target_yaw,
            use_best_candidate=use_best_candidate,
            terminal_value=terminal_value,
            disagreement_penalty=disagreement_penalty,
            model_policy_candidate_count=model_policy_candidate_count,
        )
    raise ValueError(f"Unsupported planner: {planner_name}")
