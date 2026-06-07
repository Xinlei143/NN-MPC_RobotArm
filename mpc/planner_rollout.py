from __future__ import annotations

from dataclasses import dataclass

import torch

from learned_dynamics.rollout import rollout_dynamics_batch
from mpc.constraints import apply_delta_rate_limit, apply_rate_limit, clip_to_joint_limits
from mpc.cost_functions import JointSpaceCostConfig, joint_space_tracking_cost


def construct_actuator_q_ref_sequence(
    candidate_sequence: torch.Tensor,
    current_q: torch.Tensor,
    previous_q_ref: torch.Tensor,
    mode: str,
    delta_base: str,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    joint_limit_margin: float,
    delta_q_ref_max: float | None = None,
    q_ref_rate_limit: float | None = None,
    delta_rate_limit: float | None = None,
) -> torch.Tensor:
    if candidate_sequence.ndim != 3:
        raise ValueError(f"candidate_sequence must have shape [batch, horizon, action_dim], got {tuple(candidate_sequence.shape)}")
    if mode not in {"delta", "absolute"}:
        raise ValueError(f"mode must be 'delta' or 'absolute', got {mode!r}")
    device = candidate_sequence.device
    dtype = candidate_sequence.dtype
    joint_low = joint_low.to(device=device, dtype=dtype)
    joint_high = joint_high.to(device=device, dtype=dtype)
    previous_q_ref = previous_q_ref.to(device=device, dtype=dtype)
    current_q = current_q.to(device=device, dtype=dtype)

    if mode == "absolute":
        q_ref = candidate_sequence
    else:
        delta_q_ref = candidate_sequence
        if delta_q_ref_max is not None and delta_q_ref_max > 0:
            delta_q_ref = torch.clamp(delta_q_ref, min=-float(delta_q_ref_max), max=float(delta_q_ref_max))
        delta_q_ref = apply_delta_rate_limit(delta_q_ref, delta_rate_limit)
        if delta_base == "previous_q_ref":
            base = previous_q_ref
        elif delta_base == "current_q":
            base = current_q
        else:
            raise ValueError(f"delta_base must be 'previous_q_ref' or 'current_q', got {delta_base!r}")
        if base.ndim == 1:
            base = base.unsqueeze(0).expand(candidate_sequence.shape[0], -1)
        q_ref = torch.cumsum(delta_q_ref, dim=1) + base.unsqueeze(1)

    q_ref = apply_rate_limit(q_ref, previous_q_ref, q_ref_rate_limit)
    return clip_to_joint_limits(q_ref, joint_low, joint_high, joint_limit_margin)


@dataclass(frozen=True)
class PlannerRolloutConfig:
    mode: str = "delta"
    delta_base: str = "previous_q_ref"
    delta_q_ref_max: float | None = None
    q_ref_rate_limit: float | None = None
    delta_rate_limit: float | None = None
    joint_limit_margin: float = 0.0
    rollout_batch_size: int | None = None


@dataclass
class LearnedDynamicsPlanner:
    model: torch.nn.Module
    normalizer: object
    model_type: str
    state_dim: int
    target_mode: str
    control_dt: float
    initial_history: torch.Tensor
    q_des: torch.Tensor
    dq_des: torch.Tensor | None
    previous_q_ref: torch.Tensor
    joint_low: torch.Tensor
    joint_high: torch.Tensor
    cost_config: JointSpaceCostConfig
    rollout_config: PlannerRolloutConfig

    def evaluate(self, candidate_delta_q_ref: torch.Tensor) -> dict[str, torch.Tensor]:
        current_q = self.initial_history[-1, : self.state_dim // 2]
        q_ref_sequences = construct_actuator_q_ref_sequence(
            candidate_delta_q_ref,
            current_q=current_q,
            previous_q_ref=self.previous_q_ref,
            mode=self.rollout_config.mode,
            delta_base=self.rollout_config.delta_base,
            joint_low=self.joint_low,
            joint_high=self.joint_high,
            joint_limit_margin=self.rollout_config.joint_limit_margin,
            delta_q_ref_max=self.rollout_config.delta_q_ref_max,
            q_ref_rate_limit=self.rollout_config.q_ref_rate_limit,
            delta_rate_limit=self.rollout_config.delta_rate_limit,
        )
        pred_states = rollout_dynamics_batch(
            model=self.model,
            normalizer=self.normalizer,
            model_type=self.model_type,
            initial_history=self.initial_history,
            future_q_ref=q_ref_sequences,
            state_dim=self.state_dim,
            target_mode=self.target_mode,
            control_dt=self.control_dt,
            rollout_batch_size=self.rollout_config.rollout_batch_size,
        )
        costs = joint_space_tracking_cost(
            pred_states=pred_states,
            q_des=self.q_des.to(device=pred_states.device, dtype=pred_states.dtype),
            dq_des=None if self.dq_des is None else self.dq_des.to(device=pred_states.device, dtype=pred_states.dtype),
            actuator_q_ref=q_ref_sequences,
            delta_q_ref=candidate_delta_q_ref,
            previous_q_ref=self.previous_q_ref.to(device=pred_states.device, dtype=pred_states.dtype),
            joint_low=self.joint_low.to(device=pred_states.device, dtype=pred_states.dtype),
            joint_high=self.joint_high.to(device=pred_states.device, dtype=pred_states.dtype),
            config=self.cost_config,
        )
        return {"costs": costs, "q_ref_sequences": q_ref_sequences, "pred_states": pred_states}
