from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointSpaceCostConfig:
    w_q: float = 1.0
    w_dq: float = 0.01
    w_u: float = 0.001
    w_du: float = 0.001
    w_terminal: float = 1.0
    w_joint_limit: float = 1.0


def _match_horizon(target: torch.Tensor, horizon: int) -> torch.Tensor:
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.shape[1] < horizon:
        raise ValueError(f"target horizon={target.shape[1]} is shorter than required horizon={horizon}")
    return target[:, :horizon]


def joint_space_tracking_cost(
    pred_states: torch.Tensor,
    q_des: torch.Tensor,
    dq_des: torch.Tensor | None,
    actuator_q_ref: torch.Tensor,
    delta_q_ref: torch.Tensor,
    previous_q_ref: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    config: JointSpaceCostConfig,
) -> torch.Tensor:
    if pred_states.ndim != 3:
        raise ValueError(f"pred_states must have shape [batch, horizon + 1, state_dim], got {tuple(pred_states.shape)}")
    n_joints = pred_states.shape[-1] // 2
    horizon = pred_states.shape[1] - 1
    q_pred = pred_states[:, 1:, :n_joints]
    dq_pred = pred_states[:, 1:, n_joints : 2 * n_joints]
    q_target = _match_horizon(q_des, horizon).to(device=pred_states.device, dtype=pred_states.dtype)
    if q_target.shape[0] == 1:
        q_target = q_target.expand(pred_states.shape[0], -1, -1)

    cost = torch.zeros(pred_states.shape[0], device=pred_states.device, dtype=pred_states.dtype)
    cost = cost + float(config.w_q) * torch.sum(torch.square(q_pred - q_target), dim=(1, 2))
    if dq_des is not None and config.w_dq != 0.0:
        dq_target = _match_horizon(dq_des, horizon).to(device=pred_states.device, dtype=pred_states.dtype)
        if dq_target.shape[0] == 1:
            dq_target = dq_target.expand(pred_states.shape[0], -1, -1)
        cost = cost + float(config.w_dq) * torch.sum(torch.square(dq_pred - dq_target), dim=(1, 2))
    elif config.w_dq != 0.0:
        cost = cost + float(config.w_dq) * torch.sum(torch.square(dq_pred), dim=(1, 2))

    if config.w_u != 0.0:
        current_q = pred_states[:, :-1, :n_joints]
        cost = cost + float(config.w_u) * torch.sum(torch.square(actuator_q_ref - current_q), dim=(1, 2))
    if config.w_du != 0.0:
        q_ref_prev = previous_q_ref.to(device=actuator_q_ref.device, dtype=actuator_q_ref.dtype)
        if q_ref_prev.ndim == 1:
            q_ref_prev = q_ref_prev.unsqueeze(0).expand(actuator_q_ref.shape[0], -1)
        q_ref_steps = torch.cat([q_ref_prev.unsqueeze(1), actuator_q_ref], dim=1)
        q_ref_rate = q_ref_steps[:, 1:] - q_ref_steps[:, :-1]
        delta_rate = delta_q_ref[:, 1:] - delta_q_ref[:, :-1] if delta_q_ref.shape[1] > 1 else torch.zeros_like(delta_q_ref[:, :1])
        cost = cost + float(config.w_du) * (
            torch.sum(torch.square(q_ref_rate), dim=(1, 2)) + torch.sum(torch.square(delta_rate), dim=(1, 2))
        )
    if config.w_terminal != 0.0:
        cost = cost + float(config.w_terminal) * torch.sum(torch.square(q_pred[:, -1] - q_target[:, -1]), dim=1)
    if config.w_joint_limit != 0.0:
        low_violation = torch.relu(joint_low.to(q_pred.device, q_pred.dtype) - q_pred)
        high_violation = torch.relu(q_pred - joint_high.to(q_pred.device, q_pred.dtype))
        cost = cost + float(config.w_joint_limit) * torch.sum(torch.square(low_violation) + torch.square(high_violation), dim=(1, 2))
    return cost
