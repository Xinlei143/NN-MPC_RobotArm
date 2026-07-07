from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class JointSpaceCostConfig:
    w_q: float = 1.0
    w_dq: float = 0.05
    w_u_offset: float = 0.05
    w_dqref: float = 0.05
    w_ddqref: float = 0.02
    w_terminal: float = 0.5
    w_joint_limit: float = 2.0
    q_amp_fraction: float = 0.2
    q_tol: float = 0.04
    dq_scale: float = 0.5
    u_offset_scale: float = 0.2
    dqref_scale: float = 0.08
    ddqref_scale: float = 0.05
    joint_limit_safe_margin: float = 0.08
    joint_limit_temp: float = 0.02
    velocity_cost_mode: str = "track"


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
    actuator_q_ref = actuator_q_ref.to(device=pred_states.device, dtype=pred_states.dtype)

    q_amp = q_target.max(dim=1).values - q_target.min(dim=1).values
    q_scale = torch.clamp(float(config.q_amp_fraction) * q_amp, min=float(config.q_tol)).unsqueeze(1)
    e_q = (q_pred - q_target) / q_scale

    cost = torch.zeros(pred_states.shape[0], device=pred_states.device, dtype=pred_states.dtype)
    if config.w_q != 0.0:
        cost = cost + float(config.w_q) * torch.mean(torch.square(e_q), dim=(1, 2))
    if config.w_dq != 0.0:
        if config.velocity_cost_mode not in {"track", "damping"}:
            raise ValueError(f"velocity_cost_mode must be 'track' or 'damping', got {config.velocity_cost_mode!r}")
        if dq_des is not None and config.velocity_cost_mode == "track":
            dq_target = _match_horizon(dq_des, horizon).to(device=pred_states.device, dtype=pred_states.dtype)
            if dq_target.shape[0] == 1:
                dq_target = dq_target.expand(pred_states.shape[0], -1, -1)
            e_dq = (dq_pred - dq_target) / float(config.dq_scale)
        else:
            e_dq = dq_pred / float(config.dq_scale)
        cost = cost + float(config.w_dq) * torch.mean(torch.square(e_dq), dim=(1, 2))

    if config.w_u_offset != 0.0:
        u_offset = (actuator_q_ref - q_pred) / float(config.u_offset_scale)
        cost = cost + float(config.w_u_offset) * torch.mean(torch.square(u_offset), dim=(1, 2))
    if config.w_dqref != 0.0 or config.w_ddqref != 0.0:
        q_ref_prev = previous_q_ref.to(device=actuator_q_ref.device, dtype=actuator_q_ref.dtype)
        if q_ref_prev.ndim == 1:
            q_ref_prev = q_ref_prev.unsqueeze(0).expand(actuator_q_ref.shape[0], -1)
        elif q_ref_prev.shape[0] == 1 and actuator_q_ref.shape[0] > 1:
            q_ref_prev = q_ref_prev.expand(actuator_q_ref.shape[0], -1)
        qref_full = torch.cat([q_ref_prev.unsqueeze(1), actuator_q_ref], dim=1)
        dqref = qref_full[:, 1:] - qref_full[:, :-1]
        if config.w_dqref != 0.0:
            e_dqref = dqref / float(config.dqref_scale)
            cost = cost + float(config.w_dqref) * torch.mean(torch.square(e_dqref), dim=(1, 2))
        if config.w_ddqref != 0.0:
            if dqref.shape[1] > 1:
                ddqref = dqref[:, 1:] - dqref[:, :-1]
                e_ddqref = ddqref / float(config.ddqref_scale)
                cost = cost + float(config.w_ddqref) * torch.mean(torch.square(e_ddqref), dim=(1, 2))
    if config.w_terminal != 0.0:
        cost = cost + float(config.w_terminal) * torch.mean(torch.square(e_q[:, -1]), dim=1)
    if config.w_joint_limit != 0.0:
        joint_low = joint_low.to(q_pred.device, q_pred.dtype)
        joint_high = joint_high.to(q_pred.device, q_pred.dtype)
        margin_low = q_pred - joint_low
        margin_high = joint_high - q_pred
        margin = torch.minimum(margin_low, margin_high)
        barrier = F.softplus((float(config.joint_limit_safe_margin) - margin) / float(config.joint_limit_temp))
        cost = cost + float(config.w_joint_limit) * torch.mean(barrier, dim=(1, 2))
    return cost
