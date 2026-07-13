from __future__ import annotations

import torch


def reconstruct_next_state(
    state: torch.Tensor,
    pred_target: torch.Tensor,
    target_mode: str,
    control_dt: float,
    n_joints: int,
) -> torch.Tensor:
    if target_mode == "delta_state":
        return state + pred_target
    if target_mode != "delta_dq":
        raise ValueError(f"target_mode must be 'delta_state' or 'delta_dq', got {target_mode!r}")
    if pred_target.shape[-1] != n_joints:
        raise ValueError(f"delta_dq target must have last dimension {n_joints}, got {pred_target.shape[-1]}")
    q = state[..., :n_joints]
    dq = state[..., n_joints : 2 * n_joints]
    dq_next = dq + pred_target
    q_next = q + dq_next * control_dt
    return torch.cat([q_next, dq_next], dim=-1)
