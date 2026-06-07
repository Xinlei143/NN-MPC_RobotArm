from __future__ import annotations

import torch


def joint_bounds_with_margin(
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    joint_limit_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = joint_low + float(joint_limit_margin)
    high = joint_high - float(joint_limit_margin)
    if torch.any(low >= high):
        raise ValueError("joint_limit_margin leaves no valid joint range")
    return low, high


def clip_to_joint_limits(
    q_ref: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    joint_limit_margin: float = 0.0,
) -> torch.Tensor:
    low, high = joint_bounds_with_margin(joint_low.to(q_ref.device), joint_high.to(q_ref.device), joint_limit_margin)
    return torch.minimum(torch.maximum(q_ref, low), high)


def apply_rate_limit(
    q_ref_sequence: torch.Tensor,
    previous_q_ref: torch.Tensor,
    q_ref_rate_limit: float | None,
) -> torch.Tensor:
    if q_ref_rate_limit is None or q_ref_rate_limit <= 0:
        return q_ref_sequence
    limited = []
    previous = previous_q_ref.to(device=q_ref_sequence.device, dtype=q_ref_sequence.dtype)
    if previous.ndim == 1:
        previous = previous.unsqueeze(0).expand(q_ref_sequence.shape[0], -1)
    limit = float(q_ref_rate_limit)
    for step_idx in range(q_ref_sequence.shape[1]):
        target = q_ref_sequence[:, step_idx]
        delta = torch.clamp(target - previous, min=-limit, max=limit)
        previous = previous + delta
        limited.append(previous)
    return torch.stack(limited, dim=1)


def apply_delta_rate_limit(delta_sequence: torch.Tensor, delta_rate_limit: float | None) -> torch.Tensor:
    if delta_rate_limit is None or delta_rate_limit <= 0:
        return delta_sequence
    limited = []
    previous = torch.zeros_like(delta_sequence[:, 0])
    limit = float(delta_rate_limit)
    for step_idx in range(delta_sequence.shape[1]):
        target = delta_sequence[:, step_idx]
        delta = torch.clamp(target - previous, min=-limit, max=limit)
        previous = previous + delta
        limited.append(previous)
    return torch.stack(limited, dim=1)
