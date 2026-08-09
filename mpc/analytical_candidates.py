"""Small, deterministic candidate and residual-direction helpers.

These helpers are intentionally independent of the learned model.  They are
used for diagnostics and for optional CEM branches; the default controller
does not enable either behavior.
"""
from __future__ import annotations

import numpy as np


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def parse_preview_steps(value: str | None) -> tuple[int, ...]:
    """Parse a comma-separated list of non-negative preview steps."""
    if value is None or not str(value).strip():
        return ()
    result: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step < 0:
            raise ValueError("preview steps must be non-negative")
        if step not in result:
            result.append(step)
    return tuple(result)


def build_preview_residual_candidates(
    reference: np.ndarray,
    *,
    anchor: int,
    horizon: int,
    nominal: np.ndarray,
    residual_max: np.ndarray,
    preview_steps: tuple[int, ...],
    nominal_preview_steps: int = 0,
) -> dict[str, np.ndarray]:
    """Build normalized residual candidates for ``q_des[t+k]`` branches.

    ``nominal`` is the command sequence used by the residual planner at the
    current anchor.  Returned values are normalized to ``[-1, 1]`` and have
    shape ``[horizon, joints]`` so they can be passed directly to the planner
    residual parameterizer.
    """
    reference = np.asarray(reference, dtype=np.float32)
    nominal = np.asarray(nominal, dtype=np.float32)
    residual_max = np.asarray(residual_max, dtype=np.float32)
    if reference.ndim != 2 or nominal.ndim != 2 or residual_max.ndim != 1:
        raise ValueError("reference, nominal and residual_max have incompatible ranks")
    if nominal.shape[0] != horizon or nominal.shape[1] != reference.shape[1]:
        raise ValueError("nominal must have shape [horizon, joints]")
    if residual_max.shape != (reference.shape[1],) or np.any(residual_max <= 0):
        raise ValueError("residual_max must contain one positive value per joint")
    candidates: dict[str, np.ndarray] = {}
    for preview in preview_steps:
        start = int(anchor) + int(nominal_preview_steps) + int(preview)
        stop = start + int(horizon)
        if start < 0 or stop > reference.shape[0]:
            continue
        target = reference[start:stop]
        candidates[f"preview:{preview}"] = np.clip(
            (target - nominal) / residual_max[None, :], -1.0, 1.0
        ).astype(np.float32)
    return candidates


def align_residual_to_velocity(
    residual: np.ndarray,
    dq_des: np.ndarray,
    *,
    joint_indices: tuple[int, ...] | None = None,
    velocity_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove residual components that oppose the desired velocity.

    Zero/near-zero desired velocity leaves the residual unchanged.  The
    second return value is a boolean mask of entries that were modified.
    This is a diagnostic gate, not a claim that the final controller should
    always enforce a sign constraint.
    """
    residual = np.asarray(residual, dtype=np.float32)
    dq_des = np.asarray(dq_des, dtype=np.float32)
    if residual.shape != dq_des.shape or residual.ndim != 2:
        raise ValueError("residual and dq_des must have matching [steps, joints] shapes")
    result = residual.copy()
    changed = np.zeros_like(result, dtype=bool)
    joints = tuple(range(residual.shape[1])) if joint_indices is None else tuple(joint_indices)
    for joint in joints:
        if not 0 <= joint < residual.shape[1]:
            raise ValueError(f"joint index out of range: {joint}")
        moving = np.abs(dq_des[:, joint]) > float(velocity_threshold)
        positive = moving & (dq_des[:, joint] > 0.0) & (result[:, joint] < 0.0)
        negative = moving & (dq_des[:, joint] < 0.0) & (result[:, joint] > 0.0)
        changed[:, joint] = positive | negative
        result[positive | negative, joint] = 0.0
    return result, changed

