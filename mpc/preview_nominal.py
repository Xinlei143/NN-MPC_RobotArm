"""Reference indexing for preview-IK nominal commands in residual MPC."""
from __future__ import annotations

import numpy as np


def nominal_index(step: int, preview_steps: int) -> int:
    """Return the actuator nominal index for control/planner step ``step``.

    The tracking target remains ``reference[step + 1]``.  Only the nominal
    actuator command is advanced by ``preview_steps``.
    """
    if step < 0:
        raise ValueError("step must be non-negative")
    if preview_steps < 0:
        raise ValueError("preview_steps must be non-negative")
    return step + 1 + preview_steps


def nominal_command(reference: np.ndarray, step: int, preview_steps: int) -> np.ndarray:
    """Return the raw IK nominal command for one execution step."""
    index = nominal_index(step, preview_steps)
    if index >= len(reference):
        raise IndexError(f"preview nominal index {index} is outside reference length {len(reference)}")
    return np.asarray(reference[index], dtype=np.float32).copy()


def nominal_window(reference: np.ndarray, anchor_step: int, horizon: int, preview_steps: int) -> np.ndarray:
    """Return the previewed nominal sequence while leaving cost targets unshifted."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    start = nominal_index(anchor_step, preview_steps)
    stop = start + horizon
    if stop > len(reference):
        raise IndexError(f"preview nominal window [{start}:{stop}] exceeds reference length {len(reference)}")
    return np.asarray(reference[start:stop], dtype=np.float32)
