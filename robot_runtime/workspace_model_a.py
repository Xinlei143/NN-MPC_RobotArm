"""Bounded reference programs for real SO101 Model-A data collection.

The names and identifiers intentionally match the non-MPC Model-A MuJoCo
collector.  This module is pure NumPy: it can be tested without connecting a
robot and it never sends a command by itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Keep these compatible with dynamics_modeling.neural_dynamics.parallel_collector.
WORKSPACE_MODE_IDS = {
    "hold": 0,
    "step": 1,
    "smooth_random": 2,
    "sine": 3,
    "delta_ref_random": 4,
}

# The real-data target is intentionally smaller than the simulation corpus but
# large enough to train and independently evaluate a workspace dynamics model.
# Keep this plan in one module so the collector, offline preflight, manifests,
# and documentation cannot silently disagree.
MODEL_A_PLAN_VERSION = "so101_model_a_48x15min_v1"
MODEL_A_SESSION_SECONDS = 15.0 * 60.0
MODEL_A_TRAIN_SESSIONS = 38
MODEL_A_VALIDATION_SESSIONS = 5
MODEL_A_TEST_SESSIONS = 5
MODEL_A_TOTAL_SESSIONS = (
    MODEL_A_TRAIN_SESSIONS + MODEL_A_VALIDATION_SESSIONS + MODEL_A_TEST_SESSIONS
)
_BASE_PROGRAM_SECONDS = 5.0 * 60.0


def expected_model_a_split(session_index: int) -> str:
    """Return the fixed session-level 38/5/5 split for the real Model-A corpus."""
    if not 0 <= int(session_index) < MODEL_A_TOTAL_SESSIONS:
        raise ValueError(f"session_index must be in [0, {MODEL_A_TOTAL_SESSIONS - 1}]")
    if session_index < MODEL_A_TRAIN_SESSIONS:
        return "train"
    if session_index < MODEL_A_TRAIN_SESSIONS + MODEL_A_VALIDATION_SESSIONS:
        return "validation"
    return "test"


def model_a_collection_plan_identity() -> dict[str, object]:
    """Behavior-defining plan fields that must match across appended sessions."""
    return {
        "version": MODEL_A_PLAN_VERSION,
        "session_seconds": MODEL_A_SESSION_SECONDS,
        "total_sessions": MODEL_A_TOTAL_SESSIONS,
        "split_counts": {
            "train": MODEL_A_TRAIN_SESSIONS,
            "validation": MODEL_A_VALIDATION_SESSIONS,
            "test": MODEL_A_TEST_SESSIONS,
        },
        "program_base_seconds": _BASE_PROGRAM_SECONDS,
    }


@dataclass(frozen=True)
class WorkspaceProgram:
    name: str
    mode: str
    seconds: float
    single_joint_index: int | None = None

    @property
    def mode_id(self) -> int:
        return WORKSPACE_MODE_IDS[self.mode]


def session_program(session_index: int, *, session_seconds: float = MODEL_A_SESSION_SECONDS) -> tuple[WorkspaceProgram, ...]:
    """Return the fixed, scaled Model-A schedule for one real session."""
    expected_model_a_split(session_index)
    if not np.isfinite(session_seconds) or session_seconds <= 0:
        raise ValueError("session_seconds must be positive and finite")
    # Alternating the final excitation prevents a collection-order bias; the
    # sine joint rotates over the five controlled joints.
    final_mode = "step" if session_index % 2 == 0 else "delta_ref_random"
    scale = float(session_seconds) / _BASE_PROGRAM_SECONDS
    return (
        WorkspaceProgram("hold", "hold", 45.0 * scale),
        WorkspaceProgram("single_joint_sine", "sine", 90.0 * scale, session_index % 5),
        WorkspaceProgram("multi_joint_sine", "sine", 75.0 * scale),
        WorkspaceProgram("smooth_random", "smooth_random", 60.0 * scale),
        WorkspaceProgram(final_mode, final_mode, 30.0 * scale),
    )


def effective_workspace_bounds(
    hardware_low: np.ndarray, hardware_high: np.ndarray, margin_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(hardware_low, dtype=np.float64) + float(margin_rad)
    high = np.asarray(hardware_high, dtype=np.float64) - float(margin_rad)
    if low.shape != high.shape or low.ndim != 1 or np.any(low >= high):
        raise ValueError("hardware workspace cannot accommodate the requested margin")
    return low.astype(np.float32), high.astype(np.float32)


def _rate_and_acceleration_limit(
    targets: np.ndarray,
    start: np.ndarray,
    dt: float,
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    result = np.empty_like(targets, dtype=np.float64)
    previous = np.asarray(start, dtype=np.float64).copy()
    previous_delta = np.zeros_like(previous)
    max_delta = np.asarray(velocity_limit, dtype=np.float64) * dt
    max_delta_change = np.asarray(acceleration_limit, dtype=np.float64) * dt * dt
    for index, target in enumerate(targets):
        desired_delta = np.clip(np.asarray(target, dtype=np.float64) - previous, -max_delta, max_delta)
        delta = np.clip(desired_delta, previous_delta - max_delta_change, previous_delta + max_delta_change)
        # Do not accelerate away from a target after it is already inside one
        # position tick; this is a reference limiter, not a plant model.
        delta = np.where(np.abs(desired_delta) < np.abs(delta), desired_delta, delta)
        previous = np.clip(previous + delta, low, high)
        result[index] = previous
        previous_delta = previous - result[index - 1] if index else previous - np.asarray(start, dtype=np.float64)
    return result.astype(np.float32)


def build_workspace_reference(
    *,
    rng: np.random.Generator,
    mode: str,
    start_q: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    steps: int,
    dt: float,
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
    single_joint_index: int | None = None,
) -> np.ndarray:
    """Build one bounded, rate-limited absolute-position reference sequence."""
    if mode not in WORKSPACE_MODE_IDS:
        raise ValueError(f"unsupported workspace Model-A mode: {mode!r}")
    if steps <= 0 or dt <= 0:
        raise ValueError("steps and dt must be positive")
    start = np.asarray(start_q, dtype=np.float64)
    q_low, q_high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    vmax, amax = np.asarray(velocity_limit, dtype=np.float64), np.asarray(acceleration_limit, dtype=np.float64)
    if any(value.shape != start.shape for value in (q_low, q_high, vmax, amax)) or np.any(q_low >= q_high):
        raise ValueError("workspace/reference vectors must have equal, ordered shapes")
    start = np.clip(start, q_low, q_high)
    n = start.size
    span = q_high - q_low
    t = np.arange(steps, dtype=np.float64)[:, None] * dt

    if mode == "hold":
        raw = np.repeat(rng.uniform(q_low, q_high)[None, :], steps, axis=0)
    elif mode == "step":
        raw = np.empty((steps, n), dtype=np.float64)
        block = max(1, steps // 4)
        for begin in range(0, steps, block):
            raw[begin:begin + block] = rng.uniform(q_low, q_high)
    elif mode == "smooth_random":
        knots_count = 6
        knot_steps = np.linspace(0, steps - 1, knots_count, dtype=np.int64)
        knots = np.vstack((start, rng.uniform(q_low, q_high, size=(knots_count - 1, n))))
        raw = np.empty((steps, n), dtype=np.float64)
        for joint in range(n):
            raw[:, joint] = np.interp(np.arange(steps), knot_steps, knots[:, joint])
    elif mode == "sine":
        # 20% of the available workspace at <=0.04 Hz is deliberately below
        # 70% of the deployed 0.25 rad/s command limit for this arm.
        amplitude = 0.20 * span
        base = rng.uniform(q_low + amplitude, q_high - amplitude)
        frequency = rng.uniform(0.020, 0.040, size=n)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n)
        raw = base + amplitude * np.sin(2.0 * np.pi * frequency[None, :] * t + phase[None, :])
        if single_joint_index is not None:
            if not 0 <= single_joint_index < n:
                raise ValueError("single_joint_index is outside the controlled joint range")
            frozen = np.repeat(start[None, :], steps, axis=0)
            frozen[:, single_joint_index] = raw[:, single_joint_index]
            raw = frozen
    else:  # delta_ref_random
        # Slowly varying reference increments, then the same physical limiter
        # used by every other mode.
        increments = rng.normal(0.0, 0.018 * span, size=(steps, n))
        increments = np.clip(increments, -0.05 * span, 0.05 * span)
        raw = np.clip(start + np.cumsum(increments, axis=0), q_low, q_high)

    raw = np.clip(raw, q_low, q_high)
    return _rate_and_acceleration_limit(raw, start, dt, vmax, amax, q_low, q_high)
