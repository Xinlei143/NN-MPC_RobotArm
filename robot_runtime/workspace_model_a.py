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
    # Real-only fast-excitation modes (2026-08-08).  These break the
    # position-control "command is redundant" trap by saturating the 0.25 rad/s
    # command limiter: the servo spends most of its time in pursuit/brake
    # transients, giving the dynamics network a strong (du, d-dq) signal.
    "fast_walk": 5,
    "fast_sine": 6,
    "step_hold": 7,
}

# The real-data target is intentionally smaller than the simulation corpus but
# large enough to train and independently evaluate a workspace dynamics model.
# Keep this plan in one module so the collector, offline preflight, manifests,
# and documentation cannot silently disagree.
#
# v2 (2026-08-08): the original 38/5/5 split is frozen; sessions 48..57 are a
# fast-excitation extension block, all of them train.  The append store allows
# exactly this one extension shape (see _collection_plan_compatible).
MODEL_A_PLAN_VERSION = "so101_model_a_58x15min_v2"
MODEL_A_SESSION_SECONDS = 15.0 * 60.0
MODEL_A_TRAIN_SESSIONS = 48
MODEL_A_VALIDATION_SESSIONS = 5
MODEL_A_TEST_SESSIONS = 5
MODEL_A_TOTAL_SESSIONS = (
    MODEL_A_TRAIN_SESSIONS + MODEL_A_VALIDATION_SESSIONS + MODEL_A_TEST_SESSIONS
)
# Sessions 0..37 were the original train block; 38..42 validation, 43..47 test.
MODEL_A_ORIGINAL_TRAIN_SESSIONS = 38
MODEL_A_EXTENSION_FIRST_INDEX = MODEL_A_ORIGINAL_TRAIN_SESSIONS + 10
_BASE_PROGRAM_SECONDS = 5.0 * 60.0


def expected_model_a_split(session_index: int) -> str:
    """Return the fixed session-level 38/5/5 split plus the extension block.

    Sessions 0..47 keep the original 38/5/5 split; 48..57 are extension
    sessions and always belong to train.  The extension block never enters
    validation or test, so the model gate keeps its original evaluation
    distribution.
    """
    if not 0 <= int(session_index) < MODEL_A_TOTAL_SESSIONS:
        raise ValueError(f"session_index must be in [0, {MODEL_A_TOTAL_SESSIONS - 1}]")
    if session_index < MODEL_A_EXTENSION_FIRST_INDEX:
        if session_index < MODEL_A_ORIGINAL_TRAIN_SESSIONS:
            return "train"
        if session_index < MODEL_A_ORIGINAL_TRAIN_SESSIONS + MODEL_A_VALIDATION_SESSIONS:
            return "validation"
        return "test"
    return "train"


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
    """Return the fixed, scaled Model-A schedule for one real session.

    Extension sessions (48..57) use the fast-excitation program: pure
    command-dynamics excitation with short hold brackets, repeated once with a
    different seed-drawn excitation.  The segment seconds sum exactly to
    ``session_seconds`` so the 15-minute session length is unchanged.
    """
    expected_model_a_split(session_index)
    if not np.isfinite(session_seconds) or session_seconds <= 0:
        raise ValueError("session_seconds must be positive and finite")
    if session_index >= MODEL_A_EXTENSION_FIRST_INDEX:
        program = (
            WorkspaceProgram("hold", "hold", 15.0),
            WorkspaceProgram("fast_walk", "fast_walk", 180.0),
            WorkspaceProgram("fast_sine", "fast_sine", 180.0),
            WorkspaceProgram("step_hold", "step_hold", 150.0),
            WorkspaceProgram("fast_walk", "fast_walk", 180.0),
            WorkspaceProgram("fast_sine", "fast_sine", 180.0),
            WorkspaceProgram("hold", "hold", 15.0),
        )
        if not np.isclose(sum(item.seconds for item in program), float(session_seconds)):
            raise ValueError("fast-excitation program must sum to session_seconds")
        return program
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
    fresh_center_for_sine: bool = False,
    center: np.ndarray | None = None,
) -> np.ndarray:
    """Build one bounded, rate-limited absolute-position reference sequence.

    ``center`` anchors the excitation windows (fast modes) and the hold
    posture when given; extension sessions pass ``config.home_q_ctrl`` so the
    fast-excitation program stays around home.  Without it, hold samples the
    full workspace and fast modes use the segment start as their window
    center, matching the original corpus behavior.
    """
    if mode not in WORKSPACE_MODE_IDS:
        raise ValueError(f"unsupported workspace Model-A mode: {mode!r}")
    if steps <= 0 or dt <= 0:
        raise ValueError("steps and dt must be positive")
    start = np.asarray(start_q, dtype=np.float64)
    q_low, q_high = np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    vmax, amax = np.asarray(velocity_limit, dtype=np.float64), np.asarray(acceleration_limit, dtype=np.float64)
    if any(value.shape != start.shape for value in (q_low, q_high, vmax, amax)) or np.any(q_low >= q_high):
        raise ValueError("workspace/reference vectors must have equal, ordered shapes")
    anchor = np.asarray(center, dtype=np.float64) if center is not None else start
    if anchor.shape != start.shape or np.any(anchor < q_low) or np.any(anchor > q_high):
        raise ValueError("excitation center must be inside the workspace")
    start = np.clip(start, q_low, q_high)
    n = start.size
    span = q_high - q_low
    t = np.arange(steps, dtype=np.float64)[:, None] * dt

    if mode == "hold":
        if center is None:
            raw = np.repeat(rng.uniform(q_low, q_high)[None, :], steps, axis=0)
        else:
            # Extension sessions hold near the excitation center, keeping the
            # stationary samples inside the home envelope too.
            hold_q = anchor + rng.uniform(-np.deg2rad(1.0), np.deg2rad(1.0), size=n)
            raw = np.repeat(hold_q[None, :], steps, axis=0)
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
            # By default the other joints are frozen at ``start_q`` (the
            # previous segment's endpoint).  Some endpoints sit so low that no
            # +/-amplitude sweep of the target joint can stay above the table;
            # callers then retry with a fresh random center so the frozen
            # posture and the swept arc are rejected/co-sampled.  The target
            # joint always receives the sine; only the frozen posture changes.
            if fresh_center_for_sine:
                frozen = np.repeat(rng.uniform(q_low, q_high)[None, :], steps, axis=0)
            else:
                frozen = np.repeat(start[None, :], steps, axis=0)
            frozen[:, single_joint_index] = raw[:, single_joint_index]
            raw = frozen
    elif mode == "delta_ref_random":
        # Slowly varying reference increments, then the same physical limiter
        # used by every other mode.
        increments = rng.normal(0.0, 0.018 * span, size=(steps, n))
        increments = np.clip(increments, -0.05 * span, 0.05 * span)
        raw = np.clip(start + np.cumsum(increments, axis=0), q_low, q_high)
    elif mode == "fast_walk":
        # Command-dynamics excitation: single-joint steps of 0.4-1.2 deg every
        # 12-36 ticks, accumulated into a random walk bounded to +/-5 deg of
        # the starting posture.  Steps above the 0.25 rad/s command limiter
        # force the servo into pursuit transients, giving the network a strong
        # (du, d-dq) correlation to learn from -- the original position-control
        # corpus has nearly none.
        window = np.full(n, np.deg2rad(5.0))
        raw = np.repeat(start[None, :], steps, axis=0).copy()
        pos = 0
        while pos < steps:
            end = min(pos + int(rng.integers(12, 37)), steps)
            if pos > 0:
                amplitude = rng.uniform(np.deg2rad(0.4), np.deg2rad(1.2)) * rng.choice((-1.0, 1.0))
                joint = int(rng.integers(0, n))
                previous_value = raw[pos - 1, joint]
                raw[pos:end, joint] = np.clip(previous_value + amplitude,
                                              anchor[joint] - window[joint], anchor[joint] + window[joint])
            pos = end
    elif mode == "fast_sine":
        # Above/near the servo bandwidth (~0.5 Hz): 0.3-0.8 Hz at 3-8 deg,
        # bounded to +/-8 deg of the excitation center.  The slow end is still
        # trackable; the fast end saturates the 0.25 rad/s limiter, so the
        # recorded command alternates between pursuit transients and
        # rate-limited ramps.  Note the limiter's turn-around inertia (bounded
        # by command_acceleration_limit) overshoots fast reversals by ~2 deg,
        # so the actual command envelope is the window plus the glide distance;
        # offline preflight validates the real sequence, not the window.
        amplitude = rng.uniform(np.deg2rad(3.0), np.deg2rad(8.0), size=n)
        frequency = rng.uniform(0.3, 0.8, size=n)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n)
        raw = start[None, :] + amplitude[None, :] * np.sin(2.0 * np.pi * frequency[None, :] * t + phase[None, :])
        window = np.full(n, np.deg2rad(8.0))
        raw = np.clip(raw, anchor - window, anchor + window)
    elif mode == "step_hold":
        # Well-separated 1-3 deg single-joint steps, accumulated into a random
        # walk bounded to +/-8 deg, each held 1.5-2 s so the model sees the
        # full settle transient (pursuit, overshoot, decay) to each new command
        # instead of the limiter's smooth ramps.
        window = np.full(n, np.deg2rad(8.0))
        raw = np.repeat(start[None, :], steps, axis=0).copy()
        pos = 0
        while pos < steps:
            end = min(pos + int(rng.integers(45, 61)), steps)
            if pos > 0:
                amplitude = rng.uniform(np.deg2rad(1.0), np.deg2rad(3.0)) * rng.choice((-1.0, 1.0))
                joint = int(rng.integers(0, n))
                previous_value = raw[pos - 1, joint]
                raw[pos:end, joint] = np.clip(previous_value + amplitude,
                                              anchor[joint] - window[joint], anchor[joint] + window[joint])
            pos = end
    else:
        raise ValueError(f"unsupported workspace Model-A mode: {mode!r}")

    raw = np.clip(raw, q_low, q_high)
    return _rate_and_acceleration_limit(raw, start, dt, vmax, amax, q_low, q_high)
