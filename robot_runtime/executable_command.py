"""Canonical SO101 executable-command projection.

The real controller has three consumers of command kinematics: the learned
rollout, the ASAP forecast, and the motor backend.  This module contains the
shared *state transition* used by all three.  The NumPy implementation is the
reference implementation for the hardware thread; the Torch implementation is
its fixed-shape, batched inference counterpart.

The state deliberately stores the command that was actually transmitted after
encoder quantisation.  A failed bus write must not advance this state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


ENCODER_MAX = 4095.0


@dataclass(frozen=True)
class ExecutableCommandSpec:
    joint_low: np.ndarray
    joint_high: np.ndarray
    velocity_limit: np.ndarray
    acceleration_limit: np.ndarray
    relative_limit: np.ndarray
    raw_low: np.ndarray
    raw_high: np.ndarray
    calibration_low: np.ndarray
    calibration_high: np.ndarray
    control_dt: float
    braking: bool = True

    def __post_init__(self) -> None:
        vectors = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in (
                ("joint_low", self.joint_low), ("joint_high", self.joint_high),
                ("velocity_limit", self.velocity_limit),
                ("acceleration_limit", self.acceleration_limit),
                ("relative_limit", self.relative_limit), ("raw_low", self.raw_low),
                ("raw_high", self.raw_high),
                ("calibration_low", self.calibration_low),
                ("calibration_high", self.calibration_high),
            )
        }
        shapes = {value.shape for value in vectors.values()}
        if len(shapes) != 1:
            raise ValueError(f"all executable-command vectors must share one shape, got {shapes}")
        shape = next(iter(shapes))
        if len(shape) != 1 or shape[0] == 0:
            raise ValueError("executable-command vectors must be non-empty 1-D arrays")
        if np.any(vectors["joint_low"] >= vectors["joint_high"]):
            raise ValueError("joint_low must be strictly below joint_high")
        for name in ("velocity_limit", "acceleration_limit", "relative_limit"):
            if not np.all(np.isfinite(vectors[name])) or np.any(vectors[name] <= 0):
                raise ValueError(f"{name} must contain finite positive values")
        for name in ("raw_low", "raw_high", "calibration_low", "calibration_high"):
            if not np.all(np.isfinite(vectors[name])):
                raise ValueError(f"{name} must be finite")
        if np.any(vectors["raw_low"] >= vectors["raw_high"]):
            raise ValueError("raw_low must be strictly below raw_high")
        if np.any(vectors["calibration_low"] >= vectors["calibration_high"]):
            raise ValueError("calibration_low must be strictly below calibration_high")
        if not np.isfinite(self.control_dt) or self.control_dt <= 0:
            raise ValueError("control_dt must be positive and finite")

        # Freeze copies so a worker cannot observe a config mutation while a
        # plan is being scored.
        for name, value in vectors.items():
            object.__setattr__(self, name, value.copy())
        object.__setattr__(self, "control_dt", float(self.control_dt))

    @property
    def n_joints(self) -> int:
        return int(self.joint_low.shape[0])

    def torch_vectors(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
    ) -> dict[str, torch.Tensor]:
        """Return cached-shape tensors for batched projection.

        The exact/raw-count path uses float64 to match NumPy.  CEM candidate
        ranking may request float32; its selected candidate is re-evaluated
        through the exact path before publication.
        """
        if dtype not in (torch.float32, torch.float64):
            raise ValueError(f"projection dtype must be float32 or float64, got {dtype}")
        return {
            name: torch.as_tensor(value, dtype=dtype, device=device)
            for name, value in (
                ("joint_low", self.joint_low), ("joint_high", self.joint_high),
                ("velocity_limit", self.velocity_limit),
                ("acceleration_limit", self.acceleration_limit),
                ("relative_limit", self.relative_limit), ("raw_low", self.raw_low),
                ("raw_high", self.raw_high),
                ("calibration_low", self.calibration_low),
                ("calibration_high", self.calibration_high),
            )
        }


@dataclass(frozen=True)
class ExecutableCommandState:
    previous_transmitted_q_ref: np.ndarray
    previous_command_velocity: np.ndarray

    def __post_init__(self) -> None:
        previous = np.asarray(self.previous_transmitted_q_ref, dtype=np.float64)
        velocity = np.asarray(self.previous_command_velocity, dtype=np.float64)
        if previous.ndim != 1 or velocity.shape != previous.shape:
            raise ValueError("command state vectors must be matching 1-D arrays")
        object.__setattr__(self, "previous_transmitted_q_ref", previous.copy())
        object.__setattr__(self, "previous_command_velocity", velocity.copy())

    @classmethod
    def anchored(cls, q_ref: np.ndarray) -> "ExecutableCommandState":
        q = np.asarray(q_ref, dtype=np.float64)
        return cls(q, np.zeros_like(q))


@dataclass(frozen=True)
class ExecutableCommandResult:
    requested_q_ref: np.ndarray
    projected_q_ref: np.ndarray
    transmitted_q_ref: np.ndarray
    tx_goal_position_raw: np.ndarray
    command_velocity: np.ndarray
    projection_flags: tuple[str, ...]
    next_state: ExecutableCommandState


def make_executable_command_spec(
    *,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
    velocity_limit: np.ndarray,
    acceleration_limit: np.ndarray,
    relative_limit: np.ndarray,
    raw_low: np.ndarray,
    raw_high: np.ndarray,
    calibration_low: np.ndarray,
    calibration_high: np.ndarray,
    control_dt: float,
    braking: bool = True,
) -> ExecutableCommandSpec:
    return ExecutableCommandSpec(
        joint_low=np.asarray(joint_low, dtype=np.float64),
        joint_high=np.asarray(joint_high, dtype=np.float64),
        velocity_limit=np.asarray(velocity_limit, dtype=np.float64),
        acceleration_limit=np.asarray(acceleration_limit, dtype=np.float64),
        relative_limit=np.asarray(relative_limit, dtype=np.float64),
        raw_low=np.asarray(raw_low, dtype=np.float64),
        raw_high=np.asarray(raw_high, dtype=np.float64),
        calibration_low=np.asarray(calibration_low, dtype=np.float64),
        calibration_high=np.asarray(calibration_high, dtype=np.float64),
        control_dt=float(control_dt),
        braking=bool(braking),
    )


def _degrees_to_raw_np(degrees: np.ndarray, spec: ExecutableCommandSpec) -> np.ndarray:
    value = np.asarray(degrees, dtype=np.float64)
    raw_float = value * ENCODER_MAX / 360.0 + (
        spec.calibration_low + spec.calibration_high
    ) / 2.0
    # Quantise to the nearest encoder count.  Truncation toward zero creates a
    # directional bias for negative commands; when a continuous projected
    # velocity is just below a count boundary that bias can hold the actual
    # transmitted velocity forever and accumulate multi-degree drift.
    raw = np.rint(raw_float).astype(np.int64)
    return np.clip(raw, spec.raw_low.astype(np.int64), spec.raw_high.astype(np.int64))


def _raw_to_q_np(raw: np.ndarray, spec: ExecutableCommandSpec) -> np.ndarray:
    degrees = (
        np.asarray(raw, dtype=np.float64)
        - (spec.calibration_low + spec.calibration_high) / 2.0
    ) * 360.0 / ENCODER_MAX
    return np.deg2rad(degrees)


def _raw_interval_for_velocity_np(
    previous_q: np.ndarray,
    velocity_low: np.ndarray,
    velocity_high: np.ndarray,
    spec: ExecutableCommandSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw counts whose *transmitted* velocity obeys the limits.

    A count is about 0.046 rad/s at 30 Hz, larger than one SO101 tick's
    acceleration allowance.  Clipping only the continuous velocity and then
    quantising can therefore violate the acceleration bound by one count.  We
    quantise inside the admissible velocity interval instead.
    """
    lower_q = np.clip(previous_q + velocity_low * spec.control_dt, spec.joint_low, spec.joint_high)
    upper_q = np.clip(previous_q + velocity_high * spec.control_dt, spec.joint_low, spec.joint_high)
    midpoint = (spec.calibration_low + spec.calibration_high) / 2.0
    scale = ENCODER_MAX / 360.0
    lower_raw = np.ceil(np.rad2deg(lower_q) * scale + midpoint).astype(np.int64)
    upper_raw = np.floor(np.rad2deg(upper_q) * scale + midpoint).astype(np.int64)
    # At 30 Hz one count is slightly larger than a_max*dt.  Expand by one
    # count so a stationary command is not deadlocked forever at zero; the
    # resulting one-count acceleration excess is explicitly accounted for by
    # the transmitted state on the next tick.
    return (
        np.maximum(lower_raw - 1, spec.raw_low.astype(np.int64)),
        np.minimum(upper_raw + 1, spec.raw_high.astype(np.int64)),
    )


def _step_arrays_np(
    requested: np.ndarray,
    measured_q: np.ndarray,
    previous_q_ref: np.ndarray,
    previous_velocity: np.ndarray,
    spec: ExecutableCommandSpec,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray]:
    requested = np.asarray(requested, dtype=np.float64)
    measured = np.asarray(measured_q, dtype=np.float64)
    previous = np.asarray(previous_q_ref, dtype=np.float64)
    velocity_previous = np.asarray(previous_velocity, dtype=np.float64)
    if any(value.shape != (spec.n_joints,) for value in (requested, measured, previous, velocity_previous)):
        raise ValueError(f"command vectors must all have shape ({spec.n_joints},)")
    if not np.all(np.isfinite(requested)):
        raise ValueError("requested command contains non-finite values")
    flags: list[str] = []
    target = np.clip(requested, spec.joint_low, spec.joint_high)
    if not np.array_equal(target, requested):
        flags.append("joint_limit")
    relative = np.clip(target, measured - spec.relative_limit, measured + spec.relative_limit)
    if not np.array_equal(relative, target):
        flags.append("relative_limit")
    requested_velocity = (relative - previous) / spec.control_dt
    velocity = np.clip(requested_velocity, -spec.velocity_limit, spec.velocity_limit)
    if not np.array_equal(velocity, requested_velocity):
        flags.append("velocity_limit")
    accelerated = np.clip(
        velocity,
        velocity_previous - spec.acceleration_limit * spec.control_dt,
        velocity_previous + spec.acceleration_limit * spec.control_dt,
    )
    if not np.array_equal(accelerated, velocity):
        flags.append("acceleration_limit")
    if spec.braking:
        distance_to_high = np.maximum(spec.joint_high - previous, 0.0)
        distance_to_low = np.maximum(previous - spec.joint_low, 0.0)
        positive = np.sqrt(
            np.square(spec.acceleration_limit * spec.control_dt)
            + 2.0 * spec.acceleration_limit * distance_to_high
        ) - spec.acceleration_limit * spec.control_dt
        negative = np.sqrt(
            np.square(spec.acceleration_limit * spec.control_dt)
            + 2.0 * spec.acceleration_limit * distance_to_low
        ) - spec.acceleration_limit * spec.control_dt
        braked = np.minimum(accelerated, np.maximum(positive, 0.0))
        braked = np.maximum(braked, -np.maximum(negative, 0.0))
        if not np.array_equal(braked, accelerated):
            flags.append("braking_limit")
        accelerated = braked
    projected = np.clip(previous + accelerated * spec.control_dt, spec.joint_low, spec.joint_high)
    raw = _degrees_to_raw_np(np.rad2deg(projected), spec)
    # Enforce the limits again in the discrete transmitted domain.  Without
    # this step a two-count quantisation jump can exceed ``a_max * dt`` even
    # though the pre-quantisation velocity was valid.
    velocity_low = np.maximum(-spec.velocity_limit, velocity_previous - spec.acceleration_limit * spec.control_dt)
    velocity_high = np.minimum(spec.velocity_limit, velocity_previous + spec.acceleration_limit * spec.control_dt)
    if spec.braking:
        velocity_low = np.maximum(velocity_low, -np.maximum(negative, 0.0))
        velocity_high = np.minimum(velocity_high, np.maximum(positive, 0.0))
    raw_allowed_low, raw_allowed_high = _raw_interval_for_velocity_np(
        previous, velocity_low, velocity_high, spec
    )
    raw_previous = _degrees_to_raw_np(np.rad2deg(previous), spec)
    raw_previous = np.clip(raw_previous, spec.raw_low.astype(np.int64), spec.raw_high.astype(np.int64))
    valid_interval = raw_allowed_low <= raw_allowed_high
    quantized = np.clip(raw, raw_allowed_low, raw_allowed_high)
    raw = np.where(valid_interval, quantized, raw_previous)
    if not np.array_equal(raw, _degrees_to_raw_np(np.rad2deg(projected), spec)):
        flags.append("quantization_acceleration_hold")
    transmitted = _raw_to_q_np(raw, spec)
    if not np.allclose(transmitted, projected, atol=0.0, rtol=0.0):
        flags.append("encoder_quantization")
    # The next state is always derived from what can actually be sent, not
    # from the pre-quantisation velocity.
    command_velocity = (transmitted - previous) / spec.control_dt
    return projected.astype(np.float32), raw.astype(np.int64), tuple(flags), command_velocity.astype(np.float32)


def step_executable_command_np(
    requested_q_ref: np.ndarray,
    measured_q: np.ndarray,
    state: ExecutableCommandState,
    spec: ExecutableCommandSpec,
) -> ExecutableCommandResult:
    projected, raw, flags, command_velocity = _step_arrays_np(
        requested_q_ref, measured_q, state.previous_transmitted_q_ref,
        state.previous_command_velocity, spec
    )
    transmitted = _raw_to_q_np(raw, spec).astype(np.float32)
    next_state = ExecutableCommandState(transmitted, command_velocity)
    return ExecutableCommandResult(
        requested_q_ref=np.asarray(requested_q_ref, dtype=np.float32).copy(),
        projected_q_ref=projected,
        transmitted_q_ref=transmitted,
        tx_goal_position_raw=raw,
        command_velocity=command_velocity,
        projection_flags=flags,
        next_state=next_state,
    )


def step_executable_command_torch(
    requested_q_ref: torch.Tensor,
    measured_q: torch.Tensor,
    previous_q_ref: torch.Tensor,
    previous_velocity: torch.Tensor,
    spec: ExecutableCommandSpec,
    *,
    exact: bool = True,
    vectors: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched Torch counterpart returning projected, raw, transmitted, velocity.

    ``exact=True`` is the raw-count parity path.  CEM scoring can use the
    float32 path (with precomputed ``vectors``) and then re-evaluate the
    selected action exactly before it is sent to hardware.
    """
    if requested_q_ref.ndim != 2:
        raise ValueError("requested_q_ref must have shape [batch, joints]")
    if measured_q.shape != requested_q_ref.shape:
        raise ValueError("measured_q must match requested_q_ref shape")
    device = requested_q_ref.device
    dtype = torch.float64 if exact else torch.float32
    if vectors is None:
        vectors = spec.torch_vectors(device=device, dtype=dtype)
    elif any(value.device != device or value.dtype != dtype for value in vectors.values()):
        raise ValueError("projection vectors do not match requested device/dtype")
    requested = requested_q_ref.to(dtype)
    measured = measured_q.to(dtype)
    previous = previous_q_ref.to(dtype)
    previous_velocity = previous_velocity.to(dtype)
    target = torch.clamp(requested, vectors["joint_low"], vectors["joint_high"])
    target = torch.clamp(target, measured - vectors["relative_limit"], measured + vectors["relative_limit"])
    requested_velocity = (target - previous) / spec.control_dt
    velocity = torch.clamp(requested_velocity, -vectors["velocity_limit"], vectors["velocity_limit"])
    velocity = torch.clamp(
        velocity,
        previous_velocity - vectors["acceleration_limit"] * spec.control_dt,
        previous_velocity + vectors["acceleration_limit"] * spec.control_dt,
    )
    if spec.braking:
        distance_to_high = torch.clamp(vectors["joint_high"] - previous, min=0.0)
        distance_to_low = torch.clamp(previous - vectors["joint_low"], min=0.0)
        positive = torch.sqrt(
            torch.square(vectors["acceleration_limit"] * spec.control_dt)
            + 2.0 * vectors["acceleration_limit"] * distance_to_high
        ) - vectors["acceleration_limit"] * spec.control_dt
        negative = torch.sqrt(
            torch.square(vectors["acceleration_limit"] * spec.control_dt)
            + 2.0 * vectors["acceleration_limit"] * distance_to_low
        ) - vectors["acceleration_limit"] * spec.control_dt
        velocity = torch.minimum(velocity, torch.clamp(positive, min=0.0))
        velocity = torch.maximum(velocity, -torch.clamp(negative, min=0.0))
    projected = torch.clamp(previous + velocity * spec.control_dt, vectors["joint_low"], vectors["joint_high"])
    raw_float = torch.rad2deg(projected) * ENCODER_MAX / 360.0 + (
        vectors["calibration_low"] + vectors["calibration_high"]
    ) / 2.0
    raw = torch.round(raw_float).to(torch.int64)
    raw = torch.clamp(raw, vectors["raw_low"].to(torch.int64), vectors["raw_high"].to(torch.int64))
    if not exact:
        # Candidate scoring only: continuous velocity/acceleration/braking
        # limits have already been applied above.  The discrete admissible
        # interval below is expensive (ceil/floor plus several extra kernels)
        # and can differ by at most the encoder-count guard.  The selected
        # candidate is always re-evaluated with ``exact=True`` before packet
        # publication, so this path cannot weaken the hardware safety path.
        transmitted = torch.deg2rad(
            (raw.to(dtype) - (vectors["calibration_low"] + vectors["calibration_high"]) / 2.0)
            * 360.0 / ENCODER_MAX
        )
        command_velocity = (transmitted - previous) / spec.control_dt
        return (
            projected.to(requested_q_ref.dtype),
            raw,
            transmitted.to(requested_q_ref.dtype),
            command_velocity.to(requested_q_ref.dtype),
        )
    velocity_low = torch.maximum(
        -vectors["velocity_limit"],
        previous_velocity - vectors["acceleration_limit"] * spec.control_dt,
    )
    velocity_high = torch.minimum(
        vectors["velocity_limit"],
        previous_velocity + vectors["acceleration_limit"] * spec.control_dt,
    )
    if spec.braking:
        distance_to_high = torch.clamp(vectors["joint_high"] - previous, min=0.0)
        distance_to_low = torch.clamp(previous - vectors["joint_low"], min=0.0)
        positive = torch.sqrt(
            torch.square(vectors["acceleration_limit"] * spec.control_dt)
            + 2.0 * vectors["acceleration_limit"] * distance_to_high
        ) - vectors["acceleration_limit"] * spec.control_dt
        negative = torch.sqrt(
            torch.square(vectors["acceleration_limit"] * spec.control_dt)
            + 2.0 * vectors["acceleration_limit"] * distance_to_low
        ) - vectors["acceleration_limit"] * spec.control_dt
        velocity_low = torch.maximum(velocity_low, -torch.clamp(negative, min=0.0))
        velocity_high = torch.minimum(velocity_high, torch.clamp(positive, min=0.0))
    lower_q = torch.clamp(previous + velocity_low * spec.control_dt, vectors["joint_low"], vectors["joint_high"])
    upper_q = torch.clamp(previous + velocity_high * spec.control_dt, vectors["joint_low"], vectors["joint_high"])
    midpoint = (vectors["calibration_low"] + vectors["calibration_high"]) / 2.0
    scale = ENCODER_MAX / 360.0
    raw_allowed_low = torch.maximum(
        torch.ceil(torch.rad2deg(lower_q) * scale + midpoint).to(torch.int64) - 1,
        vectors["raw_low"].to(torch.int64),
    )
    raw_allowed_high = torch.minimum(
        torch.floor(torch.rad2deg(upper_q) * scale + midpoint).to(torch.int64) + 1,
        vectors["raw_high"].to(torch.int64),
    )
    raw_previous = torch.round(torch.rad2deg(previous) * scale + midpoint).to(torch.int64)
    raw_previous = torch.clamp(raw_previous, vectors["raw_low"].to(torch.int64), vectors["raw_high"].to(torch.int64))
    valid_interval = raw_allowed_low <= raw_allowed_high
    raw = torch.where(valid_interval, torch.clamp(raw, raw_allowed_low, raw_allowed_high), raw_previous)
    transmitted = torch.deg2rad(
        (raw.to(dtype) - (vectors["calibration_low"] + vectors["calibration_high"]) / 2.0)
        * 360.0 / ENCODER_MAX
    )
    command_velocity = (transmitted - previous) / spec.control_dt
    return (
        projected.to(requested_q_ref.dtype),
        raw,
        transmitted.to(requested_q_ref.dtype),
        command_velocity.to(requested_q_ref.dtype),
    )
