from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np


class SafetyMode(str, Enum):
    RUNNING = "running"
    HOLDING = "holding"
    FAULT_LATCHED = "fault_latched"
    TORQUE_DISABLED = "torque_disabled"


@dataclass(frozen=True)
class ThermalDecision:
    mode: SafetyMode
    residual_allowed: bool
    authority_scale: float
    request_torque_disable: bool
    hottest_motor_index: int
    reason: str = ""
    filtered_temperature: np.ndarray | None = None
    filter_status: tuple[str, ...] = ()


class ThermalSupervisor:
    """Four-sample averaged thermal policy with stale-data escalation.

    A temperature sample is an independently refreshed hardware diagnostic,
    not a 30 Hz reuse of the same cached value.  Thresholds are evaluated on
    the per-motor mean of the latest four independent samples, which avoids a
    single implausible register value terminating an otherwise healthy run.
    """

    def __init__(self):
        self._fault_started_ns: int | None = None
        self._fault_temperature: float | None = None
        self._critical_temperature_sample_ns: int | None = None
        self._temperature_samples: list[np.ndarray] = []
        self._last_temperature_sample_ns: int | None = None
        self._last_filtered_temperature: np.ndarray | None = None
        self._last_filter_status: tuple[str, ...] = ()
        self._sample_unconfirmed = False

    def _record_temperature_sample(self, values: np.ndarray) -> np.ndarray:
        """Append a confirmed independent sample and return its four-sample mean."""
        self._temperature_samples.append(values.astype(np.float64, copy=True))
        self._temperature_samples = self._temperature_samples[-4:]
        return np.mean(np.stack(self._temperature_samples, axis=0), axis=0)

    def _confirm_or_filter_sample(self, values: np.ndarray,
                                  confirmation_reader: Callable[[], list[np.ndarray]] | None) -> tuple[np.ndarray | None, tuple[str, ...]]:
        """Filter implausible jumps using two independent temperature rereads.

        A new value more than 8 C from the per-motor median of accepted
        samples (or any first value >=50 C) is a candidate.  It enters the
        safety average only when at least one of two immediate rereads agrees
        within 3 C.  If neither agrees, their median replaces the candidate
        and the outlier is recorded as rejected.  A reread failure produces no
        new accepted sample and keeps the arm in HOLDING until the next normal
        2 Hz diagnostic update.
        """
        if self._temperature_samples:
            baseline = np.median(np.stack(self._temperature_samples, axis=0), axis=0)
            candidate = np.abs(values - baseline) > 8.0
        else:
            candidate = values >= 50.0
        status = np.full(values.shape, "accepted", dtype="<U32")
        if not np.any(candidate):
            return values, tuple(status.tolist())
        if confirmation_reader is None:
            status[candidate] = "confirmation_unavailable"
            return None, tuple(status.tolist())
        try:
            rereads = [np.asarray(sample, dtype=np.float64) for sample in confirmation_reader()]
        except Exception:
            rereads = []
        if len(rereads) != 2 or any(sample.shape != values.shape or not np.all(np.isfinite(sample)) for sample in rereads):
            status[candidate] = "confirmation_failed"
            return None, tuple(status.tolist())
        confirmation = np.stack(rereads, axis=0)
        selected = values.copy()
        for index in np.flatnonzero(candidate):
            agreeing = confirmation[np.abs(confirmation[:, index] - values[index]) <= 3.0, index]
            if agreeing.size:
                selected[index] = float(np.median(np.concatenate(([values[index]], agreeing))))
                status[index] = "confirmed_jump"
            else:
                selected[index] = float(np.median(confirmation[:, index]))
                status[index] = "outlier_rejected"
        return selected, tuple(status.tolist())

    def evaluate(self, temperatures: np.ndarray | None, sample_age_s: float, now_ns: int,
                 current_raw: np.ndarray | None = None,
                 diagnostic_update_timestamp_ns: int | None = None,
                 confirmation_reader: Callable[[], list[np.ndarray]] | None = None) -> ThermalDecision:
        if temperatures is None or sample_age_s > 3.0:
            return ThermalDecision(SafetyMode.FAULT_LATCHED, False, 0.0, False, -1, "temperature_stale")
        values = np.asarray(temperatures, dtype=np.float64)
        if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
            return ThermalDecision(SafetyMode.FAULT_LATCHED, False, 0.0, False, -1, "temperature_invalid")
        # The backend passes a stable timestamp until the next 2 Hz diagnostic
        # refresh.  Unit callers without one treat each invocation as a new
        # independent sample.
        sample_ns = int(diagnostic_update_timestamp_ns if diagnostic_update_timestamp_ns is not None else now_ns)
        if self._last_temperature_sample_ns != sample_ns:
            self._last_temperature_sample_ns = sample_ns
            selected, status = self._confirm_or_filter_sample(values, confirmation_reader)
            self._last_filter_status = status
            self._sample_unconfirmed = selected is None
            if selected is not None:
                self._last_filtered_temperature = self._record_temperature_sample(selected)
        if self._sample_unconfirmed or self._last_filtered_temperature is None:
            hottest_raw = int(np.argmax(values))
            return ThermalDecision(SafetyMode.HOLDING, False, 0.0, False, hottest_raw,
                                   "temperature_confirmation_pending", None, self._last_filter_status)
        averaged = self._last_filtered_temperature
        hottest = int(np.argmax(averaged)); maximum = float(averaged[hottest])

        # A normal start need not wait two seconds.  If an unusually hot value
        # appears before four samples exist, hold the last safe goal until the
        # four-sample average is available; do not fault from a single sample.
        if len(self._temperature_samples) < 4:
            provisional_peak = np.max(np.stack(self._temperature_samples, axis=0), axis=0)
            provisional_hottest = int(np.argmax(provisional_peak))
            if float(provisional_peak[provisional_hottest]) >= 50.0:
                return ThermalDecision(SafetyMode.HOLDING, False, 0.0, False, provisional_hottest,
                                       "temperature_collecting_four_sample_average", averaged, self._last_filter_status)
            return ThermalDecision(SafetyMode.RUNNING, True, 1.0, False, hottest, "", averaged,
                                   self._last_filter_status)
        if maximum >= 60.0:
            # Temperature diagnostics are refreshed much more slowly than the
            # 30 Hz control loop.  A single corrupted register read must not
            # immediately drop a gravity-loaded arm, but it must immediately
            # hold the last safe goal.  Disable torque only after a second,
            # independently refreshed critical sample confirms >=60 C.
            if self._critical_temperature_sample_ns is None:
                self._critical_temperature_sample_ns = sample_ns
                return ThermalDecision(SafetyMode.HOLDING, False, 0.0, False, hottest,
                                       "temperature_60c_pending_confirmation", averaged, self._last_filter_status)
            if sample_ns == self._critical_temperature_sample_ns:
                return ThermalDecision(SafetyMode.HOLDING, False, 0.0, False, hottest,
                                       "temperature_60c_pending_confirmation", averaged, self._last_filter_status)
            return ThermalDecision(SafetyMode.FAULT_LATCHED, False, 0.0, True, hottest,
                                   "temperature_60c_confirmed", averaged, self._last_filter_status)
        self._critical_temperature_sample_ns = None
        if maximum >= 55.0:
            if self._fault_started_ns is None:
                self._fault_started_ns, self._fault_temperature = int(now_ns), maximum
            elapsed = (int(now_ns) - self._fault_started_ns) * 1e-9
            rising = self._fault_temperature is not None and maximum > self._fault_temperature + 0.2
            sustained_current = current_raw is not None and np.max(np.abs(current_raw)) > 400
            disable = elapsed >= 5.0 and (rising or sustained_current or maximum >= 59.0)
            return ThermalDecision(SafetyMode.FAULT_LATCHED, False, 0.0, disable, hottest, "temperature_55c",
                                   averaged, self._last_filter_status)
        self._fault_started_ns = self._fault_temperature = None
        if maximum >= 50.0 or sample_age_s > 1.0:
            return ThermalDecision(SafetyMode.RUNNING, False, 0.5, False, hottest,
                                   "temperature_warning" if maximum >= 50.0 else "temperature_old",
                                   averaged, self._last_filter_status)
        return ThermalDecision(SafetyMode.RUNNING, True, 1.0, False, hottest, "", averaged,
                               self._last_filter_status)


@dataclass(frozen=True)
class VoltageDecision:
    mode: SafetyMode
    residual_allowed: bool
    reason: str = ""


def evaluate_voltage(voltage_v: np.ndarray | None, *, warning_low: float | None, warning_high: float | None,
                     hard_low: float | None, hard_high: float | None) -> VoltageDecision:
    """Apply thresholds frozen from PSU/EEPROM/direct characterization; null means monitor-only."""
    if voltage_v is None or hard_low is None:
        return VoltageDecision(SafetyMode.RUNNING, False if voltage_v is None else True, "voltage_unconfigured")
    values = np.asarray(voltage_v, dtype=np.float64)
    assert warning_low is not None and warning_high is not None and hard_high is not None
    if np.any(values <= hard_low) or np.any(values >= hard_high):
        return VoltageDecision(SafetyMode.FAULT_LATCHED, False, "voltage_hard_limit")
    if np.any(values <= warning_low) or np.any(values >= warning_high):
        return VoltageDecision(SafetyMode.RUNNING, False, "voltage_warning")
    return VoltageDecision(SafetyMode.RUNNING, True)


@dataclass(frozen=True)
class ProjectionResult:
    q_ref: np.ndarray
    velocity: np.ndarray
    flags: tuple[str, ...]


class CommandProjector:
    def __init__(self, joint_low: np.ndarray, joint_high: np.ndarray, velocity_limit: np.ndarray,
                 acceleration_limit: np.ndarray, relative_limit: np.ndarray, control_dt: float):
        vectors = [np.asarray(v, dtype=np.float64) for v in (joint_low, joint_high, velocity_limit,
                                                              acceleration_limit, relative_limit)]
        if any(v.shape != vectors[0].shape for v in vectors) or np.any(vectors[0] >= vectors[1]):
            raise ValueError("invalid projector limits")
        self.low, self.high, self.vmax, self.amax, self.relative = vectors
        self.dt = float(control_dt)
        self.previous_velocity = np.zeros_like(self.low)

    def reset(self) -> None:
        self.previous_velocity.fill(0.0)

    def project(self, requested: np.ndarray, measured_q: np.ndarray, previous_command: np.ndarray) -> ProjectionResult:
        target = np.asarray(requested, dtype=np.float64).copy()
        measured = np.asarray(measured_q, dtype=np.float64)
        previous = np.asarray(previous_command, dtype=np.float64)
        flags: list[str] = []
        clipped = np.clip(target, self.low, self.high)
        if not np.array_equal(clipped, target): flags.append("joint_limit")
        target = clipped
        relative = np.clip(target, measured - self.relative, measured + self.relative)
        if not np.array_equal(relative, target): flags.append("relative_limit")
        target = relative
        requested_v = (target - previous) / self.dt
        velocity = np.clip(requested_v, -self.vmax, self.vmax)
        if not np.array_equal(velocity, requested_v): flags.append("velocity_limit")
        low_v = self.previous_velocity - self.amax * self.dt
        high_v = self.previous_velocity + self.amax * self.dt
        accelerated = np.clip(velocity, low_v, high_v)
        if not np.array_equal(accelerated, velocity): flags.append("acceleration_limit")
        target = np.clip(previous + accelerated * self.dt, self.low, self.high)
        self.previous_velocity = accelerated
        return ProjectionResult(target.astype(np.float32), accelerated.astype(np.float32), tuple(flags))


@dataclass
class TimingDecision:
    training_valid: bool
    use_in_history: bool
    reset_history: bool
    hold: bool
    reason: str = ""


class TimingSupervisor:
    def __init__(self, control_dt: float):
        self.control_dt = float(control_dt)
        self._moderate_count = 0

    def evaluate(self, actual_dt: float, *, read_ok: bool = True, timestamps_monotonic: bool = True,
                 skipped_ticks: int = 0, wake_lateness: float = 0.0) -> TimingDecision:
        ratio = abs(float(actual_dt) - self.control_dt) / self.control_dt
        severe = (not read_ok or not timestamps_monotonic or skipped_ticks > 0 or
                  actual_dt > 1.5 * self.control_dt or wake_lateness > 0.5 * self.control_dt)
        if severe:
            self._moderate_count = 0
            return TimingDecision(False, False, True, True, "severe_timing_fault")
        if ratio > 0.25:
            self._moderate_count = 0
            return TimingDecision(False, False, True, True, "timing_deviation_gt_25pct")
        if ratio > 0.10:
            self._moderate_count += 1
            reset = self._moderate_count >= 2
            return TimingDecision(False, False, reset, True, "consecutive_jitter" if reset else "single_jitter")
        self._moderate_count = 0
        return TimingDecision(True, True, False, False)
