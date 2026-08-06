from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable, Protocol

import numpy as np

from robot_runtime.history import PendingActionHistory
from robot_runtime.interfaces import CommandResult, RobotBackend, RobotState
from robot_runtime.safety import SafetyMode, ThermalSupervisor, TimingSupervisor, evaluate_voltage
from robot_runtime.timing import advance_absolute_deadline, sleep_until_ns


class RealControlMode(str, Enum):
    DIRECT = "direct"
    SHADOW_MPC = "shadow_mpc"
    ACTIVE_MPC = "active_mpc"


@dataclass(frozen=True)
class PlannerCommand:
    residual: np.ndarray
    history_generation: int
    activation_tick: int
    publication_tick: int
    ood_valid: bool = True


class Planner(Protocol):
    def submit(self, tick_index: int, state_timestamp_ns: int, states: np.ndarray, commands: np.ndarray,
               history_generation: int) -> None: ...
    def latest(self) -> PlannerCommand | None: ...
    def clear(self, history_generation: int) -> None: ...


@dataclass(frozen=True)
class TickRecord:
    state: RobotState
    command: CommandResult
    nominal: np.ndarray
    mode: RealControlMode
    safety_mode: SafetyMode
    wake_lateness_s: float
    skipped_ticks: int
    planner_residual: np.ndarray
    planner_applied: bool
    planner_packet_available: bool
    planner_ood_valid: bool
    safety_guard_failure: str | None = None


class RealTimeRunner:
    def __init__(self, backend: RobotBackend, nominal: Callable[[int, RobotState], np.ndarray], *,
                 mode: RealControlMode = RealControlMode.DIRECT, planner: Planner | None = None,
                 history_len: int = 8, residual_limit_rad: float = np.deg2rad(2.0),
                 command_envelope: str = "experiment",
                 state_validator: Callable[[np.ndarray], object] | None = None,
                 command_validator: Callable[[np.ndarray], object] | None = None):
        if mode is not RealControlMode.DIRECT and planner is None:
            raise ValueError("shadow_mpc and active_mpc require a planner")
        if command_envelope not in {"experiment", "hardware"}:
            raise ValueError("command_envelope must be 'experiment' or 'hardware'")
        if command_envelope == "hardware" and mode is not RealControlMode.DIRECT:
            raise ValueError("the hardware envelope is allowed only for direct data collection")
        if command_envelope == "hardware" and not hasattr(backend, "send_hardware_joint_targets"):
            raise TypeError("backend does not provide the explicitly verified hardware command envelope")
        self.backend, self.nominal, self.mode, self.planner = backend, nominal, mode, planner
        self.command_envelope = command_envelope
        self.state_validator = state_validator
        self.command_validator = command_validator
        self.history_len = int(history_len)
        self.residual_limit = float(residual_limit_rad)
        self.timing = TimingSupervisor(backend.control_dt)
        self.thermal = ThermalSupervisor()
        self.safety_mode = SafetyMode.RUNNING

    def run(self, steps: int, *, on_tick: Callable[[TickRecord], None] | None = None) -> list[TickRecord]:
        if steps <= 0: return []
        period_ns = int(round(self.backend.control_dt * 1e9))
        first = self.backend.read_state(tick_index=0)
        if not first.valid: raise RuntimeError(f"invalid initial hardware state: {first.validity_flags}")
        if self.state_validator is not None:
            self.state_validator(first.q_ctrl)
        history = PendingActionHistory(first.vector, self.backend.n_joints, self.history_len)
        current = first
        last_valid = first
        previous_safe_command = first.q_ctrl.copy()
        records: list[TickRecord] = []
        deadline = time.perf_counter_ns()
        previous_timestamp = first.timestamp_ns
        for tick in range(steps):
            safety_guard_failure: str | None = None
            wake = time.perf_counter_ns()
            wake_lateness = max(0.0, (wake - deadline) * 1e-9)
            if tick:
                measured = self.backend.read_state(tick_index=tick)
                actual_dt = (measured.timestamp_ns - previous_timestamp) * 1e-9
                decision = self.timing.evaluate(actual_dt, read_ok=measured.valid,
                                                timestamps_monotonic=measured.timestamp_ns > previous_timestamp,
                                                wake_lateness=wake_lateness)
                if decision.reset_history:
                    self.backend.reset_estimator_and_history() if hasattr(self.backend, "reset_estimator_and_history") else None
                    history.reset(measured.vector if measured.valid else last_valid.vector)
                    if self.planner: self.planner.clear(history.generation)
                elif decision.use_in_history:
                    history.observe(measured.vector)
                else:
                    history.skip_and_reanchor(measured.vector if measured.valid else last_valid.vector)
                    if self.planner: self.planner.clear(history.generation)
                if measured.valid:
                    last_valid = measured
                current = measured
                self.safety_mode = SafetyMode.HOLDING if decision.hold else SafetyMode.RUNNING
                previous_timestamp = measured.timestamp_ns
                if measured.valid and self.state_validator is not None:
                    try:
                        self.state_validator(measured.q_ctrl)
                    except Exception as exc:
                        safety_guard_failure = f"state_guard: {exc}"
                        self.safety_mode = SafetyMode.FAULT_LATCHED
            states, commands, generation = history.snapshot()
            if self.planner and self.safety_mode is SafetyMode.RUNNING:
                self.planner.submit(tick, current.timestamp_ns, states, commands, generation)
            nominal = np.asarray(self.nominal(tick, current), dtype=np.float32)
            residual = np.zeros(self.backend.n_joints, dtype=np.float32)
            applied = False
            packet_available = False
            packet_ood_valid = True
            temperatures = current.diagnostics.get("motor_temperature")
            confirmation_reader = getattr(self.backend, "confirm_temperature_samples", None)
            thermal = self.thermal.evaluate(
                None if temperatures is None else np.asarray(temperatures),
                float(current.diagnostics.get("diagnostic_sample_age_s", float("inf"))),
                time.perf_counter_ns(),
                None if "motor_current_raw" not in current.diagnostics else np.asarray(current.diagnostics["motor_current_raw"]),
                int(current.diagnostics.get("diagnostic_update_timestamp_ns", 0)) or None,
                confirmation_reader if callable(confirmation_reader) else None,
            )
            # RobotState is frozen but owns a mutable diagnostics mapping. The
            # recorder retains raw temperature separately, so write only the
            # derived audit fields here.
            current.diagnostics["motor_temperature_filtered"] = (
                np.full(6, np.nan, dtype=np.float32) if thermal.filtered_temperature is None
                else np.asarray(thermal.filtered_temperature, dtype=np.float32)
            )
            current.diagnostics["temperature_filter_status"] = "|".join(thermal.filter_status)
            config = getattr(self.backend, "config", None)
            voltage = evaluate_voltage(
                None if "motor_voltage_v" not in current.diagnostics else np.asarray(current.diagnostics["motor_voltage_v"]),
                warning_low=getattr(config, "voltage_warning_low", None),
                warning_high=getattr(config, "voltage_warning_high", None),
                hard_low=getattr(config, "voltage_hard_low", None),
                hard_high=getattr(config, "voltage_hard_high", None),
            )
            if voltage.mode is SafetyMode.FAULT_LATCHED:
                self.safety_mode = SafetyMode.FAULT_LATCHED
                if self.planner: self.planner.clear(history.generation)
            if thermal.mode is SafetyMode.FAULT_LATCHED:
                self.safety_mode = SafetyMode.FAULT_LATCHED
                if self.planner: self.planner.clear(history.generation)
            if thermal.request_torque_disable:
                self.backend.disable_torque()
                self.safety_mode = SafetyMode.TORQUE_DISABLED
                break
            if thermal.mode is SafetyMode.HOLDING:
                safety_guard_failure = thermal.reason
                self.safety_mode = SafetyMode.HOLDING
            if self.planner:
                packet = self.planner.latest()
                if packet is not None and packet.history_generation == generation and tick >= packet.activation_tick:
                    packet_available = True
                    packet_ood_valid = packet.ood_valid
                    residual = np.clip(packet.residual, -self.residual_limit, self.residual_limit).astype(np.float32)
                    applied = (self.mode is RealControlMode.ACTIVE_MPC and self.safety_mode is SafetyMode.RUNNING
                               and thermal.residual_allowed and voltage.residual_allowed and packet.ood_valid)
            requested = nominal + residual if applied else nominal
            if self.safety_mode is SafetyMode.RUNNING and self.command_validator is not None:
                try:
                    self.command_validator(requested)
                except Exception as exc:
                    safety_guard_failure = f"command_guard: {exc}"
                    self.safety_mode = SafetyMode.FAULT_LATCHED
            if self.safety_mode is not SafetyMode.RUNNING:
                requested = previous_safe_command.copy()
            if self.command_envelope == "hardware":
                # Guarded above: this non-Protocol method exists only on the
                # SO101 backend and uses hardware_joint_low/high, never the
                # narrower MPC experiment envelope.
                command = self.backend.send_hardware_joint_targets(requested, tick_index=tick)  # type: ignore[attr-defined]
            else:
                command = self.backend.send_joint_targets(requested, tick_index=tick)
            history.record_transmission(command.transmitted_q_ref, tick)
            if command.tx_local_success:
                previous_safe_command = command.transmitted_q_ref.copy()
            if not command.tx_local_success:
                self.safety_mode = SafetyMode.FAULT_LATCHED
                if self.planner: self.planner.clear(history.generation)
            elif self.command_validator is not None:
                try:
                    self.command_validator(command.transmitted_q_ref)
                except Exception as exc:
                    safety_guard_failure = f"transmitted_command_guard: {exc}"
                    self.safety_mode = SafetyMode.FAULT_LATCHED
            advance = advance_absolute_deadline(deadline, time.perf_counter_ns(), period_ns)
            record = TickRecord(current, command, nominal, self.mode, self.safety_mode, wake_lateness,
                                advance.skipped_ticks, residual, applied, packet_available, packet_ood_valid,
                                safety_guard_failure)
            records.append(record)
            if on_tick: on_tick(record)
            deadline = advance.next_deadline_ns
            sleep_until_ns(deadline)
            if self.safety_mode in {SafetyMode.FAULT_LATCHED, SafetyMode.TORQUE_DISABLED}:
                break
        return records
