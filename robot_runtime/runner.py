from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable, Protocol

import numpy as np

from robot_runtime.history import PendingActionHistory
from robot_runtime.interfaces import CommandResult, RobotBackend, RobotState
from robot_runtime.safety import (SafetyMode, ThermalDecision, ThermalSupervisor,
                                  TimingSupervisor, VoltageDecision, evaluate_voltage)
from robot_runtime.timing import advance_absolute_deadline, sleep_until_ns


class RealControlMode(str, Enum):
    DIRECT = "direct"
    SHADOW_MPC = "shadow_mpc"
    ACTIVE_MPC = "active_mpc"


def active_mpc_gate_open(mode: RealControlMode, tick: int, active_start_tick: int) -> bool:
    """Whether active MPC is allowed to apply a planner packet this tick."""
    return mode is not RealControlMode.ACTIVE_MPC or int(tick) >= int(active_start_tick)


@dataclass(frozen=True)
class PlannerCommand:
    residual: np.ndarray
    history_generation: int
    activation_tick: int
    publication_tick: int
    ood_valid: bool = True
    # When present, this is the planner's pre-projection absolute request for
    # the current packet index. The backend applies the canonical executable
    # state machine against the live state. ``expected_raw`` is an audit
    # prediction from the delayed planner forecast; a mismatch is recorded,
    # not treated as an automatic Direct fallback.
    absolute_q_ref: np.ndarray | None = None
    expected_raw: np.ndarray | None = None


def compose_requested_command(
    nominal: np.ndarray,
    residual: np.ndarray,
    planner_command: PlannerCommand | None,
    *,
    applied: bool,
) -> np.ndarray:
    """Select the command passed to the backend before its final safety gate.

    An active planner packet with an absolute target is already projected in
    the same command space used by the rollout.  Reconstructing it from the
    live nominal and a raw residual would silently change the command.
    """
    if applied and planner_command is not None and planner_command.absolute_q_ref is not None:
        absolute = np.asarray(planner_command.absolute_q_ref, dtype=np.float32)
        if absolute.shape == np.asarray(nominal).shape and np.all(np.isfinite(absolute)):
            return absolute.copy()
    return (np.asarray(nominal, dtype=np.float32) + np.asarray(residual, dtype=np.float32)).astype(np.float32)


class Planner(Protocol):
    def submit(self, tick_index: int, state_timestamp_ns: int, states: np.ndarray, commands: np.ndarray,
               history_generation: int, executable_command_state: object | None = None) -> None: ...
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
                 history_len: int = 8, residual_limit_rad: float | np.ndarray = np.deg2rad(2.0),
                 command_envelope: str = "experiment",
                 state_validator: Callable[[np.ndarray], object] | None = None,
                 command_validator: Callable[[np.ndarray], object] | None = None,
                 mpc_reference_prevalidated: bool = False,
                 active_start_tick: int = 0):
        if mode is not RealControlMode.DIRECT and planner is None:
            raise ValueError("shadow_mpc and active_mpc require a planner")
        if command_envelope not in {"experiment", "hardware", "workspace"}:
            raise ValueError("command_envelope must be 'experiment', 'hardware', or 'workspace'")
        if command_envelope in {"hardware", "workspace"} and mode is not RealControlMode.DIRECT:
            # MPC references span the data-collection envelope (e.g. the 4 cm
            # circle needs pan +/-6.5 deg), not the first-motion +/-3 deg
            # authority.  The caller may relax the envelope only after the
            # joint reference passed its manifest SHA-256 match and the
            # JointFilePlayer gates (hardware envelope / home start / per-sample
            # step); the residual itself stays bounded by residual_limit_rad.
            if not (mpc_reference_prevalidated and mode in
                    {RealControlMode.SHADOW_MPC, RealControlMode.ACTIVE_MPC}):
                raise ValueError("the hardware/workspace envelopes are allowed for direct data collection, "
                                 "or for shadow/active MPC only after the joint reference passed its "
                                 "manifest + JointFilePlayer gates (mpc_reference_prevalidated=True)")
        if command_envelope == "hardware" and not hasattr(backend, "send_hardware_joint_targets"):
            raise TypeError("backend does not provide the explicitly verified hardware command envelope")
        if command_envelope == "workspace" and not hasattr(backend, "send_workspace_joint_targets"):
            raise TypeError("backend does not provide the workspace-bounded command envelope")
        self.backend, self.nominal, self.mode, self.planner = backend, nominal, mode, planner
        self.command_envelope = command_envelope
        self.state_validator = state_validator
        self.command_validator = command_validator
        self.history_len = int(history_len)
        residual_limit = np.asarray(residual_limit_rad, dtype=np.float32)
        if residual_limit.ndim not in {0, 1} or not np.all(np.isfinite(residual_limit)) or np.any(residual_limit <= 0.0):
            raise ValueError("residual_limit_rad must be a finite positive scalar or per-joint vector")
        self.residual_limit = (
            float(residual_limit)
            if residual_limit.ndim == 0
            else residual_limit.copy()
        )
        if active_start_tick < 0:
            raise ValueError("active_start_tick must be non-negative")
        self.active_start_tick = int(active_start_tick)
        self.timing = TimingSupervisor(backend.control_dt)
        self.thermal = ThermalSupervisor()
        self.safety_mode = SafetyMode.RUNNING
        # Populated only when run() stops before exhausting its step budget; a
        # human-readable reason for the early stop.  None on a full run.
        self.last_stop_reason: str | None = None

    def _describe_stop(self, *, safety_guard_failure: str | None, thermal: ThermalDecision,
                       voltage: VoltageDecision, tx_ok: bool,
                       filtered_temperature: np.ndarray | None) -> str:
        """Build a concise, specific reason for an early loop stop.

        Thermal faults deliberately do not set safety_guard_failure, so the
        reason must be reconstructed from the supervisor decisions that were in
        scope when the loop broke.  Guard failures (which already embed their
        trigger in the message) take priority, then thermal, voltage, and a
        local transmission failure.
        """
        if safety_guard_failure:
            return safety_guard_failure
        if thermal.request_torque_disable or thermal.mode is SafetyMode.FAULT_LATCHED:
            description = f"thermal({thermal.reason or 'fault_latched'})"
            values = np.asarray(filtered_temperature, dtype=np.float64).reshape(-1)
            if values.size and np.any(np.isfinite(values)):
                hottest = int(np.nanargmax(values))
                description += f" hottest_motor={hottest} Tmax={float(values[hottest]):.1f}C"
            return description
        if voltage.mode is SafetyMode.FAULT_LATCHED:
            return f"voltage({voltage.reason or 'hard_limit'})"
        if not tx_ok:
            return "tx_local_success=False"
        return f"safety_mode={self.safety_mode.value}"

    def run(self, steps: int, *, on_tick: Callable[[TickRecord], None] | None = None) -> list[TickRecord]:
        if steps <= 0: return []
        self.last_stop_reason = None
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
            active_gate_open = active_mpc_gate_open(self.mode, tick, self.active_start_tick)
            if self.planner and self.safety_mode is SafetyMode.RUNNING:
                if active_gate_open:
                    self.planner.submit(
                        tick, current.timestamp_ns, states, commands, generation,
                        executable_command_state=getattr(self.backend, "executable_command_state", lambda: None)(),
                    )
                else:
                    # Do not let CEM packets accumulated during a static
                    # startup hold leak into the first moving reference.
                    self.planner.clear(generation)
            nominal = np.asarray(self.nominal(tick, current), dtype=np.float32)
            residual = np.zeros(self.backend.n_joints, dtype=np.float32)
            applied = False
            packet_available = False
            packet_ood_valid = True
            packet: PlannerCommand | None = None
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
                # command is not yet defined this tick, so a transmission check
                # is not meaningful; the thermal decision fully explains this.
                self.last_stop_reason = self._describe_stop(
                    safety_guard_failure=safety_guard_failure, thermal=thermal, voltage=voltage,
                    tx_ok=True, filtered_temperature=current.diagnostics.get("motor_temperature_filtered"))
                break
            if thermal.mode is SafetyMode.HOLDING:
                safety_guard_failure = thermal.reason
                self.safety_mode = SafetyMode.HOLDING
            if self.planner and active_gate_open:
                packet = self.planner.latest()
                if packet is not None and packet.history_generation == generation and tick >= packet.activation_tick:
                    packet_available = True
                    packet_ood_valid = packet.ood_valid
                    residual = np.clip(packet.residual, -self.residual_limit, self.residual_limit).astype(np.float32)
                    applied = (self.mode is RealControlMode.ACTIVE_MPC and self.safety_mode is SafetyMode.RUNNING
                               and thermal.residual_allowed and voltage.residual_allowed and packet.ood_valid)
            # The planner may provide the absolute, kinematically projected
            # target that was used during CEM scoring.  Use it directly in
            # active mode so the command sent to the backend is aligned with
            # the rollout.  The backend still applies its final hardware
            # safety projector; this is the last safety boundary, not a
            # second residual re-anchoring step.
            requested = (
                compose_requested_command(nominal, residual, packet, applied=True)
                if applied
                else np.asarray(nominal, dtype=np.float32).copy()
            )
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
                command = self.backend.send_hardware_joint_targets(  # type: ignore[attr-defined]
                    requested, tick_index=tick,
                    expected_raw=(packet.expected_raw if applied and packet is not None else None),
                    fallback_q_ref=(nominal if applied and packet is not None else None),
                )
            elif self.command_envelope == "workspace":
                command = self.backend.send_workspace_joint_targets(  # type: ignore[attr-defined]
                    requested, tick_index=tick,
                    expected_raw=(packet.expected_raw if applied and packet is not None else None),
                    fallback_q_ref=(nominal if applied and packet is not None else None),
                )
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
            planner_applied = applied and "planner_raw_mismatch_direct_fallback" not in command.projection_flags
            record = TickRecord(current, command, nominal, self.mode, self.safety_mode, wake_lateness,
                                advance.skipped_ticks, residual, planner_applied, packet_available, packet_ood_valid,
                                safety_guard_failure)
            records.append(record)
            if on_tick: on_tick(record)
            deadline = advance.next_deadline_ns
            sleep_until_ns(deadline)
            if self.safety_mode in {SafetyMode.FAULT_LATCHED, SafetyMode.TORQUE_DISABLED}:
                self.last_stop_reason = self._describe_stop(
                    safety_guard_failure=safety_guard_failure, thermal=thermal, voltage=voltage,
                    tx_ok=command.tx_local_success,
                    filtered_temperature=current.diagnostics.get("motor_temperature_filtered"))
                break
        return records
