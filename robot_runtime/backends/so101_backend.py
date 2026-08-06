from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from robot_runtime.config import SO101HardwareConfig, file_sha256
from robot_runtime.interfaces import CommandResult, RobotState
from robot_runtime.safety import CommandProjector
from robot_runtime.state_estimation import CausalVelocityEstimator


ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ALL_MOTORS = ARM_JOINTS + ("gripper",)
ENCODER_MAX = 4095


class CalibrationMismatchError(RuntimeError):
    pass


class BusOwnershipError(RuntimeError):
    pass


def degrees_to_raw(degrees: np.ndarray, range_min: np.ndarray, range_max: np.ndarray) -> np.ndarray:
    value = np.asarray(degrees, dtype=np.float64)
    low = np.asarray(range_min, dtype=np.float64)
    high = np.asarray(range_max, dtype=np.float64)
    return ((value * ENCODER_MAX / 360.0) + (low + high) / 2.0).astype(np.int64)


def raw_to_degrees(raw: np.ndarray, range_min: np.ndarray, range_max: np.ndarray) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    return (value - (np.asarray(range_min) + np.asarray(range_max)) / 2.0) * 360.0 / ENCODER_MAX


class SO101Backend:
    """Single-owner, one-position-read/one-goal-write SO101 backend.

    ``sync_write`` has no motor acknowledgement. CommandResult therefore uses
    ``transmitted`` terminology and low-rate Goal_Position readback audits the
    delivery interval.
    """

    n_joints = 5

    def __init__(self, config: SO101HardwareConfig, *, follower_factory: Callable[[SO101HardwareConfig], Any] | None = None):
        self.config = config
        self.control_dt = config.control_dt
        self.joint_low = config.joint_low.astype(np.float32)
        self.joint_high = config.joint_high.astype(np.float32)
        self._factory = follower_factory or self._make_follower
        self.follower: Any | None = None
        self.bus: Any | None = None
        self._owner_thread: int | None = None
        self._connected = False
        self._torque_enabled = False
        self._read_only = False
        self._history_generation = 0
        self._estimator = CausalVelocityEstimator(self.n_joints, 4.0)
        self._projector = CommandProjector(config.joint_low, config.joint_high,
                                           config.command_velocity_limit, config.command_acceleration_limit,
                                           config.relative_target_limit, config.control_dt)
        # Startup targets may legitimately be farther than the narrow
        # experiment ``relative_target_limit`` from the measured pose.  The
        # hardware projector is still bounded by the verified hardware range
        # and its velocity/acceleration limits; using the full hardware span
        # here avoids a static position error permanently pinning a target
        # three degrees away from home.
        self._hardware_projector = (None if config.hardware_joint_low is None else CommandProjector(
            config.hardware_joint_low, config.hardware_joint_high,
            config.command_velocity_limit, config.command_acceleration_limit,
            config.hardware_joint_high - config.hardware_joint_low, config.control_dt))
        self._last_state: RobotState | None = None
        self._last_command = config.home_q_ctrl.astype(np.float32).copy()
        self._gripper_raw: int | None = config.gripper_hold_raw
        self._last_goal_readback_ns = 0
        self._delivery_uncertain = False
        self._last_matching_readback_tick = -1
        self._diagnostic_cache: dict[str, Any] = {}
        self._diagnostic_timestamp_ns = 0
        self._diagnostic_sample_count = 0

    @property
    def hardware_startup_envelope_configured(self) -> bool:
        return self._hardware_projector is not None

    @staticmethod
    def _make_follower(config: SO101HardwareConfig) -> Any:
        try:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        except ImportError as exc:
            raise RuntimeError("SO101 hardware requires the pinned LeRobot environment") from exc
        follower_config = SO101FollowerConfig(
            port=config.port, id=config.robot_id, calibration_dir=config.calibration_path.parent,
            use_degrees=True, max_relative_target=None, num_read_retries=config.read_retries,
            position_p_coefficient=config.position_p, position_i_coefficient=config.position_i,
            position_d_coefficient=config.position_d,
        )
        return SO101Follower(follower_config)

    def _assert_owner(self) -> None:
        ident = threading.get_ident()
        if self._owner_thread is None:
            self._owner_thread = ident
        elif self._owner_thread != ident:
            raise BusOwnershipError("Feetech bus access is restricted to the control/I/O thread")

    def _calibration_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.follower is not None
        calibration = self.follower.calibration
        return (np.asarray([calibration[name].range_min for name in ALL_MOTORS], dtype=np.int64),
                np.asarray([calibration[name].range_max for name in ALL_MOTORS], dtype=np.int64))

    @property
    def is_connected(self) -> bool:
        return self._connected and self.bus is not None

    def q_ctrl_from_raw(self, raw: np.ndarray) -> np.ndarray:
        """Convert six calibrated raw encoder values to the five q_ctrl values.

        This is intentionally public for audited recovery/table-path checks.
        It performs no bus I/O and does not assert the q hardware envelope:
        a configured raw recovery endpoint can sit on an EEPROM raw endpoint
        while the normal control envelope remains inset from it.
        """
        values = np.asarray(raw, dtype=np.int64)
        if values.shape != (len(ALL_MOTORS),):
            raise ValueError("raw pose must contain exactly six motor values")
        low, high = self._calibration_arrays()
        return np.deg2rad(raw_to_degrees(values[:self.n_joints], low[:self.n_joints], high[:self.n_joints])).astype(np.float32)

    def _write_raw_goal(self, raw: np.ndarray, *, tick_index: int, requested_q: np.ndarray,
                        flags: tuple[str, ...] = ()) -> CommandResult:
        """Write an already validated six-motor raw target without projection."""
        assert self.bus is not None and self._connected
        values = np.asarray(raw, dtype=np.int64)
        if values.shape != (len(ALL_MOTORS),):
            raise ValueError("raw target must contain exactly six motor values")
        goals = {name: int(value) for name, value in zip(ALL_MOTORS, values, strict=True)}
        start = time.perf_counter_ns()
        success = True
        try:
            self.bus.sync_write("Goal_Position", goals, normalize=False)
        except Exception:
            success = False
        end = time.perf_counter_ns()
        q = np.asarray(requested_q, dtype=np.float32)
        if success:
            self._last_command = q.copy()
            self._last_tx_raw = values.copy()
        return CommandResult(q.copy(), q.copy(), q.copy(), values.copy(), success, int(tick_index),
                             start, end, flags, self._delivery_uncertain,
                             {"last_matching_goal_readback_tick": self._last_matching_readback_tick})

    def connect(self) -> None:
        self._assert_owner()
        path = self.config.calibration_path
        if not path.is_file():
            raise FileNotFoundError(f"SO101 calibration file not found: {path}")
        if file_sha256(path) != self.config.calibration_sha256:
            raise CalibrationMismatchError("local SO101 calibration SHA-256 does not match hardware config")
        self.follower = self._factory(self.config)
        # The factory must load precisely the file that was hashed.
        if Path(self.follower.calibration_fpath).resolve() != path.resolve():
            self.follower._load_calibration(path)
        self.bus = self.follower.bus
        self.bus.calibration = self.follower.calibration
        self.bus.connect()
        try:
            self.bus.disable_torque()
            if not self.bus.is_calibrated:
                raise CalibrationMismatchError("motor EEPROM calibration differs from the pinned calibration")
            calibration_low, calibration_high = self._calibration_arrays()
            if np.any(self.config.raw_low < calibration_low) or np.any(self.config.raw_high > calibration_high):
                raise ValueError("software raw limits must lie inside the calibrated EEPROM ranges")
            current_raw, _ = self._sync_read("Present_Position")
            if set(current_raw) != set(ALL_MOTORS):
                raise ConnectionError("initial position read did not return all six motors")
            # Preload the measured pose before changing configuration so a later
            # torque enable can never revive an EEPROM-stale target.
            self.bus.sync_write("Goal_Position", current_raw, normalize=False)
            # Configure without ever passing through LeRobot's default 254/254 connect path.
            self.bus.configure_motors(
                return_delay_time=self.config.return_delay_time,
                maximum_acceleration=self.config.maximum_acceleration,
                acceleration=self.config.acceleration,
            )
            for motor in ALL_MOTORS:
                self.bus.write("Operating_Mode", motor, 0)
                self.bus.write("P_Coefficient", motor, self.config.position_p)
                self.bus.write("I_Coefficient", motor, self.config.position_i_for_motor(motor))
                self.bus.write("D_Coefficient", motor, self.config.position_d)
            self.bus.write("Max_Torque_Limit", "gripper", 500)
            self.bus.write("Protection_Current", "gripper", 250)
            self.bus.write("Overload_Torque", "gripper", 25)
            self._verify_registers()
            self._gripper_raw = int(current_raw["gripper"]) if self._gripper_raw is None else self._gripper_raw
            self.bus.enable_torque()
            self._torque_enabled = True
            self._connected = True
        except Exception:
            try:
                self.bus.disable_torque()
            finally:
                self.bus.disconnect(False)
            raise

    def connect_read_only(self) -> None:
        """Connect for encoder/diagnostic reads without configuring or enabling motors.

        This is intended for manually supported geometry calibration only.  It
        deliberately performs no Goal_Position write and no EEPROM register
        write, and leaves torque disabled throughout the session.
        """
        self._assert_owner()
        path = self.config.calibration_path
        if not path.is_file():
            raise FileNotFoundError(f"SO101 calibration file not found: {path}")
        if file_sha256(path) != self.config.calibration_sha256:
            raise CalibrationMismatchError("local SO101 calibration SHA-256 does not match hardware config")
        self.follower = self._factory(self.config)
        if Path(self.follower.calibration_fpath).resolve() != path.resolve():
            self.follower._load_calibration(path)
        self.bus = self.follower.bus
        self.bus.calibration = self.follower.calibration
        self.bus.connect()
        try:
            self.bus.disable_torque()
            if not self.bus.is_calibrated:
                raise CalibrationMismatchError("motor EEPROM calibration differs from the pinned calibration")
            calibration_low, calibration_high = self._calibration_arrays()
            if np.any(self.config.raw_low < calibration_low) or np.any(self.config.raw_high > calibration_high):
                raise ValueError("software raw limits must lie inside the calibrated EEPROM ranges")
            current_raw, _ = self._sync_read("Present_Position")
            if set(current_raw) != set(ALL_MOTORS):
                raise ConnectionError("initial position read did not return all six motors")
            self._connected = True
            self._read_only = True
        except Exception:
            try:
                self.bus.disable_torque()
            finally:
                self.bus.disconnect(False)
            raise

    def _verify_registers(self) -> None:
        assert self.bus is not None
        checks = {
            "Return_Delay_Time": lambda _motor: self.config.return_delay_time,
            "Acceleration": lambda _motor: self.config.acceleration,
            "P_Coefficient": lambda _motor: self.config.position_p,
            "I_Coefficient": self.config.position_i_for_motor,
            "D_Coefficient": lambda _motor: self.config.position_d,
        }
        if getattr(self.bus, "protocol_version", 0) == 0:
            checks["Maximum_Acceleration"] = lambda _motor: self.config.maximum_acceleration
        for register, expected_for_motor in checks.items():
            for motor in ALL_MOTORS:
                expected = expected_for_motor(motor)
                actual = int(self.bus.read(register, motor, normalize=False))
                if actual != expected:
                    raise RuntimeError(f"{register} readback mismatch on {motor}: {actual} != {expected}")

    def _sync_read(self, register: str) -> tuple[dict[str, Any], int]:
        """Read all motors while recording retries actually attempted by this backend.

        LeRobot's internal retry counter is not exposed.  We therefore perform
        one-attempt reads ourselves rather than reporting a configured retry
        limit as though it were an observed quantity.
        """
        assert self.bus is not None
        last_error: Exception | None = None
        for retries in range(self.config.read_retries + 1):
            try:
                values = self.bus.sync_read(register, normalize=False, num_retry=0)
                if set(values) != set(ALL_MOTORS):
                    raise ConnectionError(f"{register} did not return all six motors")
                return values, retries
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def read_state(self, *, tick_index: int = 0) -> RobotState:
        self._assert_owner()
        if not self._connected or self.bus is None:
            raise RuntimeError("SO101 backend is not connected")
        start = time.perf_counter_ns()
        flags: list[str] = []
        try:
            positions, position_retries = self._sync_read("Present_Position")
        except Exception:
            positions = {}
            position_retries = self.config.read_retries
            flags.append("position_read_failed")
        end = time.perf_counter_ns()
        timestamp = (start + end) // 2
        raw = np.asarray([positions.get(name, -1) for name in ALL_MOTORS], dtype=np.int64)
        low, high = self._calibration_arrays()
        if np.any(raw < self.config.raw_low) or np.any(raw > self.config.raw_high):
            flags.append("raw_position_out_of_range")
        degrees = raw_to_degrees(raw[:5], low[:5], high[:5])
        q = np.deg2rad(degrees).astype(np.float32)
        try:
            dq = self._estimator.update(q, timestamp)
        except ValueError:
            dq = np.full(5, np.nan, dtype=np.float32)
            flags.append("non_monotonic_timestamp")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)): flags.append("non_finite_state")
        if self.config.hardware_joint_low is not None and (
                np.any(q < self.config.hardware_joint_low) or np.any(q > self.config.hardware_joint_high)):
            flags.append("hardware_joint_range")
        if np.any(np.abs(dq) > self.config.measured_velocity_plausible): flags.append("implausible_velocity")
        if np.any(np.abs(dq) > self.config.measured_velocity_emergency): flags.append("emergency_velocity")
        diagnostics = {"present_position_raw": raw, "raw_dq": self._estimator.last_raw.astype(np.float32),
                       "read_duration_s": (end - start) * 1e-9, "read_retry_count": position_retries,
                       "command_delivery_uncertain": self._delivery_uncertain,
                       "inside_experiment_envelope": bool(np.all(q >= self.joint_low) and np.all(q <= self.joint_high))}
        self._maybe_read_diagnostics(timestamp, tick_index, diagnostics, flags)
        diagnostics.update(self._diagnostic_cache)
        diagnostics["last_matching_goal_readback_tick"] = self._last_matching_readback_tick
        diagnostics["diagnostic_sample_age_s"] = ((timestamp - self._diagnostic_timestamp_ns) * 1e-9
                                                   if self._diagnostic_timestamp_ns else float("inf"))
        diagnostics["diagnostic_update_timestamp_ns"] = self._diagnostic_timestamp_ns
        diagnostics["diagnostic_sample_count"] = self._diagnostic_sample_count
        state = RobotState(q, dq, timestamp, (end - start) // 2, start, end, int(tick_index),
                           self._history_generation, self._estimator.generation, not flags, tuple(flags), diagnostics)
        self._last_state = state
        return state

    def _maybe_read_diagnostics(self, now_ns: int, tick_index: int, output: dict[str, Any], flags: list[str]) -> None:
        interval_ns = int(1e9 / self.config.goal_readback_rate_hz)
        if now_ns - self._last_goal_readback_ns < interval_ns or self.bus is None:
            return
        self._last_goal_readback_ns = now_ns
        fresh: dict[str, Any] = {}
        retry_count = 0
        complete = True
        for register, key in (("Present_Temperature", "motor_temperature"),
                              ("Present_Voltage", "motor_voltage_raw"), ("Present_Current", "motor_current_raw"),
                              ("Present_Load", "motor_load_raw"), ("Present_Velocity", "raw_present_velocity")):
            try:
                values, retries = self._sync_read(register)
                retry_count += retries
                value = np.asarray([values[name] for name in ALL_MOTORS])
                # The gripper is not part of the 5-DOF state estimator.
                fresh[key] = value[:5] if key == "raw_present_velocity" else value
            except Exception:
                output[f"{key}_read_failed"] = True
                complete = False
        try:
            goals, retries = self._sync_read("Goal_Position")
            retry_count += retries
            actual = np.asarray([goals[name] for name in ALL_MOTORS], dtype=np.int64)
            fresh["goal_position_readback_raw"] = actual
            # _last_tx_raw is deliberately private state: readback only bounds an uncertain interval.
            mismatch = hasattr(self, "_last_tx_raw") and not np.array_equal(actual, self._last_tx_raw)
            fresh["goal_readback_mismatch"] = mismatch
            if mismatch:
                self._delivery_uncertain = True
                flags.append("goal_readback_mismatch")
            else:
                self._delivery_uncertain = False
                self._last_matching_readback_tick = tick_index
        except Exception:
            output["goal_readback_failed"] = True
            complete = False
        self._diagnostic_cache.update(fresh)
        if "motor_voltage_raw" in fresh:
            self._diagnostic_cache["motor_voltage_v"] = np.asarray(fresh["motor_voltage_raw"], dtype=np.float32) * 0.1
        self._diagnostic_cache["diagnostic_retry_count"] = retry_count
        if complete:
            self._diagnostic_timestamp_ns = now_ns
            self._diagnostic_sample_count += 1

    def confirm_temperature_samples(self) -> list[np.ndarray]:
        """Return two fresh all-motor temperature rereads for anomaly confirmation.

        This is called only after a large temperature jump.  It performs no
        position read, no goal write, and no register modification.  A failed
        reread raises so the safety layer can hold rather than trusting a
        possibly corrupt high-temperature packet.
        """
        self._assert_owner()
        if self.bus is None or not self._connected:
            raise RuntimeError("SO101 backend is not connected")
        samples: list[np.ndarray] = []
        for _ in range(2):
            values, _retries = self._sync_read("Present_Temperature")
            samples.append(np.asarray([values[name] for name in ALL_MOTORS], dtype=np.float64))
        return samples

    def send_joint_targets(self, q_ref: np.ndarray, *, tick_index: int = 0) -> CommandResult:
        self._assert_owner()
        if self._read_only:
            raise RuntimeError("read-only SO101 connection cannot send joint targets")
        if self._last_state is None or self.bus is None or not self._connected:
            raise RuntimeError("read_state must succeed before sending a target")
        return self._send_joint_targets(q_ref, tick_index=tick_index, projector=self._projector)

    def _send_joint_targets(self, q_ref: np.ndarray, *, tick_index: int, projector: CommandProjector) -> CommandResult:
        assert self._last_state is not None and self.bus is not None and self._connected
        requested = np.asarray(q_ref, dtype=np.float32)
        projected = projector.project(requested, self._last_state.q_ctrl, self._last_command)
        cal_low, cal_high = self._calibration_arrays()
        arm_raw = degrees_to_raw(np.rad2deg(projected.q_ref), cal_low[:5], cal_high[:5])
        raw = np.concatenate((arm_raw, [int(self._gripper_raw)]))
        raw = np.clip(raw, self.config.raw_low, self.config.raw_high).astype(np.int64)
        quantized = np.deg2rad(raw_to_degrees(raw[:5], cal_low[:5], cal_high[:5])).astype(np.float32)
        goals = {name: int(value) for name, value in zip(ALL_MOTORS, raw, strict=True)}
        start = time.perf_counter_ns()
        success = True
        try:
            self.bus.sync_write("Goal_Position", goals, normalize=False)
        except Exception:
            success = False
        end = time.perf_counter_ns()
        if success:
            self._last_command = quantized.copy()
            self._last_tx_raw = raw.copy()
        return CommandResult(requested.copy(), projected.q_ref.copy(), quantized, raw, success, int(tick_index),
                             start, end, projected.flags, self._delivery_uncertain,
                             {"last_matching_goal_readback_tick": self._last_matching_readback_tick})

    def startup_to_home(self, duration_s: float = 3.0, *, convergence_timeout_s: float = 15.0,
                        home_tolerance_rad: float = np.deg2rad(1.0),
                        path_validator: Callable[[np.ndarray], Any] | None = None) -> RobotState:
        """Safely enter the narrow experiment envelope from a verified pose.

        ``duration_s`` is the commanded, rate-limited trajectory duration;
        it is not an assertion that a compliant position servo has settled by
        then.  The subsequent hold continues to transmit home until the real
        encoder reaches a home tolerance that leaves room inside the experiment
        envelope, or the explicit timeout.
        """
        self._assert_owner()
        if duration_s <= 0 or convergence_timeout_s <= 0 or home_tolerance_rad <= 0:
            raise ValueError("startup duration, convergence timeout, and home tolerance must be positive")
        initial = self.prepare_hardware_motion()
        if path_validator is not None:
            path_validator(initial.q_ctrl)
        self.move_to_configuration(self.config.home_q_ctrl, duration_s, envelope="hardware",
                                   path_validator=path_validator)
        settled = self.read_state(tick_index=0)
        deadline = time.monotonic() + convergence_timeout_s
        tick = 0
        def home_converged(observation: RobotState) -> bool:
            return bool(np.max(np.abs(observation.q_ctrl - self.config.home_q_ctrl)) <= home_tolerance_rad)

        while not home_converged(settled) and time.monotonic() < deadline:
            result = self._send_joint_targets(self.config.home_q_ctrl, tick_index=tick,
                                                projector=self._hardware_projector)
            if not result.tx_local_success:
                raise RuntimeError("startup hold write failed")
            time.sleep(self.control_dt)
            tick += 1
            settled = self.read_state(tick_index=tick)
            if not settled.valid:
                raise RuntimeError(f"startup hold reached invalid hardware state: {settled.validity_flags}")
        if not home_converged(settled):
            error_deg = np.rad2deg(settled.q_ctrl - self.config.home_q_ctrl)
            goal_readback = np.asarray(settled.diagnostics.get("goal_position_readback_raw", np.full(6, -1))).tolist()
            last_tx = (np.asarray(getattr(self, "_last_tx_raw", np.full(6, -1))).astype(int).tolist())
            present = np.asarray(settled.diagnostics.get("present_position_raw", np.full(6, -1))).tolist()
            raise RuntimeError("startup did not converge to home before timeout; "
                               f"home_tolerance_deg={np.rad2deg(home_tolerance_rad):.3f}, "
                               f"final_q_ctrl_rad={settled.q_ctrl.tolist()}, error_deg={error_deg.tolist()}, "
                               f"present_raw={present}, last_transmitted_raw={last_tx}, "
                               f"goal_position_readback_raw={goal_readback}")
        if not bool(settled.diagnostics["inside_experiment_envelope"]):
            raise RuntimeError("home convergence result unexpectedly lies outside the experiment envelope")
        return settled

    def prepare_hardware_motion(self) -> RobotState:
        """Latch the measured pose as the first hardware-motion target."""
        self._assert_owner()
        if self._hardware_projector is None:
            raise RuntimeError("motion requires manually verified hardware_joint_low/high in the hardware config")
        state = self.read_state(tick_index=0)
        if not state.valid:
            raise RuntimeError(f"refusing hardware motion outside valid state: {state.validity_flags}")
        # ``connect`` preloads raw motor targets; mirror that in the software
        # projector so its first acceleration-limited command is continuous.
        self._last_command = state.q_ctrl.copy()
        self._hardware_projector.reset()
        return state

    def send_hardware_joint_targets(self, q_ref: np.ndarray, *, tick_index: int = 0) -> CommandResult:
        """Send a target under the wide verified startup envelope only."""
        self._assert_owner()
        if self._read_only:
            raise RuntimeError("read-only SO101 connection cannot send hardware targets")
        if self._hardware_projector is None:
            raise RuntimeError("hardware startup envelope is not configured")
        if self._last_state is None:
            raise RuntimeError("prepare_hardware_motion or read_state must run before hardware targets")
        return self._send_joint_targets(q_ref, tick_index=tick_index, projector=self._hardware_projector)

    def move_selected_joints_to_configuration(self, target_q: np.ndarray, active_mask: np.ndarray,
                                               duration_s: float) -> None:
        """Move only selected joints through the hardware startup envelope."""
        if duration_s <= 0 or self._last_state is None:
            raise ValueError("duration_s must be positive and a state must have been read")
        mask = np.asarray(active_mask, dtype=bool)
        target = np.asarray(target_q, dtype=np.float32)
        if mask.shape != (self.n_joints,) or target.shape != (self.n_joints,):
            raise ValueError("active_mask and target_q must have one value per controlled joint")
        start = self._last_state.q_ctrl.copy()
        steps = max(1, int(np.ceil(duration_s / self.control_dt)))
        for index in range(1, steps + 1):
            state = self.read_state(tick_index=index)
            blend = 0.5 - 0.5 * np.cos(np.pi * index / steps)
            desired = start.copy()
            desired[mask] = start[mask] + blend * (target[mask] - start[mask])
            result = self.send_hardware_joint_targets(desired, tick_index=index)
            if not state.valid or not result.tx_local_success:
                raise RuntimeError("selected hardware move aborted due to invalid state or write failure")
            time.sleep(self.control_dt)

    def manual_raw_safety_jog(self, joint_index: int, delta_rad: float, *, tick_index: int = 0) -> CommandResult:
        """B2-only, operator-supervised single tiny jog before q-range freeze.

        This deliberately bypasses the experiment envelope, but never clips a
        requested target into the raw safety envelope: an out-of-range target
        is refused.  It is unsuitable for direct control, data collection, or
        MPC and exists only to establish manually verified hardware limits.
        """
        self._assert_owner()
        if self._last_state is None or self.bus is None or not self._connected:
            raise RuntimeError("read_state must succeed before a manual raw-safety jog")
        if not 0 <= int(joint_index) < self.n_joints:
            raise ValueError("joint_index is out of range")
        if not 0 < abs(float(delta_rad)) <= np.deg2rad(1.0):
            raise ValueError("B2 manual jog is limited to a nonzero 1.0 degree step")
        requested = self._last_state.q_ctrl.astype(np.float32).copy()
        requested[int(joint_index)] += float(delta_rad)
        cal_low, cal_high = self._calibration_arrays()
        present_raw = np.asarray(self._last_state.diagnostics["present_position_raw"], dtype=np.int64)
        # Preserve the other five motors' *measured raw values* exactly.  A
        # q->raw round trip can otherwise move an unselected motor by one
        # encoder count purely due to quantization.
        raw = present_raw.copy()
        target_raw = degrees_to_raw(np.rad2deg(requested[int(joint_index)]),
                                    cal_low[int(joint_index)], cal_high[int(joint_index)])
        raw[int(joint_index)] = int(target_raw)
        if np.any(raw < self.config.raw_low) or np.any(raw > self.config.raw_high):
            raise RuntimeError("B2 jog target is outside configured raw safety envelope")
        goals = {name: int(value) for name, value in zip(ALL_MOTORS, raw, strict=True)}
        start = time.perf_counter_ns()
        success = True
        try:
            self.bus.sync_write("Goal_Position", goals, normalize=False)
        except Exception:
            success = False
        end = time.perf_counter_ns()
        if success:
            self._last_command = requested.copy()
            self._last_tx_raw = raw.copy()
        return CommandResult(requested.copy(), requested.copy(), requested.copy(), raw, success, int(tick_index),
                             start, end, ("b2_manual_raw_safety_jog",), self._delivery_uncertain,
                             {"b2_only": True, "last_matching_goal_readback_tick": self._last_matching_readback_tick})

    def read_actuation_registers(self) -> dict[str, dict[str, int]]:
        """Read per-motor torque/mode registers for a no-motion hardware diagnosis."""
        self._assert_owner()
        if self.bus is None or not self._connected:
            raise RuntimeError("SO101 backend is not connected")
        registers = ("Torque_Enable", "Lock", "Operating_Mode", "Min_Position_Limit", "Max_Position_Limit",
                     "CW_Dead_Zone", "CCW_Dead_Zone", "Max_Torque_Limit", "Torque_Limit",
                     "P_Coefficient", "I_Coefficient", "D_Coefficient", "Acceleration", "Maximum_Acceleration",
                     "Present_Voltage", "Present_Temperature", "Present_Current", "Present_Load")
        result: dict[str, dict[str, int]] = {}
        for register in registers:
            result[register] = {}
            for motor in ALL_MOTORS:
                result[register][motor] = int(self.bus.read(register, motor, normalize=False,
                                                             num_retry=self.config.read_retries))
        return result

    def retransmit_measured_position(self, *, tick_index: int = 0) -> CommandResult:
        """Echo the latest six raw positions without applying control limits.

        This is intentionally restricted to static I/O benchmarking. It does
        not move toward ``home_q_ctrl`` and refuses any raw measurement outside
        the configured hardware safety envelope.
        """
        self._assert_owner()
        if self._last_state is None or self.bus is None or not self._connected:
            raise RuntimeError("read_state must succeed before retransmitting the measured pose")
        raw = np.asarray(self._last_state.diagnostics.get("present_position_raw"), dtype=np.int64)
        if raw.shape != (len(ALL_MOTORS),):
            raise RuntimeError("latest state does not contain all six raw motor positions")
        if np.any(raw < self.config.raw_low) or np.any(raw > self.config.raw_high):
            raise RuntimeError("refusing to retransmit a measured pose outside the raw safety envelope")
        q = self._last_state.q_ctrl.astype(np.float32).copy()
        goals = {name: int(value) for name, value in zip(ALL_MOTORS, raw, strict=True)}
        start = time.perf_counter_ns()
        success = True
        try:
            self.bus.sync_write("Goal_Position", goals, normalize=False)
        except Exception:
            success = False
        end = time.perf_counter_ns()
        if success:
            self._last_command = q.copy()
            self._last_tx_raw = raw.copy()
        return CommandResult(q.copy(), q.copy(), q.copy(), raw.copy(), success, int(tick_index),
                             start, end, (), self._delivery_uncertain,
                             {"benchmark_echo": True,
                              "last_matching_goal_readback_tick": self._last_matching_readback_tick})

    def move_to_configuration(self, target_q: np.ndarray, duration_s: float, *, envelope: str = "hardware",
                              path_validator: Callable[[np.ndarray], Any] | None = None) -> None:
        if duration_s <= 0 or self._last_state is None:
            raise ValueError("duration_s must be positive and a state must have been read")
        if envelope == "hardware":
            if self._hardware_projector is None:
                raise RuntimeError("hardware startup envelope is not configured")
            projector = self._hardware_projector
        elif envelope == "experiment":
            projector = self._projector
        else:
            raise ValueError("envelope must be 'hardware' or 'experiment'")
        start = self._last_state.q_ctrl.copy()
        steps = max(1, int(np.ceil(duration_s / self.control_dt)))
        target = np.asarray(target_q, dtype=np.float32)
        if path_validator is not None:
            # Validate every planned cosine-interpolation point before the
            # first transmission; a table constraint may not be checked only
            # at the start/end configurations.
            planned = [start + (0.5 - 0.5 * np.cos(np.pi * index / steps)) * (target - start)
                       for index in range(1, steps + 1)]
            path_validator(start)
            for point in planned:
                path_validator(point)
        for index in range(1, steps + 1):
            state = self.read_state(tick_index=index)
            blend = 0.5 - 0.5 * np.cos(np.pi * index / steps)
            result = self._send_joint_targets(start + blend * (target - start), tick_index=index,
                                              projector=projector)
            if not state.valid or not result.tx_local_success:
                raise RuntimeError("safe move aborted due to invalid state or write failure")
            time.sleep(self.control_dt)

    def move_to_shutdown_recovery_raw(self, target_raw: np.ndarray, duration_s: float, *,
                                      path_validator: Callable[[np.ndarray], Any] | None = None) -> dict[str, Any]:
        """Perform the collector-only, guarded return to a configured raw pose.

        The measured state must be valid *before* any recovery command.  The
        trajectory is checked to the exact raw endpoint by ``path_validator``
        (the E3 fine-mesh table gate), then executed through the regular
        hardware q envelope.  A final raw write is permitted only for a tiny
        endpoint difference (at most eight encoder counts per arm joint), so
        an EEPROM endpoint such as elbow raw=3071 can be reached without
        widening the normal data-collection safety envelope.

        This method deliberately does not try to recover from a bad state or
        a communication failure; its caller must skip movement in those cases.
        """
        self._assert_owner()
        if duration_s <= 0:
            raise ValueError("shutdown recovery duration must be positive")
        if self._read_only:
            raise RuntimeError("read-only SO101 connection cannot execute shutdown recovery")
        if self._hardware_projector is None:
            raise RuntimeError("shutdown recovery requires a verified hardware envelope")
        requested_raw = np.asarray(target_raw, dtype=np.int64)
        if requested_raw.shape != (len(ALL_MOTORS),):
            raise ValueError("shutdown recovery target must contain six raw values")
        if np.any(requested_raw < self.config.raw_low) or np.any(requested_raw > self.config.raw_high):
            raise RuntimeError("shutdown recovery raw target is outside the configured raw safety envelope")

        initial = self.read_state(tick_index=0)
        if not initial.valid:
            raise RuntimeError(f"shutdown recovery refuses invalid starting state: {initial.validity_flags}")
        target_q = self.q_ctrl_from_raw(requested_raw)
        if path_validator is not None:
            steps = max(1, int(np.ceil(duration_s / self.control_dt)))
            path_validator(initial.q_ctrl)
            for index in range(1, steps + 1):
                blend = 0.5 - 0.5 * np.cos(np.pi * index / steps)
                path_validator(initial.q_ctrl + blend * (target_q - initial.q_ctrl))

        # This clips only the regular q portion of the move.  The requested
        # raw endpoint is written after that move, and must be only a few
        # counts farther than this validated q target.
        q_envelope_target = np.clip(target_q, self.config.hardware_joint_low,
                                    self.config.hardware_joint_high).astype(np.float32)
        cal_low, cal_high = self._calibration_arrays()
        envelope_raw = np.concatenate((
            degrees_to_raw(np.rad2deg(q_envelope_target), cal_low[:self.n_joints], cal_high[:self.n_joints]),
            [int(requested_raw[-1])],
        )).astype(np.int64)
        final_delta_counts = np.abs(requested_raw[:self.n_joints] - envelope_raw[:self.n_joints])
        if np.any(final_delta_counts > 8):
            raise RuntimeError(
                "shutdown recovery endpoint requires more than eight raw counts outside the hardware q envelope: "
                f"{final_delta_counts.tolist()}"
            )

        # The ordinary move keeps all intermediate state/commands within the
        # q hardware envelope and rechecks validity before every transmission.
        self._last_command = initial.q_ctrl.copy()
        self._hardware_projector.reset()
        self.move_to_configuration(q_envelope_target, duration_s, envelope="hardware",
                                   path_validator=path_validator)
        before_final = self.read_state(tick_index=max(1, int(np.ceil(duration_s / self.control_dt))) + 1)
        if not before_final.valid:
            raise RuntimeError(f"shutdown recovery aborted before final raw step: {before_final.validity_flags}")
        final = self._write_raw_goal(requested_raw, tick_index=before_final.tick_index + 1,
                                     requested_q=target_q,
                                     flags=("guarded_shutdown_raw_endpoint",))
        if not final.tx_local_success:
            raise RuntimeError("shutdown recovery final raw write failed")
        # Only the final elbow correction is allowed outside the normal q
        # envelope.  Verify its actual raw position directly rather than
        # calling read_state(), whose q-envelope flag is expected at raw=3071.
        #
        # Some gravity-loaded joints (especially shoulder_lift) need longer
        # than the previous fixed 0.5 s to close their final encoder error.
        # Keep torque and the already-validated target active for at most
        # five seconds, while retaining the original 12-count acceptance
        # criterion.  This is a convergence wait, not a relaxation of the
        # recovery safety contract.
        settle_timeout_s = 5.0
        settle_poll_s = 0.25
        settle_started = time.monotonic()
        settled_retries = 0
        while True:
            settled_values, read_retries = self._sync_read("Present_Position")
            settled_retries += int(read_retries)
            settled_raw = np.asarray([settled_values[name] for name in ALL_MOTORS], dtype=np.int64)
            final_error_counts = np.abs(settled_raw - requested_raw)
            if np.all(final_error_counts <= 12):
                break
            elapsed_s = time.monotonic() - settle_started
            if elapsed_s >= settle_timeout_s:
                raise RuntimeError(
                    "shutdown recovery did not settle within 12 raw counts before "
                    f"{settle_timeout_s:.1f}s: target={requested_raw.tolist()}, "
                    f"present={settled_raw.tolist()}, error={final_error_counts.tolist()}"
                )
            time.sleep(min(settle_poll_s, settle_timeout_s - elapsed_s))
        return {
            "target_raw": requested_raw.astype(int).tolist(),
            "target_q_ctrl_rad": target_q.tolist(),
            "hardware_envelope_target_raw": envelope_raw.astype(int).tolist(),
            "final_endpoint_delta_counts": final_delta_counts.astype(int).tolist(),
            "last_goal_raw": final.tx_goal_position_raw.astype(int).tolist(),
            "settled_raw": settled_raw.astype(int).tolist(),
            "settle_error_counts": final_error_counts.astype(int).tolist(),
            "settle_read_retries": int(settled_retries),
            "settle_wait_s": float(time.monotonic() - settle_started),
        }

    def disable_torque(self) -> None:
        self._assert_owner()
        if self.bus is not None and self._torque_enabled:
            self.bus.disable_torque()
            self._torque_enabled = False

    def reset_estimator_and_history(self) -> None:
        self._history_generation += 1
        self._estimator.reset()
        self._projector.reset()
        if self._hardware_projector is not None:
            self._hardware_projector.reset()

    def close(self) -> None:
        self._assert_owner()
        if self.bus is not None:
            if self._torque_enabled:
                self.bus.disable_torque()
            self.bus.disconnect(False)
        self._connected = False
        self._torque_enabled = False
        self._read_only = False
