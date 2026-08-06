from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array(payload: dict[str, Any], key: str, length: int, dtype: Any = np.float64) -> np.ndarray:
    value = np.asarray(payload[key], dtype=dtype)
    if value.shape != (length,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain {length} finite values")
    return value


@dataclass(frozen=True)
class SO101HardwareConfig:
    port: str
    robot_id: str
    calibration_path: Path
    calibration_sha256: str
    control_dt: float
    joint_names: tuple[str, ...]
    gripper_name: str
    joint_low: np.ndarray
    joint_high: np.ndarray
    hardware_joint_low: np.ndarray | None
    hardware_joint_high: np.ndarray | None
    raw_low: np.ndarray
    raw_high: np.ndarray
    home_q_ctrl: np.ndarray
    command_velocity_limit: np.ndarray
    command_acceleration_limit: np.ndarray
    relative_target_limit: np.ndarray
    measured_velocity_plausible: np.ndarray
    measured_velocity_emergency: np.ndarray
    position_p: int
    position_i: int
    position_d: int
    acceleration: int
    maximum_acceleration: int
    return_delay_time: int
    gripper_hold_raw: int | None
    read_retries: int
    diagnostic_rate_hz: float
    goal_readback_rate_hz: float
    lerobot_commit: str
    config_sha256: str
    voltage_warning_low: float | None = None
    voltage_warning_high: float | None = None
    voltage_hard_low: float | None = None
    voltage_hard_high: float | None = None
    # Optional per-motor I-gain overrides.  Motors not listed use
    # ``position_i``.  This keeps the safe all-zero default explicit while
    # permitting a gravity-loaded joint to receive a small integral term.
    position_i_by_motor: dict[str, int] | None = None
    # Optional six-motor pose used only by the E3 collector's guarded
    # shutdown recovery.  It is deliberately raw so the gripper can be
    # included and the requested physical pose is unambiguous.
    shutdown_recovery_raw: np.ndarray | None = None
    shutdown_recovery_seconds: float = 10.0

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def position_i_for_motor(self, motor: str) -> int:
        if self.position_i_by_motor is None:
            return self.position_i
        return int(self.position_i_by_motor.get(motor, self.position_i))

    def effective_position_i_by_motor(self) -> dict[str, int]:
        return {motor: self.position_i_for_motor(motor)
                for motor in (*self.joint_names, self.gripper_name)}

    def plant_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "domain": "real",
            "hardware": "SO101 follower 12V",
            "joint_names": list(self.joint_names),
            "auxiliary_joints": [self.gripper_name],
            "calibration_sha256": self.calibration_sha256,
            "hardware_config_sha256": self.config_sha256,
            "control_dt": self.control_dt,
            "pid": [self.position_p, self.position_i, self.position_d],
            "pid_i_by_motor": self.effective_position_i_by_motor(),
            "acceleration": self.acceleration,
            "maximum_acceleration": self.maximum_acceleration,
            "state_estimator": "backward_difference_causal_lpf_4hz_v1",
            "lerobot_commit": self.lerobot_commit,
            "voltage_warning": [self.voltage_warning_low, self.voltage_warning_high],
            "voltage_hard": [self.voltage_hard_low, self.voltage_hard_high],
            "hardware_startup_envelope_configured": self.hardware_joint_low is not None,
            "shutdown_recovery_raw": (None if self.shutdown_recovery_raw is None
                                      else self.shutdown_recovery_raw.astype(int).tolist()),
            "shutdown_recovery_seconds": self.shutdown_recovery_seconds,
        }


def load_hardware_config(path: str | Path) -> SO101HardwareConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("hardware config must be a mapping")
    names = tuple(payload.get("joint_names", ()))
    expected_names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    if names != expected_names:
        raise ValueError(f"SO101 controlled joint_names must be exactly {expected_names}")
    calibration = Path(payload["calibration_path"]).expanduser()
    if not calibration.is_absolute():
        calibration = (config_path.parent / calibration).resolve()
    n = len(names)
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    config_hash = _hash_payload(canonical)
    hardware_low_value = payload.get("hardware_joint_low")
    hardware_high_value = payload.get("hardware_joint_high")
    if (hardware_low_value is None) != (hardware_high_value is None):
        raise ValueError("hardware_joint_low and hardware_joint_high must be set together")
    hardware_low = None if hardware_low_value is None else _array(payload, "hardware_joint_low", n)
    hardware_high = None if hardware_high_value is None else _array(payload, "hardware_joint_high", n)
    i_overrides_value = payload.get("position_i_by_motor")
    i_overrides: dict[str, int] | None = None
    if i_overrides_value is not None:
        if not isinstance(i_overrides_value, dict):
            raise ValueError("position_i_by_motor must be a motor-to-integer mapping")
        allowed_motors = set((*names, str(payload.get("gripper_name", "gripper"))))
        unknown_motors = set(i_overrides_value) - allowed_motors
        if unknown_motors:
            raise ValueError(f"position_i_by_motor has unknown motors: {sorted(unknown_motors)}")
        if any(isinstance(value, bool) or int(value) != value or not 0 <= int(value) <= 254
               for value in i_overrides_value.values()):
            raise ValueError("position_i_by_motor values must be integer Feetech coefficients in [0, 254]")
        i_overrides = {str(motor): int(value) for motor, value in i_overrides_value.items()}
    recovery_raw_value = payload.get("shutdown_recovery_raw")
    recovery_raw = (None if recovery_raw_value is None
                    else _array(payload, "shutdown_recovery_raw", n + 1, np.int64))
    result = SO101HardwareConfig(
        port=str(payload["port"]), robot_id=str(payload["robot_id"]),
        calibration_path=calibration, calibration_sha256=str(payload["calibration_sha256"]),
        control_dt=float(payload.get("control_dt", 1 / 30)), joint_names=names,
        gripper_name=str(payload.get("gripper_name", "gripper")),
        joint_low=_array(payload, "joint_low", n), joint_high=_array(payload, "joint_high", n),
        hardware_joint_low=hardware_low, hardware_joint_high=hardware_high,
        raw_low=_array(payload, "raw_low", n + 1, np.int64), raw_high=_array(payload, "raw_high", n + 1, np.int64),
        home_q_ctrl=_array(payload, "home_q_ctrl", n),
        command_velocity_limit=_array(payload, "command_velocity_limit", n),
        command_acceleration_limit=_array(payload, "command_acceleration_limit", n),
        relative_target_limit=_array(payload, "relative_target_limit", n),
        measured_velocity_plausible=_array(payload, "measured_velocity_plausible", n),
        measured_velocity_emergency=_array(payload, "measured_velocity_emergency", n),
        position_p=int(payload.get("position_p", 16)), position_i=int(payload.get("position_i", 0)),
        position_d=int(payload.get("position_d", 32)), acceleration=int(payload.get("acceleration", 32)),
        maximum_acceleration=int(payload.get("maximum_acceleration", 32)),
        return_delay_time=int(payload.get("return_delay_time", 0)),
        gripper_hold_raw=None if payload.get("gripper_hold_raw") is None else int(payload["gripper_hold_raw"]),
        read_retries=int(payload.get("read_retries", 2)),
        diagnostic_rate_hz=float(payload.get("diagnostic_rate_hz", 2.0)),
        goal_readback_rate_hz=float(payload.get("goal_readback_rate_hz", 2.0)),
        lerobot_commit=str(payload["lerobot_commit"]), config_sha256=config_hash,
        voltage_warning_low=None if payload.get("voltage_warning_low") is None else float(payload["voltage_warning_low"]),
        voltage_warning_high=None if payload.get("voltage_warning_high") is None else float(payload["voltage_warning_high"]),
        voltage_hard_low=None if payload.get("voltage_hard_low") is None else float(payload["voltage_hard_low"]),
        voltage_hard_high=None if payload.get("voltage_hard_high") is None else float(payload["voltage_hard_high"]),
        position_i_by_motor=i_overrides,
        shutdown_recovery_raw=recovery_raw,
        shutdown_recovery_seconds=float(payload.get("shutdown_recovery_seconds", 10.0)),
    )
    if result.control_dt <= 0 or result.read_retries < 0 or result.diagnostic_rate_hz <= 0 or result.goal_readback_rate_hz <= 0:
        raise ValueError("control period, diagnostic rates, and retry count are invalid")
    if np.any(result.joint_low >= result.joint_high) or np.any(result.home_q_ctrl < result.joint_low) or np.any(result.home_q_ctrl > result.joint_high):
        raise ValueError("home_q_ctrl must lie inside non-empty joint limits")
    if result.hardware_joint_low is not None:
        assert result.hardware_joint_high is not None
        if (np.any(result.hardware_joint_low >= result.hardware_joint_high)
                or np.any(result.home_q_ctrl < result.hardware_joint_low)
                or np.any(result.home_q_ctrl > result.hardware_joint_high)):
            raise ValueError("home_q_ctrl must lie inside non-empty hardware joint limits")
        if (np.any(result.joint_low < result.hardware_joint_low)
                or np.any(result.joint_high > result.hardware_joint_high)):
            raise ValueError("experiment joint limits must lie inside hardware joint limits")
    if np.any(result.raw_low < 0) or np.any(result.raw_high > 4095) or np.any(result.raw_low >= result.raw_high):
        raise ValueError("raw limits must be ordered inside [0, 4095]")
    if result.gripper_hold_raw is not None and not (result.raw_low[-1] <= result.gripper_hold_raw <= result.raw_high[-1]):
        raise ValueError("gripper_hold_raw must lie inside the gripper raw safety range")
    if result.shutdown_recovery_raw is not None and (
            np.any(result.shutdown_recovery_raw < result.raw_low)
            or np.any(result.shutdown_recovery_raw > result.raw_high)):
        raise ValueError("shutdown_recovery_raw must lie inside the configured raw safety range")
    if not np.isfinite(result.shutdown_recovery_seconds) or result.shutdown_recovery_seconds <= 0:
        raise ValueError("shutdown_recovery_seconds must be positive")
    if np.any(result.command_velocity_limit <= 0) or np.any(result.command_acceleration_limit <= 0) or np.any(result.relative_target_limit <= 0):
        raise ValueError("command limits must be positive")
    if not (0 <= result.acceleration <= 254 and 0 <= result.maximum_acceleration <= 254):
        raise ValueError("Feetech acceleration registers must be in [0, 254]")
    voltages = (result.voltage_warning_low, result.voltage_warning_high, result.voltage_hard_low, result.voltage_hard_high)
    if any(value is not None for value in voltages):
        if any(value is None for value in voltages):
            raise ValueError("all four voltage thresholds must be set together")
        assert all(value is not None for value in voltages)
        if not (result.voltage_hard_low < result.voltage_warning_low < result.voltage_warning_high < result.voltage_hard_high):
            raise ValueError("voltage thresholds must satisfy hard_low < warning_low < warning_high < hard_high")
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
