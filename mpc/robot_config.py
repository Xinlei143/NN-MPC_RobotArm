"""Robot-specific configuration and MuJoCo model contract validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import mujoco
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_CONFIG = PROJECT_ROOT / "configs" / "robots" / "abb_irb2400.yaml"


def _vector(
    payload: Mapping[str, Any],
    key: str,
    n_joints: int,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Robot config is missing required field {key!r}")
    value = np.asarray(payload[key], dtype=np.float64)
    if value.shape != (n_joints,):
        raise ValueError(f"{key} must have shape ({n_joints},), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain finite values")
    if positive and np.any(value <= 0.0):
        raise ValueError(f"{key} must contain positive values")
    if nonnegative and np.any(value < 0.0):
        raise ValueError(f"{key} must contain non-negative values")
    return value.astype(np.float32)


def _names(payload: Mapping[str, Any], key: str, n_joints: int) -> tuple[str, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list) or len(raw) != n_joints:
        raise ValueError(f"{key} must contain exactly {n_joints} names")
    names = tuple(str(value) for value in raw)
    if any(not value for value in names) or len(set(names)) != len(names):
        raise ValueError(f"{key} must contain unique non-empty names")
    return names


def _resolve_config_path(path: str | Path | None) -> Path:
    if path is None:
        return DEFAULT_ROBOT_CONFIG
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    config_path: Path
    model_xml: Path
    model_xml_source: str
    n_joints: int
    joint_names: tuple[str, ...]
    actuator_names: tuple[str, ...]
    ee_site_name: str
    home_q: np.ndarray
    frame_skip: int
    expected_control_dt: float
    gravity_compensation: bool
    gravity_compensation_zero_indices: tuple[int, ...]
    command_velocity_limit: np.ndarray
    command_acceleration_limit: np.ndarray
    state_velocity_limit: np.ndarray
    residual_max: np.ndarray
    servo_scale: np.ndarray
    data_reset_low: np.ndarray
    data_reset_high: np.ndarray
    data_target_low: np.ndarray
    data_target_high: np.ndarray
    data_joint_limit_margin: float
    contact_semantics: str
    source: dict[str, Any]
    legacy_artifacts: dict[str, tuple[str, ...]]
    spec_sha256: str
    model_xml_sha256: str

    def with_runtime_overrides(
        self,
        *,
        model_xml: str | Path | None = None,
        ee_site_name: str | None = None,
        home_q: np.ndarray | None = None,
    ) -> "RobotSpec":
        """Return an effective spec for explicit legacy CLI overrides."""

        resolved_model = self.model_xml
        model_source = self.model_xml_source
        if model_xml is not None:
            candidate = Path(model_xml).expanduser()
            if not candidate.is_absolute():
                root_candidate = PROJECT_ROOT / candidate
                dynamics_candidate = PROJECT_ROOT / "dynamics_modeling" / candidate
                candidate = root_candidate if root_candidate.exists() else dynamics_candidate
            resolved_model = candidate.resolve()
            model_source = str(candidate)
        effective_home = self.home_q if home_q is None else np.asarray(home_q, dtype=np.float32)
        if effective_home.shape != (self.n_joints,):
            raise ValueError(f"home_q must have shape ({self.n_joints},), got {effective_home.shape}")
        effective_site = self.ee_site_name if ee_site_name is None else str(ee_site_name)
        identity = self.identity_payload(
            model_xml_source=model_source,
            model_xml_sha256=file_sha256(resolved_model),
            ee_site_name=effective_site,
            home_q=effective_home,
        )
        payload = self.canonical_payload(identity=identity)
        return replace(
            self,
            model_xml=resolved_model,
            model_xml_source=model_source,
            model_xml_sha256=identity["model_xml_sha256"],
            ee_site_name=effective_site,
            home_q=effective_home.astype(np.float32),
            spec_sha256=hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
        )

    def identity_payload(
        self,
        *,
        model_xml_source: str | None = None,
        model_xml_sha256: str | None = None,
        ee_site_name: str | None = None,
        home_q: np.ndarray | None = None,
    ) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "model_xml": self.model_xml_source if model_xml_source is None else model_xml_source,
            "model_xml_sha256": self.model_xml_sha256 if model_xml_sha256 is None else model_xml_sha256,
            "n_joints": self.n_joints,
            "joint_names": list(self.joint_names),
            "actuator_names": list(self.actuator_names),
            "ee_site_name": self.ee_site_name if ee_site_name is None else ee_site_name,
            "home_q": (
                self.home_q.tolist()
                if home_q is None
                else np.asarray(home_q, dtype=np.float32).tolist()
            ),
            "frame_skip": self.frame_skip,
            "expected_control_dt": self.expected_control_dt,
            "gravity_compensation": self.gravity_compensation,
            "gravity_compensation_zero_indices": list(self.gravity_compensation_zero_indices),
            "contact_semantics": self.contact_semantics,
        }

    def artifact_identity(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["robot_spec_sha256"] = self.spec_sha256
        return payload

    def canonical_payload(
        self,
        *,
        identity: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return every behavior-defining profile field used for the spec hash."""

        return {
            "identity": dict(self.identity_payload() if identity is None else identity),
            "command_velocity_limit": self.command_velocity_limit.tolist(),
            "command_acceleration_limit": self.command_acceleration_limit.tolist(),
            "state_velocity_limit": self.state_velocity_limit.tolist(),
            "residual_max": self.residual_max.tolist(),
            "servo_scale": self.servo_scale.tolist(),
            "data_reset_low": self.data_reset_low.tolist(),
            "data_reset_high": self.data_reset_high.tolist(),
            "data_target_low": self.data_target_low.tolist(),
            "data_target_high": self.data_target_high.tolist(),
            "data_joint_limit_margin": self.data_joint_limit_margin,
            "source": self.source,
        }

    def collection_bounds(self, joint_low: np.ndarray, joint_high: np.ndarray) -> dict[str, np.ndarray]:
        low = np.asarray(joint_low, dtype=np.float32)
        high = np.asarray(joint_high, dtype=np.float32)
        margin = np.float32(self.data_joint_limit_margin)
        reset_low = np.maximum(self.data_reset_low, low + margin)
        reset_high = np.minimum(self.data_reset_high, high - margin)
        target_low = np.maximum(self.data_target_low, low + margin)
        target_high = np.minimum(self.data_target_high, high - margin)
        if np.any(reset_low >= reset_high) or np.any(target_low >= target_high):
            raise ValueError("Robot collection workspace is empty after applying joint-limit margin")
        return {
            "reset_low": reset_low,
            "reset_high": reset_high,
            "target_low": target_low,
            "target_high": target_high,
        }

    def is_legacy_artifact_allowed(self, kind: str, sha256: str) -> bool:
        return sha256 in self.legacy_artifacts.get(kind, ())


def load_robot_spec(
    path: str | Path | None = None,
    *,
    validate_model: bool = False,
) -> RobotSpec:
    config_path = _resolve_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Robot config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Robot config must contain a mapping: {config_path}")

    robot_id = str(payload.get("robot_id", "")).strip()
    if not robot_id:
        raise ValueError("robot_id must be non-empty")
    n_joints = int(payload.get("n_joints", 0))
    if n_joints <= 0:
        raise ValueError("n_joints must be positive")
    model_source = str(payload.get("model_xml", "")).strip()
    if not model_source:
        raise ValueError("model_xml must be non-empty")
    model_xml = Path(model_source).expanduser()
    if not model_xml.is_absolute():
        model_xml = PROJECT_ROOT / model_xml
    model_xml = model_xml.resolve()
    if not model_xml.is_file():
        raise FileNotFoundError(f"Robot model XML does not exist: {model_xml}")

    home = _vector(payload, "home_q", n_joints)
    reset_low_delta = _vector(
        payload,
        "data_reset_low_delta",
        n_joints,
        nonnegative=True,
    ) if "data_reset_low_delta" in payload else _vector(
        payload, "data_reset_delta", n_joints, nonnegative=True
    )
    reset_high_delta = _vector(
        payload,
        "data_reset_high_delta",
        n_joints,
        nonnegative=True,
    ) if "data_reset_high_delta" in payload else reset_low_delta.copy()
    target_low_delta = _vector(
        payload,
        "data_target_low_delta",
        n_joints,
        nonnegative=True,
    ) if "data_target_low_delta" in payload else _vector(
        payload, "data_target_delta", n_joints, nonnegative=True
    )
    target_high_delta = _vector(
        payload,
        "data_target_high_delta",
        n_joints,
        nonnegative=True,
    ) if "data_target_high_delta" in payload else target_low_delta.copy()

    zero_indices = tuple(int(value) for value in payload.get("gravity_compensation_zero_indices", []))
    if len(set(zero_indices)) != len(zero_indices) or any(
        value < 0 or value >= n_joints for value in zero_indices
    ):
        raise ValueError("gravity_compensation_zero_indices contains invalid indices")
    frame_skip = int(payload.get("frame_skip", 5))
    expected_control_dt = float(payload.get("expected_control_dt", 0.01))
    margin = float(payload.get("data_joint_limit_margin", 0.05))
    if frame_skip <= 0 or expected_control_dt <= 0.0 or margin < 0.0:
        raise ValueError("frame_skip/control_dt must be positive and data margin non-negative")

    legacy_raw = payload.get("legacy_artifacts", {})
    if not isinstance(legacy_raw, dict):
        raise ValueError("legacy_artifacts must be a mapping")
    legacy = {
        str(kind): tuple(str(value) for value in values)
        for kind, values in legacy_raw.items()
    }
    source = payload.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("source must be a mapping")
    model_hash = file_sha256(model_xml)

    provisional = RobotSpec(
        robot_id=robot_id,
        config_path=config_path,
        model_xml=model_xml,
        model_xml_source=model_source,
        n_joints=n_joints,
        joint_names=_names(payload, "joint_names", n_joints),
        actuator_names=_names(payload, "actuator_names", n_joints),
        ee_site_name=str(payload.get("ee_site_name", "")).strip(),
        home_q=home,
        frame_skip=frame_skip,
        expected_control_dt=expected_control_dt,
        gravity_compensation=bool(payload.get("gravity_compensation", True)),
        gravity_compensation_zero_indices=zero_indices,
        command_velocity_limit=_vector(payload, "command_velocity_limit", n_joints, positive=True),
        command_acceleration_limit=_vector(payload, "command_acceleration_limit", n_joints, positive=True),
        state_velocity_limit=_vector(payload, "state_velocity_limit", n_joints, positive=True),
        residual_max=_vector(payload, "residual_max", n_joints, positive=True),
        servo_scale=_vector(payload, "servo_scale", n_joints, positive=True),
        data_reset_low=home - reset_low_delta,
        data_reset_high=home + reset_high_delta,
        data_target_low=home - target_low_delta,
        data_target_high=home + target_high_delta,
        data_joint_limit_margin=margin,
        contact_semantics=str(payload.get("contact_semantics", "unspecified")),
        source=dict(source),
        legacy_artifacts=legacy,
        spec_sha256="",
        model_xml_sha256=model_hash,
    )
    if not provisional.ee_site_name:
        raise ValueError("ee_site_name must be non-empty")
    canonical = provisional.canonical_payload()
    spec = replace(
        provisional,
        spec_sha256=hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest(),
    )
    if validate_model:
        validate_robot_model_contract(spec)
    return spec


def validate_robot_model_contract(spec: RobotSpec) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(spec.model_xml))
    if model.nq < spec.n_joints or model.nv < spec.n_joints or model.nu < spec.n_joints:
        raise ValueError("Robot model does not expose the configured controlled prefix")

    actual_joints: list[str] = []
    actual_actuators: list[str] = []
    for index in range(spec.n_joints):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        actual_joints.append("" if joint_name is None else joint_name)
        actual_actuators.append("" if actuator_name is None else actuator_name)
        if int(model.jnt_type[index]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Controlled joint {index} is not a hinge")
        if int(model.jnt_qposadr[index]) != index or int(model.jnt_dofadr[index]) != index:
            raise ValueError("Controlled joints must occupy the contiguous qpos/qvel prefix")
        if int(model.actuator_trnid[index, 0]) != index:
            raise ValueError("Controlled actuators must target the matching joint prefix")
    if tuple(actual_joints) != spec.joint_names:
        raise ValueError(f"Joint order mismatch: model={actual_joints}, config={spec.joint_names}")
    if tuple(actual_actuators) != spec.actuator_names:
        raise ValueError(f"Actuator order mismatch: model={actual_actuators}, config={spec.actuator_names}")

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, spec.ee_site_name)
    if site_id < 0:
        raise ValueError(f"Robot model is missing TCP site {spec.ee_site_name!r}")
    joint_low = np.asarray(model.jnt_range[: spec.n_joints, 0], dtype=np.float32)
    joint_high = np.asarray(model.jnt_range[: spec.n_joints, 1], dtype=np.float32)
    ctrl_low = np.asarray(model.actuator_ctrlrange[: spec.n_joints, 0], dtype=np.float32)
    ctrl_high = np.asarray(model.actuator_ctrlrange[: spec.n_joints, 1], dtype=np.float32)
    if (
        not np.all(np.isfinite(joint_low))
        or not np.all(np.isfinite(joint_high))
        or np.any(spec.home_q < joint_low)
        or np.any(spec.home_q > joint_high)
        or np.any(spec.home_q < ctrl_low)
        or np.any(spec.home_q > ctrl_high)
    ):
        raise ValueError("Configured home is outside finite joint or actuator limits")
    control_dt = float(model.opt.timestep * spec.frame_skip)
    if not np.isclose(control_dt, spec.expected_control_dt, atol=1e-12, rtol=0.0):
        raise ValueError(
            f"Control period mismatch: XML timestep*frame_skip={control_dt}, "
            f"expected {spec.expected_control_dt}"
        )
    spec.collection_bounds(joint_low, joint_high)
    kp = np.asarray(model.actuator_gainprm[: spec.n_joints, 0], dtype=np.float64)
    kd = -np.asarray(model.actuator_biasprm[: spec.n_joints, 2], dtype=np.float64)
    if np.any(kp <= 0.0) or np.any(kd < 0.0):
        raise ValueError("Configured actuators are not finite positive-gain position servos")
    if spec.contact_semantics == "disabled":
        active = np.asarray(model.geom_contype) | np.asarray(model.geom_conaffinity)
        if np.any(active):
            raise ValueError("Robot profile declares disabled contacts but active contact geoms exist")
    return {
        "robot_id": spec.robot_id,
        "robot_spec_sha256": spec.spec_sha256,
        "model_xml_sha256": spec.model_xml_sha256,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "control_dt": control_dt,
        "ee_site_name": spec.ee_site_name,
        "home_q": spec.home_q.tolist(),
        "position_kp": kp.tolist(),
        "position_kd": kd.tolist(),
        "contact_semantics": spec.contact_semantics,
    }
