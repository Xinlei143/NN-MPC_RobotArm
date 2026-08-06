"""Conservative table-clearance checks for the real SO101 workspace collector.

The table is modelled as a plane in the MuJoCo base frame.  This is a
geometric gate, not a contact simulation: every selected link capsule and
probe must remain above the required *physical* clearance after accounting
for calibration/model uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np
import yaml

from robot_runtime.kinematics import JointCoordinateMapping


PROFILE_SCHEMA_VERSION = 1
# The base/shoulder pedestal is intentionally seated on the tabletop and is
# static with respect to it.  The 50 mm moving-clearance rule applies to the
# articulated links and tool; pedestal seating is verified separately.
DEFAULT_SAFETY_BODIES = ("upper_arm", "forearm", "wrist", "tool", "gripper")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_bundle_sha256(model_xml: str | Path) -> str:
    """Hash a self-contained MuJoCo model directory, including mesh assets.

    An XML-only hash is insufficient for a mesh collision model: changing an
    STL can change its table clearance while leaving the XML untouched.  The
    detailed SO101 model is intentionally kept in its own directory, so this
    directory-level identity is both explicit and inexpensive to calculate.
    """
    root = Path(model_xml).expanduser().resolve().parent
    digest = hashlib.sha256()
    for path in sorted(
        candidate for candidate in root.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".xml", ".stl"}
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class TableSafetyProfile:
    path: Path
    sha256: str
    status: str
    table_normal: np.ndarray
    table_offset_m: float
    minimum_clearance_m: float
    offline_minimum_clearance_m: float
    calibrated_overprediction_m: float
    static_margin_m: float
    geometry_padding_m: float
    mapping: JointCoordinateMapping
    safety_body_names: tuple[str, ...]
    table_probe_body: str
    table_probe_local_position_m: np.ndarray
    table_probe_radius_m: float
    tcp_center_body: str
    tcp_center_local_position_m: np.ndarray
    tcp_center_minimum_height_m: float
    enforce_tcp_center: bool
    base_roll_deg: float
    base_pitch_deg: float
    model_xml_sha256: str
    mesh_collision_model_xml: Path | None
    mesh_collision_bundle_sha256: str | None
    mesh_collision_floor_geom: str | None
    mesh_collision_joint_names: tuple[str, ...]

    @property
    def effective_prediction_threshold_m(self) -> float:
        return self.minimum_clearance_m + self.calibrated_overprediction_m + self.static_margin_m

    def for_offline_planning(self) -> "TableSafetyProfile":
        """Return the stricter profile used for MuJoCo reference screening."""
        return replace(self, minimum_clearance_m=self.offline_minimum_clearance_m)

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "path": str(self.path),
            "sha256": self.sha256,
            "minimum_clearance_m": self.minimum_clearance_m,
            "offline_minimum_clearance_m": self.offline_minimum_clearance_m,
            "effective_prediction_threshold_m": self.effective_prediction_threshold_m,
            "table_plane": {"normal": self.table_normal.tolist(), "offset_m": self.table_offset_m},
            "tcp_center_minimum_height_m": self.tcp_center_minimum_height_m,
            "enforce_tcp_center": self.enforce_tcp_center,
            "mapping": self.mapping.identity(),
            "mesh_collision": None if self.mesh_collision_model_xml is None else {
                "model_xml": str(self.mesh_collision_model_xml),
                "bundle_sha256": self.mesh_collision_bundle_sha256,
                "floor_geom": self.mesh_collision_floor_geom,
                "joint_names": list(self.mesh_collision_joint_names),
            },
        }


def load_table_safety_profile(path: str | Path, *, n_joints: int, expected_model_xml_sha256: str,
                              require_approved: bool = True) -> TableSafetyProfile:
    profile_path = Path(path).expanduser().resolve()
    with profile_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("table safety profile must be a mapping")
    if int(payload.get("schema_version", -1)) != PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported table safety profile schema_version")
    status = str(payload.get("status", "draft"))
    if require_approved and status != "approved":
        raise ValueError("table safety profile must have status: approved")
    tcp_calibration = payload.get("tcp_probe_calibration")
    verification_mode = str(payload.get("verification_mode", ""))
    fitted_tcp = isinstance(tcp_calibration, dict) and tcp_calibration.get("status") == "fitted_from_known_height_contacts"
    assumed_tcp = verification_mode == "assumed_level_base_model_tcp_center"
    assumed_mesh = verification_mode == "assumed_level_base_mesh_collision"
    if require_approved and not (fitted_tcp or assumed_tcp or assumed_mesh):
        raise ValueError("approved table safety profile requires a fitted registration or explicit model assumption")
    plane = payload.get("table_plane")
    if not isinstance(plane, dict):
        raise ValueError("table_plane must be a mapping")
    normal = _vector(plane.get("normal"), 3, "table_plane.normal")
    normal_norm = float(np.linalg.norm(normal))
    if not np.isclose(normal_norm, 1.0, atol=1e-6):
        raise ValueError("table_plane.normal must be unit length")
    offset = float(plane.get("offset_m", np.nan))
    if not np.isfinite(offset):
        raise ValueError("table_plane.offset_m must be finite")
    clearance = float(payload.get("minimum_clearance_m", np.nan))
    offline_clearance = float(payload.get("offline_minimum_clearance_m", clearance))
    overprediction = float(payload.get("calibrated_overprediction_m", np.nan))
    static_margin = float(payload.get("static_margin_m", np.nan))
    geometry_padding = float(payload.get("geometry_padding_m", np.nan))
    # Runtime clearance can be negative only relative to a deliberately
    # raised virtual guard.  Keep that exception bounded; offline screening
    # remains non-negative and is normally stricter.
    if (not np.isfinite(clearance) or clearance < -0.030
            or not np.isfinite(offline_clearance) or offline_clearance < 0
            or not np.isfinite(overprediction) or overprediction < 0
            or not np.isfinite(static_margin) or static_margin < 0 or not np.isfinite(geometry_padding)
            or geometry_padding < 0):
        raise ValueError("minimum_clearance_m must be finite and >= -0.030 m; offline_minimum_clearance_m and other table margins must be finite non-negative values")
    mapping_raw = payload.get("q_ctrl_to_q_kin")
    if not isinstance(mapping_raw, dict):
        raise ValueError("q_ctrl_to_q_kin must be a mapping")
    sign = _vector(mapping_raw.get("joint_sign"), n_joints, "q_ctrl_to_q_kin.joint_sign")
    offset_q = _vector(mapping_raw.get("joint_offset"), n_joints, "q_ctrl_to_q_kin.joint_offset")
    if not np.all(np.isin(sign, (-1.0, 1.0))):
        raise ValueError("q_ctrl_to_q_kin.joint_sign must contain only -1 or 1")
    mapping = JointCoordinateMapping(sign.astype(np.float32), offset_q.astype(np.float32),
                                    str(mapping_raw.get("source", "unverified")))
    bodies_raw = payload.get("safety_body_names", DEFAULT_SAFETY_BODIES)
    if not isinstance(bodies_raw, list) or not bodies_raw:
        raise ValueError("safety_body_names must be a non-empty list")
    bodies = tuple(str(value) for value in bodies_raw)
    if any(not value for value in bodies) or len(set(bodies)) != len(bodies):
        raise ValueError("safety_body_names must contain unique non-empty values")
    probe = payload.get("table_probe")
    if not isinstance(probe, dict):
        raise ValueError("table_probe must be a mapping")
    probe_body = str(probe.get("body", ""))
    probe_position = _vector(probe.get("local_position_m"), 3, "table_probe.local_position_m")
    probe_radius = float(probe.get("radius_m", np.nan))
    if not probe_body or not np.isfinite(probe_radius) or probe_radius < 0:
        raise ValueError("table_probe body/radius is invalid")
    tcp_center = payload.get("tcp_center")
    if not isinstance(tcp_center, dict):
        raise ValueError("tcp_center must be a mapping")
    tcp_body = str(tcp_center.get("body", ""))
    tcp_position = _vector(tcp_center.get("local_position_m"), 3, "tcp_center.local_position_m")
    tcp_minimum = float(tcp_center.get("minimum_height_m", np.nan))
    enforce_tcp = bool(tcp_center.get("enabled", True))
    if not tcp_body or not np.isfinite(tcp_minimum) or tcp_minimum < 0:
        raise ValueError("tcp_center body/minimum_height_m is invalid")
    level = payload.get("base_level", {})
    if not isinstance(level, dict):
        raise ValueError("base_level must be a mapping")
    roll, pitch = float(level.get("roll_deg", np.nan)), float(level.get("pitch_deg", np.nan))
    if not np.isfinite(roll) or not np.isfinite(pitch) or abs(roll) > .5 or abs(pitch) > .5:
        raise ValueError("base_level roll/pitch must be measured within +/-0.5 deg")
    model_hash = str(payload.get("model_xml_sha256", ""))
    if model_hash != expected_model_xml_sha256:
        raise ValueError("table safety profile model_xml_sha256 does not match current SO101 model")
    mesh_raw = payload.get("mesh_collision")
    mesh_xml: Path | None = None
    mesh_hash: str | None = None
    mesh_floor: str | None = None
    mesh_joint_names: tuple[str, ...] = ()
    if mesh_raw is not None:
        if not isinstance(mesh_raw, dict):
            raise ValueError("mesh_collision must be a mapping")
        mesh_source = str(mesh_raw.get("model_xml", "")).strip()
        if not mesh_source:
            raise ValueError("mesh_collision.model_xml must be non-empty")
        mesh_xml = Path(mesh_source).expanduser()
        if not mesh_xml.is_absolute():
            mesh_xml = (Path.cwd() / mesh_xml).resolve()
        if not mesh_xml.is_file():
            raise ValueError(f"mesh collision model does not exist: {mesh_xml}")
        mesh_hash = str(mesh_raw.get("bundle_sha256", ""))
        if not mesh_hash or mesh_hash != model_bundle_sha256(mesh_xml):
            raise ValueError("mesh_collision.bundle_sha256 does not match the detailed model and assets")
        mesh_floor = str(mesh_raw.get("floor_geom", "")).strip()
        if not mesh_floor:
            raise ValueError("mesh_collision.floor_geom must be non-empty")
        names_raw = mesh_raw.get("joint_names")
        if not isinstance(names_raw, list) or len(names_raw) != n_joints:
            raise ValueError("mesh_collision.joint_names must contain one name per controlled joint")
        mesh_joint_names = tuple(str(name) for name in names_raw)
        if any(not name for name in mesh_joint_names) or len(set(mesh_joint_names)) != len(mesh_joint_names):
            raise ValueError("mesh_collision.joint_names must be unique non-empty names")
    return TableSafetyProfile(
        profile_path, _sha256(profile_path), status, normal, offset, clearance, offline_clearance, overprediction, static_margin,
        geometry_padding, mapping, bodies, probe_body, probe_position, probe_radius,
        tcp_body, tcp_position, tcp_minimum, enforce_tcp, roll, pitch, model_hash,
        mesh_xml, mesh_hash, mesh_floor, mesh_joint_names,
    )


@dataclass(frozen=True)
class ClearanceResult:
    predicted_clearance_m: float
    effective_clearance_m: float
    nearest_component: str
    component_clearances_m: dict[str, float]
    tcp_center_height_m: float
    tcp_center_minimum_height_m: float
    safe: bool


class TableClearanceChecker:
    """Private-MjData table gate; safe to call from an offline planning thread."""

    def __init__(self, model_xml: str | Path, profile: TableSafetyProfile, n_joints: int = 5,
                 *, gripper_model_angle_rad: float | None = None):
        self.model_xml = Path(model_xml).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.model_xml))
        self.data = mujoco.MjData(self.model)
        self.profile, self.n_joints = profile, int(n_joints)
        self._qpos_addresses = np.asarray(self.model.jnt_qposadr[:self.n_joints], dtype=np.intp)
        self._gripper_angle: float | None = gripper_model_angle_rad
        self._gripper_qposadr: int | None = None
        if gripper_model_angle_rad is not None:
            gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
            if gripper_id >= 0:
                self._gripper_qposadr = int(self.model.jnt_qposadr[gripper_id])
        self._body_ids = {name: self._body_id(name) for name in profile.safety_body_names}
        self._probe_body_id = self._body_id(profile.table_probe_body)
        self._tcp_center_body_id = self._body_id(profile.tcp_center_body)
        self._geometry_by_body: dict[str, tuple[int, ...]] = {}
        for name, body_id in self._body_ids.items():
            geom_ids = tuple(int(index) for index in np.flatnonzero(self.model.geom_bodyid == body_id))
            if not geom_ids:
                raise ValueError(f"table safety body {name!r} has no MuJoCo geometry")
            self._geometry_by_body[name] = geom_ids

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model has no body {name!r} required by table safety profile")
        return int(body_id)

    def body_world_pose(self, q_ctrl: np.ndarray, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the selected body's position and rotation in the base frame.

        This is deliberately exposed for the read-only tabletop calibration
        tool.  It lets that tool identify an actual physical TCP point from
        repeated contacts with a known-height gauge, instead of asking an
        operator to know a MuJoCo-local coordinate.
        """
        body_id = self._body_id(body_name)
        self._forward(q_ctrl)
        position = np.asarray(self.data.xpos[body_id], dtype=np.float64).copy()
        rotation = np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()
        return position, rotation

    def _forward(self, q_ctrl: np.ndarray) -> None:
        values = np.asarray(q_ctrl, dtype=np.float64)
        if values.shape != (self.n_joints,) or not np.all(np.isfinite(values)):
            raise ValueError(f"q_ctrl must have {self.n_joints} finite values")
        q_kin = self.profile.mapping.to_kinematics(values)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_addresses] = q_kin
        if self._gripper_qposadr is not None and self._gripper_angle is not None:
            self.data.qpos[self._gripper_qposadr] = self._gripper_angle
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _signed_plane_distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.profile.table_normal - self.profile.table_offset_m

    def _geom_clearance(self, geom_id: int) -> float:
        kind = int(self.model.geom_type[geom_id])
        center = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        rotation = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        radius = float(self.model.geom_size[geom_id, 0]) + self.profile.geometry_padding_m
        # Capsules and cylinders are locally aligned with z.  Checking both
        # endpoint spheres is exact for a capsule and conservative for a
        # cylinder; other primitive types use a conservative bounding sphere.
        if kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            half_length = float(self.model.geom_size[geom_id, 1])
            axis = rotation[:, 2]
            points = np.vstack((center - half_length * axis, center + half_length * axis))
            return float(np.min(self._signed_plane_distance(points)) - radius)
        bounding_radius = radius + float(np.linalg.norm(self.model.geom_size[geom_id, 1:]))
        return float(self._signed_plane_distance(center[None, :])[0] - bounding_radius)

    def evaluate(self, q_ctrl: np.ndarray) -> ClearanceResult:
        self._forward(q_ctrl)
        components: dict[str, float] = {}
        for body_name, geom_ids in self._geometry_by_body.items():
            components[body_name] = min(self._geom_clearance(geom_id) for geom_id in geom_ids)
        probe_position = (np.asarray(self.data.xpos[self._probe_body_id], dtype=np.float64)
                          + np.asarray(self.data.xmat[self._probe_body_id], dtype=np.float64).reshape(3, 3)
                          @ self.profile.table_probe_local_position_m)
        components["table_probe"] = float(
            self._signed_plane_distance(probe_position[None, :])[0]
            - self.profile.table_probe_radius_m - self.profile.geometry_padding_m
        )
        tcp_center_position = (np.asarray(self.data.xpos[self._tcp_center_body_id], dtype=np.float64)
                               + np.asarray(self.data.xmat[self._tcp_center_body_id], dtype=np.float64).reshape(3, 3)
                               @ self.profile.tcp_center_local_position_m)
        tcp_center_height = float(self._signed_plane_distance(tcp_center_position[None, :])[0])
        nearest = min(components, key=components.__getitem__)
        if self.profile.enforce_tcp_center and tcp_center_height < self.profile.tcp_center_minimum_height_m:
            nearest = "tcp_center"
            components[nearest] = tcp_center_height
        predicted = float(components[nearest])
        effective = (predicted if nearest == "tcp_center"
                     else predicted - self.profile.calibrated_overprediction_m - self.profile.static_margin_m)
        link_safe = all(
            value - self.profile.calibrated_overprediction_m - self.profile.static_margin_m
            >= self.profile.minimum_clearance_m
            for name, value in components.items() if name != "tcp_center"
        )
        tcp_safe = (not self.profile.enforce_tcp_center
                    or tcp_center_height >= self.profile.tcp_center_minimum_height_m)
        return ClearanceResult(predicted, effective, nearest, components, tcp_center_height,
                               self.profile.tcp_center_minimum_height_m, bool(link_safe and tcp_safe))

    def require_safe(self, q_ctrl: np.ndarray) -> ClearanceResult:
        result = self.evaluate(q_ctrl)
        if not result.safe:
            raise ValueError(
                f"table clearance violation: component={result.nearest_component}, "
                f"predicted_clearance_m={result.predicted_clearance_m:.4f}, "
                f"effective_clearance_m={result.effective_clearance_m:.4f}, "
                f"required_m={self.profile.minimum_clearance_m:.4f}, "
                f"tcp_center_height_m={result.tcp_center_height_m:.4f}, "
                f"tcp_center_required_m={result.tcp_center_minimum_height_m:.4f}"
            )
        return result

    def sequence_result(self, sequence: Iterable[np.ndarray]) -> ClearanceResult:
        results = [self.evaluate(q) for q in sequence]
        if not results:
            raise ValueError("table safety sequence cannot be empty")
        # An unsafe TCP-center sample may have a numerically larger height
        # than a separately safe link's 50 mm effective clearance.  Never
        # let that ordering hide an unsafe point from sequence validation.
        unsafe = [item for item in results if not item.safe]
        return min(unsafe or results, key=lambda item: item.effective_clearance_m)

    def require_safe_sequence(self, sequence: Iterable[np.ndarray]) -> ClearanceResult:
        result = self.sequence_result(sequence)
        if not result.safe:
            raise ValueError(
                f"table clearance violation in trajectory: component={result.nearest_component}, "
                f"effective_clearance_m={result.effective_clearance_m:.4f}"
            )
        return result


class MeshTableClearanceChecker:
    """Table-clearance checker using the detailed MuJoCo mesh collision model.

    ``mj_geomDistance`` measures the distance from every selected collision
    mesh to the model's table plane.  MuJoCo uses the mesh convex hull for
    collision queries, which is conservative for a horizontal table.  This
    is deliberately distinct from :class:`TableClearanceChecker`: the latter
    remains the legacy capsule approximation for non-safety simulation work.
    """

    def __init__(self, profile: TableSafetyProfile, n_joints: int = 5,
                 *, gripper_model_angle_rad: float | None = None):
        if profile.mesh_collision_model_xml is None:
            raise ValueError("mesh table checker requires mesh_collision in the table safety profile")
        self.model_xml = profile.mesh_collision_model_xml
        self.model = mujoco.MjModel.from_xml_path(str(self.model_xml))
        self.data = mujoco.MjData(self.model)
        self.profile, self.n_joints = profile, int(n_joints)
        if len(profile.mesh_collision_joint_names) != self.n_joints:
            raise ValueError("detailed collision joint ordering does not match controlled joint count")
        addresses: list[int] = []
        for name in profile.mesh_collision_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"detailed collision model has no joint {name!r}")
            addresses.append(int(self.model.jnt_qposadr[joint_id]))
        self._qpos_addresses = np.asarray(addresses, dtype=np.intp)
        self._gripper_angle: float | None = gripper_model_angle_rad
        self._gripper_qposadr: int | None = None
        if gripper_model_angle_rad is not None:
            gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
            if gripper_id < 0:
                raise ValueError("gripper mapping requires a 'gripper' joint in the detailed model")
            self._gripper_qposadr = int(self.model.jnt_qposadr[gripper_id])
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, profile.mesh_collision_floor_geom)
        if floor_id < 0:
            raise ValueError(f"detailed collision model has no floor geom {profile.mesh_collision_floor_geom!r}")
        self._floor_geom_id = int(floor_id)
        self._body_ids = {name: self._body_id(name) for name in profile.safety_body_names}
        self._probe_body_id = self._body_id(profile.table_probe_body)
        self._tcp_center_body_id = self._body_id(profile.tcp_center_body)
        self._geom_ids_by_body: dict[str, tuple[int, ...]] = {}
        for name, body_id in self._body_ids.items():
            ids = tuple(
                int(index) for index in np.flatnonzero(self.model.geom_bodyid == body_id)
                if int(index) != self._floor_geom_id
                and bool(self.model.geom_contype[index] and self.model.geom_conaffinity[index])
            )
            if not ids:
                raise ValueError(f"detailed collision body {name!r} has no collision geometry")
            self._geom_ids_by_body[name] = ids

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"detailed collision model has no body {name!r}")
        return int(body_id)

    def _forward(self, q_ctrl: np.ndarray) -> None:
        values = np.asarray(q_ctrl, dtype=np.float64)
        if values.shape != (self.n_joints,) or not np.all(np.isfinite(values)):
            raise ValueError(f"q_ctrl must have {self.n_joints} finite values")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_addresses] = self.profile.mapping.to_kinematics(values)
        if self._gripper_qposadr is not None and self._gripper_angle is not None:
            self.data.qpos[self._gripper_qposadr] = self._gripper_angle
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def body_world_pose(self, q_ctrl: np.ndarray, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = self._body_id(body_name)
        self._forward(q_ctrl)
        return (
            np.asarray(self.data.xpos[body_id], dtype=np.float64).copy(),
            np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy(),
        )

    def _floor_distance(self, geom_id: int) -> float:
        # ``distmax`` is only a compute shortcut.  One metre is well above the
        # SO101 workspace and therefore cannot hide a near-table geometry.
        return float(mujoco.mj_geomDistance(
            self.model, self.data, self._floor_geom_id, geom_id, 1.0, np.zeros(6, dtype=np.float64)
        ))

    def evaluate(self, q_ctrl: np.ndarray) -> ClearanceResult:
        self._forward(q_ctrl)
        components: dict[str, float] = {
            name: min(self._floor_distance(geom_id) for geom_id in geom_ids)
            for name, geom_ids in self._geom_ids_by_body.items()
        }
        probe_position = (
            np.asarray(self.data.xpos[self._probe_body_id], dtype=np.float64)
            + np.asarray(self.data.xmat[self._probe_body_id], dtype=np.float64).reshape(3, 3)
            @ self.profile.table_probe_local_position_m
        )
        components["table_probe"] = float(
            probe_position @ self.profile.table_normal - self.profile.table_offset_m
            - self.profile.table_probe_radius_m
        )
        tcp_position = (
            np.asarray(self.data.xpos[self._tcp_center_body_id], dtype=np.float64)
            + np.asarray(self.data.xmat[self._tcp_center_body_id], dtype=np.float64).reshape(3, 3)
            @ self.profile.tcp_center_local_position_m
        )
        tcp_height = float(tcp_position @ self.profile.table_normal - self.profile.table_offset_m)
        buffer = self.profile.geometry_padding_m + self.profile.calibrated_overprediction_m + self.profile.static_margin_m
        link_nearest = min(components, key=components.__getitem__)
        nearest = ("tcp_center" if self.profile.enforce_tcp_center
                   and tcp_height < self.profile.tcp_center_minimum_height_m else link_nearest)
        predicted = tcp_height if nearest == "tcp_center" else float(components[nearest])
        effective = predicted if nearest == "tcp_center" else predicted - buffer
        # A zero threshold means "do not touch the virtual guard plane".  It
        # must be strict: MuJoCo reports a distance of zero at contact.
        def link_passes(value: float) -> bool:
            effective_value = value - buffer
            return (effective_value > 0.0 if self.profile.minimum_clearance_m == 0.0
                    else effective_value >= self.profile.minimum_clearance_m)

        link_safe = all(link_passes(value) for value in components.values())
        tcp_safe = (not self.profile.enforce_tcp_center
                    or tcp_height >= self.profile.tcp_center_minimum_height_m)
        return ClearanceResult(predicted, effective, nearest, components, tcp_height,
                               self.profile.tcp_center_minimum_height_m, bool(link_safe and tcp_safe))

    def require_safe(self, q_ctrl: np.ndarray) -> ClearanceResult:
        result = self.evaluate(q_ctrl)
        if not result.safe:
            raise ValueError(
                f"mesh table clearance violation: component={result.nearest_component}, "
                f"predicted_clearance_m={result.predicted_clearance_m:.4f}, "
                f"effective_clearance_m={result.effective_clearance_m:.4f}, "
                f"required_m={self.profile.minimum_clearance_m:.4f}, "
                f"tcp_center_height_m={result.tcp_center_height_m:.4f}, "
                f"tcp_center_required_m={result.tcp_center_minimum_height_m:.4f}"
            )
        return result

    def sequence_result(self, sequence: Iterable[np.ndarray]) -> ClearanceResult:
        results = [self.evaluate(q) for q in sequence]
        if not results:
            raise ValueError("table safety sequence cannot be empty")
        unsafe = [item for item in results if not item.safe]
        return min(unsafe or results, key=lambda item: item.effective_clearance_m)

    def require_safe_sequence(self, sequence: Iterable[np.ndarray]) -> ClearanceResult:
        result = self.sequence_result(sequence)
        if not result.safe:
            raise ValueError(
                f"mesh table clearance violation in trajectory: component={result.nearest_component}, "
                f"effective_clearance_m={result.effective_clearance_m:.4f}"
            )
        return result


def make_table_clearance_checker(model_xml: str | Path, profile: TableSafetyProfile,
                                 n_joints: int = 5, *,
                                 gripper_model_angle_rad: float | None = None) -> TableClearanceChecker | MeshTableClearanceChecker:
    """Select detailed mesh checking only when the profile explicitly pins it."""
    if profile.mesh_collision_model_xml is not None:
        return MeshTableClearanceChecker(profile, n_joints, gripper_model_angle_rad=gripper_model_angle_rad)
    return TableClearanceChecker(model_xml, profile, n_joints, gripper_model_angle_rad=gripper_model_angle_rad)


def profile_with_finalized_calibration(profile_path: str | Path, captures: list[dict[str, Any]], *,
                                       output_path: str | Path, expected_model_xml_sha256: str,
                                       required_components: Iterable[str] = DEFAULT_SAFETY_BODIES) -> dict[str, Any]:
    """Finalize a draft profile from externally measured clearance captures."""
    source = Path(profile_path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("table safety profile must be a mapping")
    if str(payload.get("model_xml_sha256", "")) != expected_model_xml_sha256:
        raise ValueError("draft table safety profile model hash does not match current SO101 model")
    tcp_calibration = payload.get("tcp_probe_calibration")
    if not isinstance(tcp_calibration, dict) or tcp_calibration.get("status") != "fitted_from_known_height_contacts":
        raise ValueError("finalization requires a draft produced by --fit-probe")
    required = tuple(required_components)
    counts = {name: 0 for name in required}
    errors: list[float] = []
    for capture in captures:
        component = str(capture.get("component", ""))
        if component not in counts:
            continue
        predicted, measured = float(capture["predicted_clearance_m"]), float(capture["measured_clearance_m"])
        if not np.isfinite(predicted) or not np.isfinite(measured) or measured < 0:
            raise ValueError("capture clearance values must be finite and non-negative")
        counts[component] += 1
        errors.append(predicted - measured)
    missing = [name for name, count in counts.items() if count < 2]
    if missing:
        raise ValueError(f"need at least two externally measured captures for each safety component: {missing}")
    maximum_overprediction = max(0.0, max(errors, default=0.0))
    if maximum_overprediction > .010 + 1e-12:
        raise ValueError(f"model overpredicts clearance by {maximum_overprediction * 1000:.1f} mm; limit is 10.0 mm")
    payload["status"] = "approved"
    payload["calibrated_overprediction_m"] = float(maximum_overprediction)
    payload["verification"] = {
        "capture_count": len(captures), "per_component_count": counts,
        "maximum_overprediction_m": float(maximum_overprediction),
        "capture_sha256": hashlib.sha256(json.dumps(captures, sort_keys=True).encode()).hexdigest(),
    }
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload
