from __future__ import annotations

"""MuJoCo TCP kinematics helpers used by offline task-space reference generation.

All stateful calculations use a private ``MjData`` instance.  This keeps FK and
IK work independent from the data object used by the live MuJoCo environment.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


def _site_id(model: mujoco.MjModel, ee_site_name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    if site_id < 0:
        raise ValueError(f"MuJoCo model does not contain TCP site {ee_site_name!r}")
    return int(site_id)


def _controlled_joint_addresses(model: mujoco.MjModel, n_joints: int) -> tuple[np.ndarray, np.ndarray]:
    if n_joints <= 0:
        raise ValueError(f"n_joints must be positive, got {n_joints}")
    if model.njnt < n_joints:
        raise ValueError(f"model has {model.njnt} joints, but n_joints={n_joints}")

    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    for joint_id in range(n_joints):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise ValueError(
                "Task-space IK requires one-DoF hinge or slide controlled joints; "
                f"joint {joint_id} ({joint_name!r}) has MuJoCo type {joint_type}"
            )
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_addresses, dtype=np.intp), np.asarray(dof_addresses, dtype=np.intp)


def controlled_joint_limits(model: mujoco.MjModel, n_joints: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Return hard lower and upper limits for the first controlled joints."""

    _controlled_joint_addresses(model, n_joints)
    limited = np.asarray(model.jnt_limited[:n_joints], dtype=bool)
    if not np.all(limited):
        missing = np.flatnonzero(~limited).tolist()
        raise ValueError(f"Task-space IK requires finite joint limits; joints without limits: {missing}")
    bounds = np.asarray(model.jnt_range[:n_joints], dtype=np.float64)
    low = bounds[:, 0].copy()
    high = bounds[:, 1].copy()
    if np.any(low >= high):
        raise ValueError("MuJoCo joint limits must have lower < upper")
    return low, high


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap angular differences to ``[-pi, pi)`` elementwise."""

    values = np.asarray(angles, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def rotation_log_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle logarithm vector of a 3-by-3 rotation matrix."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must contain only finite values")

    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew_vector = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    )

    if angle < 1e-8:
        return 0.5 * skew_vector
    if np.pi - angle < 1e-5:
        # Near pi, sin(angle) is too small for the standard log-map formula.
        # The eigenvector for eigenvalue one gives a stable rotation axis.
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        axis /= axis_norm
        # Preserve an available sign preference just away from the exact-pi case.
        if float(np.dot(axis, skew_vector)) < 0.0:
            axis = -axis
        return angle * axis
    return (angle / (2.0 * np.sin(angle))) * skew_vector


def orientation_error(target_rotation: np.ndarray, current_rotation: np.ndarray) -> np.ndarray:
    """Return the world-frame orientation error that rotates current to target."""

    target = np.asarray(target_rotation, dtype=np.float64)
    current = np.asarray(current_rotation, dtype=np.float64)
    if target.shape != (3, 3) or current.shape != (3, 3):
        raise ValueError(
            "target_rotation and current_rotation must both have shape (3, 3), "
            f"got {target.shape} and {current.shape}"
        )
    return rotation_log_vector(target @ current.T)


def quintic_time_scaling(tau: np.ndarray | float) -> np.ndarray:
    """Return the zero-velocity/zero-acceleration quintic time scale on [0, 1]."""

    normalized_time = np.asarray(tau, dtype=np.float64)
    return 10.0 * normalized_time**3 - 15.0 * normalized_time**4 + 6.0 * normalized_time**5


def site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_name: str = "ee_site",
) -> tuple[np.ndarray, np.ndarray]:
    """Read a TCP pose from caller-owned, already-forwarded MuJoCo data."""

    site_id = _site_id(model, ee_site_name)
    position = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    rotation = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
    return position, rotation


def site_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site_name: str = "ee_site",
    n_joints: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return TCP translational and rotational Jacobians for controlled joints."""

    site_id = _site_id(model, ee_site_name)
    _, dof_addresses = _controlled_joint_addresses(model, n_joints)
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site_id)
    return jacobian_position[:, dof_addresses].copy(), jacobian_rotation[:, dof_addresses].copy()


@dataclass
class MujocoKinematics:
    """Independent FK and TCP Jacobian access for the controlled arm joints."""

    model: mujoco.MjModel
    ee_site_name: str = "ee_site"
    n_joints: int = 6

    def __post_init__(self) -> None:
        self._site_id = _site_id(self.model, self.ee_site_name)
        self._qpos_addresses, self._dof_addresses = _controlled_joint_addresses(self.model, self.n_joints)
        self.joint_low, self.joint_high = controlled_joint_limits(self.model, self.n_joints)
        self.data = mujoco.MjData(self.model)

    def _validate_q(self, q: np.ndarray) -> np.ndarray:
        values = np.asarray(q, dtype=np.float64)
        if values.shape != (self.n_joints,):
            raise ValueError(f"q must have shape ({self.n_joints},), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("q must contain only finite values")
        return values

    def _forward(self, q: np.ndarray) -> np.ndarray:
        values = self._validate_q(q)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_addresses] = values
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return values

    def forward(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute TCP world position and rotation at ``q`` using private data."""

        self._forward(q)
        return site_pose(self.model, self.data, self.ee_site_name)

    def jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute TCP translational and rotational Jacobians at ``q``."""

        self._forward(q)
        return site_jacobian(self.model, self.data, self.ee_site_name, self.n_joints)

    def pose_jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute TCP pose and a stacked [translation; rotation] Jacobian at ``q``."""

        self._forward(q)
        position, rotation = site_pose(self.model, self.data, self.ee_site_name)
        jacobian_position, jacobian_rotation = site_jacobian(
            self.model,
            self.data,
            self.ee_site_name,
            self.n_joints,
        )
        return position, rotation, np.vstack([jacobian_position, jacobian_rotation])

    def joint_limit_margin(self, q: np.ndarray) -> float:
        """Return the signed closest distance to a hard joint limit in radians."""

        values = self._validate_q(q)
        return float(np.min(np.minimum(values - self.joint_low, self.joint_high - values)))

    def sigma_min(self, q: np.ndarray) -> float:
        """Return the minimum singular value of the full six-dimensional TCP Jacobian."""

        _, _, jacobian = self.pose_jacobian(q)
        return float(np.linalg.svd(jacobian, compute_uv=False)[-1])
