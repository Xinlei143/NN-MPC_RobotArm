"""Reusable passive MuJoCo mirror for rendering real SO101 joint states.

This class is extracted from ``LiveMuJoCoMirror`` in
``dynamics_modeling/scripts/collect_real_workspace_model_a.py`` so that
read-only visualization can be driven from other entry points (for example
the per-joint mapping-calibration direct-control tool).  The collector keeps
its own copy; this module is the shared, standalone form.

The viewer owns a separate MuJoCo model/data pair so rendering cannot mutate
the model/data pair used by the table-clearance safety checker.  It is
deliberately observational: closing its window never interrupts a real run and
it never contributes to command generation.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class MuJoCoMirror:
    """Passive detailed-model viewer driven only by measured SO101 positions.

    Parameters
    ----------
    table_profile:
        An object exposing ``mesh_collision_model_xml``,
        ``mesh_collision_joint_names`` and ``mapping`` (a
        ``JointCoordinateMapping``); the approved table-safety profile is the
        canonical source.
    n_joints:
        Number of controlled arm joints to drive.
    mapping:
        Optional override mapping used for rendering instead of
        ``table_profile.mapping``.  This is how the calibration tool previews
        a candidate ``joint_offset``/``joint_sign`` without editing the
        profile: pass ``JointCoordinateMapping(sign, offset + preview, ...)``.
    """

    def __init__(self, table_profile: Any, n_joints: int, *, mapping: Any | None = None,
                 gripper_model_angle_rad: float | None = None) -> None:
        if table_profile.mesh_collision_model_xml is None:
            raise ValueError("a detailed mesh collision model is required for live visualization")
        try:
            import mujoco
            import mujoco.viewer
        except Exception as exc:
            raise RuntimeError("MuJoCo mirror requires the Python MuJoCo viewer package") from exc
        self._mujoco = mujoco
        self._profile = table_profile
        self._mapping = table_profile.mapping if mapping is None else mapping
        self._model = mujoco.MjModel.from_xml_path(str(table_profile.mesh_collision_model_xml))
        self._data = mujoco.MjData(self._model)
        addresses: list[int] = []
        for name in table_profile.mesh_collision_joint_names:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"viewer model has no joint {name!r}")
            addresses.append(int(self._model.jnt_qposadr[joint_id]))
        if len(addresses) != n_joints:
            raise ValueError("viewer joint count does not match controlled joint count")
        self._qpos_addresses = np.asarray(addresses, dtype=np.intp)
        self._gripper_angle: float | None = gripper_model_angle_rad
        self._gripper_qposadr: int | None = None
        if gripper_model_angle_rad is not None:
            gripper_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
            if gripper_id >= 0:
                self._gripper_qposadr = int(self._model.jnt_qposadr[gripper_id])
        try:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        except Exception as exc:
            raise RuntimeError(
                "Could not open the MuJoCo mirror. Use a local graphical DISPLAY or omit --visualize-mujoco."
            ) from exc
        self._closed_notice_emitted = False

    @property
    def mapping(self) -> Any:
        """The mapping currently applied for rendering (may be a preview)."""
        return self._mapping

    def update(self, q_ctrl: np.ndarray) -> None:
        """Render one measured position without affecting the real run."""
        if not self._viewer.is_running():
            if not self._closed_notice_emitted:
                print("MuJoCo mirror window was closed; the real run continues without visualization.")
                self._closed_notice_emitted = True
            return
        values = np.asarray(q_ctrl, dtype=np.float64)
        if values.shape != (len(self._qpos_addresses),) or not np.all(np.isfinite(values)):
            return
        with self._viewer.lock():
            self._mujoco.mj_resetData(self._model, self._data)
            self._data.qpos[self._qpos_addresses] = self._mapping.to_kinematics(values)
            if self._gripper_qposadr is not None and self._gripper_angle is not None:
                self._data.qpos[self._gripper_qposadr] = self._gripper_angle
            self._data.qvel[:] = 0.0
            self._mujoco.mj_forward(self._model, self._data)
        self._viewer.sync()

    def close(self) -> None:
        try:
            self._viewer.close()
        except Exception:
            pass
