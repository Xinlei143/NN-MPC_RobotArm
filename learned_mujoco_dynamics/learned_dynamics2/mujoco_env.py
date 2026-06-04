from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np


class MuJoCoArmEnv:
    """Small MuJoCo wrapper for joint-space arm dynamics collection."""

    _EE_NAMES = ("ee_site", "tool0", "flange", "end_effector")

    def __init__(
        self,
        model_xml: str,
        n_joints: int = 6,
        control_mode: str = "velocity",
        frame_skip: int = 5,
        dt: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model_xml = Path(model_xml).expanduser()
        if not self.model_xml.exists():
            raise FileNotFoundError(f"MuJoCo XML file does not exist: {self.model_xml}")
        if n_joints <= 0:
            raise ValueError(f"n_joints must be positive, got {n_joints}")
        if frame_skip <= 0:
            raise ValueError(f"frame_skip must be positive, got {frame_skip}")
        if control_mode not in {"velocity", "position"}:
            raise ValueError(f"control_mode must be 'velocity' or 'position', got {control_mode!r}")

        self.n_joints = int(n_joints)
        self.control_mode = control_mode
        self.frame_skip = int(frame_skip)
        self.rng = np.random.default_rng(seed)

        self.model = mujoco.MjModel.from_xml_path(str(self.model_xml))
        self.data = mujoco.MjData(self.model)
        if self.model.nu < self.n_joints:
            raise ValueError(
                f"Actuator count mismatch: model has {self.model.nu} actuators, "
                f"but n_joints={self.n_joints}. Provide a model with at least one actuator per controlled joint."
            )
        if self.model.nq < self.n_joints or self.model.nv < self.n_joints:
            raise ValueError(
                f"Joint state dimension mismatch: model has nq={self.model.nq}, nv={self.model.nv}, "
                f"but n_joints={self.n_joints}."
            )
        if dt is not None:
            if dt <= 0:
                raise ValueError(f"dt must be positive when provided, got {dt}")
            self.model.opt.timestep = float(dt)
        self.action_low = np.asarray(self.model.actuator_ctrlrange[: self.n_joints, 0], dtype=np.float32)
        self.action_high = np.asarray(self.model.actuator_ctrlrange[: self.n_joints, 1], dtype=np.float32)

    @property
    def state_dim(self) -> int:
        return 2 * self.n_joints

    @property
    def action_dim(self) -> int:
        return self.n_joints

    @property
    def control_dt(self) -> float:
        return float(self.model.opt.timestep * self.frame_skip)

    def get_state(self) -> np.ndarray:
        qpos = np.asarray(self.data.qpos[: self.n_joints], dtype=np.float64)
        qvel = np.asarray(self.data.qvel[: self.n_joints], dtype=np.float64)
        return np.concatenate([qpos, qvel]).astype(np.float32)

    def step(self, action: np.ndarray) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != (self.n_joints,):
            raise ValueError(f"Action must have shape ({self.n_joints},), got {action_array.shape}")

        action_array = np.clip(action_array, self.action_low, self.action_high)
        self.data.ctrl[: self.n_joints] = action_array
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def reset_random(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[: self.n_joints] = self.rng.uniform(-0.25, 0.25, size=self.n_joints)
        self.data.qvel[: self.n_joints] = self.rng.uniform(-0.05, 0.05, size=self.n_joints)
        self.data.ctrl[: self.n_joints] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def get_ee_position(self) -> Optional[np.ndarray]:
        for name in self._EE_NAMES:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if site_id >= 0:
                return np.asarray(self.data.site_xpos[site_id], dtype=np.float32).copy()

        for name in self._EE_NAMES:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return np.asarray(self.data.xpos[body_id], dtype=np.float32).copy()
        return None

    def close(self) -> None:
        return None
