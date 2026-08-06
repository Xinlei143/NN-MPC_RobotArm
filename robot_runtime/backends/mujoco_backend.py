from __future__ import annotations

import time
import numpy as np

from robot_runtime.interfaces import CommandResult, RobotState


class MuJoCoBackend:
    """Protocol-compatible adapter used for runtime tests without changing legacy simulation entrypoints."""

    def __init__(self, env):
        self.env = env
        self.n_joints = int(env.n_joints)
        self.control_dt = float(env.control_dt)
        self.joint_low = np.asarray(env.joint_low, dtype=np.float32)
        self.joint_high = np.asarray(env.joint_high, dtype=np.float32)
        self._state: np.ndarray | None = None
        self._generation = 0

    def connect(self) -> None:
        self._state = np.asarray(self.env.reset_to_configuration(self.env.home_q), dtype=np.float32)

    def read_state(self, *, tick_index: int = 0) -> RobotState:
        if self._state is None: raise RuntimeError("MuJoCo backend is not connected")
        now = time.perf_counter_ns()
        return RobotState(self._state[:self.n_joints].copy(), self._state[self.n_joints:2*self.n_joints].copy(),
                          now, 0, now, now, tick_index, self._generation, self._generation, True,
                          diagnostics={"domain": "simulation"})

    def send_joint_targets(self, q_ref: np.ndarray, *, tick_index: int = 0) -> CommandResult:
        requested = np.asarray(q_ref, dtype=np.float32)
        projected = np.clip(requested, self.joint_low, self.joint_high).astype(np.float32)
        start = time.perf_counter_ns()
        self._state = np.asarray(self.env.step(projected), dtype=np.float32)
        end = time.perf_counter_ns()
        return CommandResult(requested, projected, projected.copy(), np.empty(0, dtype=np.int64), True,
                             tick_index, start, end, ("joint_limit",) if not np.array_equal(requested, projected) else ())

    def move_to_configuration(self, target_q: np.ndarray, duration_s: float) -> None:
        del duration_s
        self._state = np.asarray(self.env.reset_to_configuration(np.asarray(target_q)), dtype=np.float32)

    def disable_torque(self) -> None: pass

    def close(self) -> None: self._state = None
