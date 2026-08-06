from __future__ import annotations

import numpy as np


class SafeExcitation:
    """Deterministic local excitation; final limits are always enforced by the backend."""

    def __init__(self, home: np.ndarray, control_dt: float, amplitude_rad: float = np.deg2rad(3.0), seed: int = 0):
        self.home = np.asarray(home, dtype=np.float32)
        self.dt = float(control_dt)
        self.amplitude = float(amplitude_rad)
        self.rng = np.random.default_rng(seed)
        self.phase = self.rng.uniform(0, 2 * np.pi, self.home.size)
        self.frequency = np.linspace(0.08, 0.28, self.home.size)

    def __call__(self, tick: int, state: object) -> np.ndarray:
        t = tick * self.dt
        return (self.home + self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase)).astype(np.float32)


class ModelAExcitation:
    """Explicit, reproducible low-authority motion modes for real-data sessions."""

    MOTION_MODE_IDS = {"hold_perturbation": 0, "single_joint_sine": 1, "multi_joint_sine": 2,
                       "smooth_random": 3, "static_hold": 4}

    def __init__(self, home: np.ndarray, control_dt: float, *, mode: str, amplitude_rad: float,
                 joint_index: int = 0, seed: int = 0):
        if mode not in self.MOTION_MODE_IDS:
            raise ValueError(f"unsupported Model-A motion mode: {mode}")
        self.home = np.asarray(home, dtype=np.float32)
        self.dt, self.mode, self.amplitude = float(control_dt), mode, float(amplitude_rad)
        self.joint_index = int(joint_index)
        self._sine = SafeExcitation(self.home, self.dt, self.amplitude, seed)
        self._rng = np.random.default_rng(seed)
        self._random_target = self.home.copy()

    @property
    def motion_mode_id(self) -> int:
        return self.MOTION_MODE_IDS[self.mode]

    def __call__(self, tick: int, state: object) -> np.ndarray:
        t = tick * self.dt
        if self.mode == "static_hold":
            return self.home.copy()
        if self.mode == "multi_joint_sine":
            return self._sine(tick, state)
        if self.mode == "single_joint_sine":
            result = self.home.copy()
            result[self.joint_index] += self.amplitude * np.sin(2 * np.pi * .10 * t)
            return result
        if self.mode == "hold_perturbation":
            result = self.home.copy()
            result[self.joint_index] += .25 * self.amplitude * np.sin(2 * np.pi * .06 * t)
            return result
        # Update a bounded random target at 1 Hz, then low-pass it to prevent steps.
        if tick % max(1, round(1 / self.dt)) == 0:
            self._random_target = self.home + self._rng.uniform(-self.amplitude, self.amplitude, self.home.size)
        return (.96 * np.asarray(state.q_ctrl, dtype=np.float32) + .04 * self._random_target).astype(np.float32)
