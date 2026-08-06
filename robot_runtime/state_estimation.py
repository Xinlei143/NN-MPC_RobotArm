from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CausalVelocityEstimator:
    n_joints: int
    cutoff_hz: float = 4.0

    def __post_init__(self) -> None:
        if self.n_joints <= 0 or self.cutoff_hz <= 0:
            raise ValueError("n_joints and cutoff_hz must be positive")
        self.generation = 0
        self.reset()

    def reset(self) -> None:
        self._previous_q: np.ndarray | None = None
        self._previous_timestamp_ns: int | None = None
        self._filtered = np.zeros(self.n_joints, dtype=np.float64)
        self.last_raw = self._filtered.copy()
        self.generation += 1

    def update(self, q: np.ndarray, timestamp_ns: int) -> np.ndarray:
        value = np.asarray(q, dtype=np.float64)
        if value.shape != (self.n_joints,):
            raise ValueError(f"q must have shape ({self.n_joints},)")
        if self._previous_q is None:
            self._previous_q = value.copy()
            self._previous_timestamp_ns = int(timestamp_ns)
            return self._filtered.astype(np.float32)
        dt = (int(timestamp_ns) - int(self._previous_timestamp_ns)) * 1e-9
        if dt <= 0:
            raise ValueError("velocity estimator timestamps must be strictly increasing")
        raw = (value - self._previous_q) / dt
        alpha = float(np.exp(-2.0 * np.pi * self.cutoff_hz * dt))
        self._filtered = alpha * self._filtered + (1.0 - alpha) * raw
        self.last_raw = raw
        self._previous_q = value.copy()
        self._previous_timestamp_ns = int(timestamp_ns)
        return self._filtered.astype(np.float32)
