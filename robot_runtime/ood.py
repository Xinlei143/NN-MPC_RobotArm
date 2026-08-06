from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustEnvelope:
    median: np.ndarray
    scale: np.ndarray
    threshold: float

    @classmethod
    def fit(cls, training_tokens: np.ndarray, validation_tokens: np.ndarray, percentile: float = 99.5) -> "RobustEnvelope":
        train = np.asarray(training_tokens, dtype=np.float64)
        validation = np.asarray(validation_tokens, dtype=np.float64)
        median = np.median(train, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(train - median), axis=0), 1e-6)
        scores = np.max(np.abs((validation - median) / scale), axis=-1)
        return cls(median.astype(np.float32), scale.astype(np.float32), float(np.percentile(scores, percentile)))

    def score(self, tokens: np.ndarray) -> np.ndarray:
        return np.max(np.abs((np.asarray(tokens) - self.median) / self.scale), axis=-1)

    def contains(self, tokens: np.ndarray) -> np.ndarray:
        return self.score(tokens) <= self.threshold


def training_action_bounds(actions: np.ndarray, safety_low: np.ndarray, safety_high: np.ndarray,
                           expansion: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.percentile(np.asarray(actions), [0.5, 99.5], axis=0)
    margin = expansion * (high - low)
    return np.maximum(low - margin, safety_low), np.minimum(high + margin, safety_high)
