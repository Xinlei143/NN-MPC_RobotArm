from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "dynamics_modeling") not in sys.path:
    sys.path.insert(0, str(ROOT / "dynamics_modeling"))

from mpc.asap_runner import compose_requested_correction


def test_hard_uncertainty_gate_produces_strict_nominal_correction() -> None:
    """An old CEM prediction cannot reintroduce feedback after hard fallback."""
    feedback_raw, feedback, requested = compose_requested_correction(
        np.zeros(2, dtype=np.float32),
        np.array([4.0, -4.0, 3.0, -3.0], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        packet_age=0,
        uncertainty_gate=True,
        feedback_kq=0.30,
        feedback_kdq=0.015,
        feedback_max=np.full(2, 0.015, dtype=np.float32),
        residual_max=np.full(2, 0.12, dtype=np.float32),
    )
    assert np.allclose(feedback_raw, 0.0)
    assert np.allclose(feedback, 0.0)
    assert np.allclose(requested, 0.0)


def test_hard_uncertainty_gate_ignores_an_unexpected_nonzero_residual() -> None:
    """The strict-fallback contract must not depend on worker-side zeroing."""
    feedback_raw, feedback, requested = compose_requested_correction(
        np.array([0.08, -0.06], dtype=np.float32),
        np.array([4.0, -4.0, 3.0, -3.0], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        packet_age=0,
        uncertainty_gate=True,
        feedback_kq=0.30,
        feedback_kdq=0.015,
        feedback_max=np.full(2, 0.015, dtype=np.float32),
        residual_max=np.full(2, 0.12, dtype=np.float32),
    )
    assert np.allclose(feedback_raw, 0.0)
    assert np.allclose(feedback, 0.0)
    assert np.allclose(requested, 0.0)


def test_ungated_packet_retains_feedback() -> None:
    _, feedback, requested = compose_requested_correction(
        np.zeros(2, dtype=np.float32),
        np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        packet_age=0,
        uncertainty_gate=False,
        feedback_kq=0.30,
        feedback_kdq=0.015,
        feedback_max=np.full(2, 0.015, dtype=np.float32),
        residual_max=np.full(2, 0.12, dtype=np.float32),
    )
    assert np.allclose(feedback, [0.015, -0.015])
    assert np.allclose(requested, feedback)
