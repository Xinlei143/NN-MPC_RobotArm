from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "dynamics_modeling") not in sys.path:
    sys.path.insert(0, str(ROOT / "dynamics_modeling"))

from mpc.budgeted_uncertainty import high_risk_prediction, selected_cem_candidates, soft_residual_scale


def test_selected_cem_candidates_uses_only_executable_selected_branch() -> None:
    result = SimpleNamespace(
        selected_q_ref_sequence=np.zeros((20, 6), dtype=np.float32),
        selected_predicted_state_sequence=np.zeros((21, 12), dtype=np.float32),
    )
    commands, primary = selected_cem_candidates(result, horizon=5)
    assert list(commands) == ["selected"]
    assert commands["selected"].shape == (5, 6)
    assert primary["selected"].shape == (6, 12)


def test_soft_residual_scale_is_continuous_and_bounded() -> None:
    assert soft_residual_scale(0.003, 0.0037, 0.0060) == 1.0
    assert soft_residual_scale(0.0065, 0.0037, 0.0060) == 0.0
    assert np.isclose(soft_residual_scale(0.00485, 0.0037, 0.0060), 0.5)


def test_high_risk_prediction_requires_a_concrete_risk_signal() -> None:
    predicted = np.zeros((4, 4), dtype=np.float32)
    reference = np.zeros((3, 2), dtype=np.float32)
    common = dict(
        joint_limit_margin=0.1,
        residual_saturation_fraction=0.98,
        current_tracking_error=0.01,
        tracking_error_growth_ratio=1.25,
        min_tracking_error=0.02,
    )
    assert not high_risk_prediction(
        predicted, reference, np.zeros((3, 2), dtype=np.float32), np.ones(2, dtype=np.float32),
        -np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32), **common,
    )
    assert high_risk_prediction(
        predicted, reference, np.ones((3, 2), dtype=np.float32), np.ones(2, dtype=np.float32),
        -np.ones(2, dtype=np.float32), np.ones(2, dtype=np.float32), **common,
    )
