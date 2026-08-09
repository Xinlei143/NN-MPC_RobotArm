from __future__ import annotations

import numpy as np

from scripts.analyze_mpc_residual_alignment import _series_report
from scripts.benchmark_candidate_ranking import _ranking_metrics
from scripts.run_real_direct_control import transform_joint_reference
from mpc.analytical_candidates import (
    align_residual_to_velocity,
    build_preview_residual_candidates,
    parse_preview_steps,
)


def test_preview_transform_preserves_target_and_shifts_only_command() -> None:
    reference = np.zeros((12, 5), dtype=np.float32)
    reference[5:, 0] = np.arange(7, dtype=np.float32)
    command, target, metadata = transform_joint_reference(reference, 1.0 / 30.0, preview_steps=2)
    np.testing.assert_array_equal(target, reference)
    np.testing.assert_array_equal(command[0], reference[2])
    np.testing.assert_array_equal(command[-1], reference[-1])
    assert metadata["kind"] == "preview"


def test_lead_transform_is_capped_and_keeps_reference_shape() -> None:
    reference = np.zeros((10, 5), dtype=np.float32)
    reference[:, 0] = np.arange(10, dtype=np.float32)
    command, target, metadata = transform_joint_reference(
        reference, 1.0, lead_time_s=1.0, lead_max_rad=0.25
    )
    np.testing.assert_array_equal(target, reference)
    assert command.shape == reference.shape
    assert np.max(np.abs(command - reference)) <= 0.25 + 1e-6
    assert metadata["kind"] == "lead"


def test_alignment_reports_opposite_sign_residual() -> None:
    velocity = np.zeros((8, 5), dtype=np.float64)
    velocity[:, 0] = np.linspace(0.8, 1.5, 8)
    residual = -0.1 * velocity
    report = _series_report(
        residual,
        velocity,
        active_start_tick=0,
        velocity_threshold=0.5,
        residual_threshold=0.01,
    )
    shoulder = report["per_joint"][0]
    assert shoulder["pearson_corr"] < -0.99
    assert shoulder["sign_agreement"] == 0.0
    assert shoulder["opposite_sign_rate"] == 1.0


def test_candidate_ranking_metrics_distinguish_correct_order() -> None:
    metrics = _ranking_metrics(np.asarray([1.0, 2.0, 3.0]), np.asarray([0.5, 1.5, 2.5]))
    assert metrics["spearman"] > 0.99
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["top1_correct"] == 1


def test_preview_candidate_builder_uses_anchor_and_nominal_preview() -> None:
    reference = np.arange(20, dtype=np.float32).reshape(10, 2)
    nominal = reference[3:6]
    candidates = build_preview_residual_candidates(
        reference,
        anchor=3,
        horizon=3,
        nominal=nominal,
        residual_max=np.ones(2, dtype=np.float32),
        preview_steps=parse_preview_steps("0,2"),
    )
    np.testing.assert_allclose(candidates["preview:0"], 0.0)
    np.testing.assert_allclose(candidates["preview:2"], 1.0)


def test_directional_gate_removes_only_opposing_components() -> None:
    residual = np.asarray([[0.2, -0.3], [-0.2, 0.4]], dtype=np.float32)
    dq_des = np.asarray([[1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    gated, changed = align_residual_to_velocity(residual, dq_des)
    np.testing.assert_allclose(gated, [[0.2, 0.0], [-0.2, 0.0]])
    np.testing.assert_array_equal(changed, [[False, True], [False, True]])
