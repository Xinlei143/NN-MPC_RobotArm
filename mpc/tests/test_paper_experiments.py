from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mpc.delay_protocol import PROTOCOL_NAMES, resolve_delay_protocol
from mpc.task_space_reference import generate_task_space_trajectory
from scripts.experiment_utils.bootstrap import paired_bootstrap_rows
from scripts.paper_experiments.evaluation import summarize_arrays
from scripts.paper_experiments.merge_mpc_ik_results import _deduplicate_ik
from scripts.paper_experiments.workflow import (
    _base_args,
    _legacy_compatibility_backfill,
    _select_preview,
    suite_cases,
)
from dynamics_modeling.scripts.eval_dynamics import parse_action_std_groups


def load_runner():
    spec = importlib.util.spec_from_file_location("paper_protocol_test_runner", ROOT / "scripts" / "run_cem_mpc.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


RUNNER = load_runner()


class DelayProtocolTests(unittest.TestCase):
    def test_canonical_protocol_matrix_has_no_duplicate_rows(self) -> None:
        rows = []
        for name in PROTOCOL_NAMES:
            protocol = resolve_delay_protocol(name)
            rows.append((protocol.future_state, protocol.future_reference, protocol.reanchor_residual, protocol.feedback))
        self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual(rows[0], (True, True, True, True))
        self.assertEqual(resolve_delay_protocol("no_future_alignment").future_reference, False)

    def test_anchor_only_protocol_semantics(self) -> None:
        protocol = resolve_delay_protocol("anchor_only")
        self.assertEqual(
            (protocol.future_state, protocol.future_reference, protocol.reanchor_residual, protocol.feedback),
            (True, True, False, False),
        )

    def test_zero_delay_plan_is_active_on_the_same_logical_tick(self) -> None:
        args = RUNNER.parse_args([
            "--dynamics_backend", "mujoco_oracle", "--device", "cpu",
            "--planner_projection", "off", "--planner_projection_strategy", "full",
            "--exact_task_space_cost", "off",
            "--multirate_mode", "virtual_asap", "--delay_protocol", "full",
            "--anticipation_delay_steps", "0", "--reference_mode", "multi_joint_sine",
            "--episode_len", "12", "--max_execution_steps", "2", "--settle_steps", "1",
            "--horizon", "3", "--num_samples", "3", "--cem_iters", "1",
            "--replan_interval_steps", "1", "--mpc_warmup_plans", "0",
        ])
        arrays = RUNNER.run_closed_loop_mpc(args)["arrays"]
        self.assertEqual(arrays["anticipation_delay_steps"].item(), 0)
        self.assertEqual(arrays["packet_age"][0], 0)
        self.assertIn("packet_activated_zero_delay", arrays["packet_event"][0])

    def test_naive_packet_age_starts_at_zero_when_delay_expires(self) -> None:
        args = RUNNER.parse_args([
            "--dynamics_backend", "mujoco_oracle", "--device", "cpu",
            "--planner_projection", "off", "--planner_projection_strategy", "full",
            "--exact_task_space_cost", "off",
            "--multirate_mode", "virtual_asap", "--delay_protocol", "naive_delayed",
            "--anticipation_delay_steps", "1", "--reference_mode", "multi_joint_sine",
            "--episode_len", "12", "--max_execution_steps", "2", "--settle_steps", "1",
            "--horizon", "3", "--num_samples", "3", "--cem_iters", "1",
            "--replan_interval_steps", "1", "--mpc_warmup_plans", "0",
        ])
        arrays = RUNNER.run_closed_loop_mpc(args)["arrays"]
        self.assertEqual(arrays["packet_age"].tolist(), [-1, 0])


class PreviewSelectionTests(unittest.TestCase):
    def test_orientation_guard_selects_best_tcp_among_eligible_candidates(self) -> None:
        rows = [
            {"preview_steps": 3, "tcp_rmse_m": 0.0346, "orientation_rmse_rad": 0.0309},
            {"preview_steps": 7, "tcp_rmse_m": 0.0311, "orientation_rmse_rad": 0.0333},
            {"preview_steps": 8, "tcp_rmse_m": 0.0303, "orientation_rmse_rad": 0.0346},
        ]
        selected, limit = _select_preview(rows, 0.10)
        self.assertAlmostEqual(limit, 0.03399)
        self.assertEqual(selected["preview_steps"], 7)

    def test_gru_action_std_groups_cover_frozen_training_support(self) -> None:
        values = parse_action_std_groups("0.5:10,0.8:10", 20)
        self.assertEqual(values, [0.5] * 10 + [0.8] * 10)


class RoundedSquareTests(unittest.TestCase):
    def test_rounded_square_is_closed_finite_and_distinct_from_square(self) -> None:
        common = dict(
            start_position=np.zeros(3), center=np.zeros(3), plane_axis_u=(1, 0, 0),
            plane_axis_v=(0, 1, 0), fixed_rotation=np.eye(3), control_dt=0.01,
            approach_duration=0.1, lap_duration=1.0, return_duration=0.1,
            repeat_count=1, square_half_side=0.03,
        )
        rounded = generate_task_space_trajectory(
            shape_name="rounded_square", rounded_square_corner_radius=0.008, **common
        )
        strict = generate_task_space_trajectory(shape_name="square", **common)
        mask = rounded.lap_ids == 0
        loop = rounded.positions[mask]
        np.testing.assert_allclose(loop[0], loop[-1], atol=1e-12)
        self.assertTrue(np.all(np.isfinite(loop)))
        self.assertFalse(np.allclose(loop[: min(len(loop), len(strict.positions))], strict.positions[: min(len(loop), len(strict.positions))]))
        self.assertLess(float(np.max(np.linalg.norm(np.diff(loop, axis=0), axis=1))), 0.01)


class ExperimentStatisticsTests(unittest.TestCase):
    def test_bootstrap_pairs_cases_not_timesteps(self) -> None:
        rows = [
            {"label": "naive", "case_id": "circle:0", "metric": 3.0},
            {"label": "full", "case_id": "circle:0", "metric": 1.0},
            {"label": "naive", "case_id": "ellipse:0", "metric": 4.0},
            {"label": "full", "case_id": "ellipse:0", "metric": 2.0},
        ]
        report = paired_bootstrap_rows(rows, left="naive", right="full", metrics=("metric",), samples=100, seed=1)
        self.assertEqual(report["metrics"]["metric"]["n"], 2)
        self.assertEqual(report["metrics"]["metric"]["mean_delta_right_minus_left"], -2.0)

    def test_failure_rate_is_case_level_not_timestep_pseudoreplication(self) -> None:
        summary = summarize_arrays(
            "full",
            {
                "actuator_q_ref": np.zeros((4, 1), dtype=np.float32),
                "failure_flags": np.asarray([0, 0, 1, 0], dtype=np.int64),
            },
        )
        self.assertEqual(summary["failure_rate"], 1.0)
        self.assertEqual(summary["planner_failure_step_rate"], 0.25)

    def test_threaded_failure_uses_unique_events_not_persistent_status(self) -> None:
        summary = summarize_arrays(
            "threaded",
            {
                "actuator_q_ref": np.zeros((5, 1), dtype=np.float32),
                "failure_flags": np.zeros(5, dtype=np.int64),
                "planner_failure": np.ones(5, dtype=np.int64),
                "planner_failure_event": np.asarray([0, 1, 0, 0, 0], dtype=np.int64),
                "planner_failure_count": np.asarray(1, dtype=np.int64),
            },
        )
        self.assertEqual(summary["failure_rate"], 1.0)
        self.assertEqual(summary["planner_failure_count"], 1)
        self.assertEqual(summary["planner_failure_step_rate"], 0.2)

    def test_final_pool_diagnostics_are_aggregated_by_plan_event(self) -> None:
        summary = summarize_arrays(
            "full", {"actuator_q_ref": np.zeros((4, 1), dtype=np.float32)},
            [
                {"selection_mode": "baseline", "candidate_diagnostics": {"exact_validation_count": 6, "exact_valid_count": 5, "exact_selection_changed": 1, "exact_final_pool_time_s": 0.002}},
                {"selection_mode": "best", "candidate_diagnostics": {"exact_validation_count": 6, "exact_valid_count": 6, "exact_selection_changed": 0, "exact_final_pool_time_s": 0.004}},
            ],
        )
        self.assertEqual(summary["planner_event_count"], 2)
        self.assertEqual(summary["exact_pool_candidate_count_mean"], 6.0)
        self.assertEqual(summary["exact_selection_changed_rate"], 0.5)
        self.assertEqual(summary["baseline_selection_rate"], 0.5)

    def test_timing_summary_contains_max_values(self) -> None:
        summary = summarize_arrays(
            "threaded",
            {
                "actuator_q_ref": np.zeros((3, 1)),
                "planning_time": np.asarray([0.01, 0.02, 0.03]),
                "mpc_replanned": np.ones(3, dtype=bool),
                "planner_end_to_end_latency_s": np.asarray([0.02, 0.04, 0.03]),
                "control_step_wall_time": np.asarray([0.001, 0.002, 0.003]),
                "actual_control_period_s": np.asarray([0.01, 0.011, 0.012]),
                "control_wakeup_lateness_s": np.asarray([0.0, 0.001, 0.002]),
                "control_start_jitter_s": np.asarray([0.0, 0.002, 0.004]),
            },
        )
        self.assertEqual(summary["solve_max_s"], 0.03)
        self.assertEqual(summary["e2e_max_s"], 0.04)
        self.assertEqual(summary["control_compute_max_s"], 0.003)
        self.assertEqual(summary["control_period_max_s"], 0.012)

    def test_baseline_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            rows = []
            for seed in (0, 1):
                run_dir = Path(temporary) / str(seed)
                run_dir.mkdir()
                np.savez_compressed(
                    run_dir / "rollout.npz",
                    actual_states=np.zeros((2, 2)),
                    ee_position_errors=np.zeros(2),
                    ee_orientation_errors=np.zeros(2),
                )
                rows.append({
                    "method": "physical", "trajectory": "circle", "condition": "nominal",
                    "perturbation": "nominal", "level": "0", "reference_sha256": "ref",
                    "seed": seed, "run_dir": str(run_dir),
                })
            unique, report = _deduplicate_ik(rows)
            self.assertEqual(len(unique), 1)
            self.assertTrue(report[0]["deterministic"])


class PaperMatrixTests(unittest.TestCase):
    def test_paper_base_args_freeze_stage_one_off(self) -> None:
        args = _base_args()
        self.assertEqual((args.stage_one_task_space_cost, args.stage_one_task_compile), ("off", "off"))

    def test_paper_base_args_use_full_residual_parameterization(self) -> None:
        self.assertEqual(_base_args().residual_parameterization, "full")

    def test_paper_base_args_pin_the_final_gru_two_stage_method(self) -> None:
        args = _base_args()
        self.assertEqual(args.model_type, "gru")
        self.assertEqual(args.history_len, 16)
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.num_samples, 128)
        self.assertEqual(args.cem_iters, 2)
        self.assertEqual(args.rollout_batch_size, 128)
        self.assertEqual(args.planner_projection, "on")
        self.assertEqual(args.planner_projection_backend, "compiled")
        self.assertEqual(args.planner_projection_strategy, "two_stage")
        self.assertEqual(args.exact_task_space_cost, "on")
        self.assertEqual(args.w_task_position, 1.0)
        self.assertEqual(args.w_task_orientation, 0.25)
        self.assertEqual(args.uncertainty_mode, "off")
        self.assertEqual(args.cost_profile, "blackbox")
        self.assertEqual(args.residual_parameterization, "full")
        self.assertEqual(args.stage_one_task_space_cost, "off")
        self.assertEqual(args.stage_one_task_compile, "off")
        self.assertEqual(args.cem_execute, "lowest_cost")
        self.assertEqual(args.mpc_preview_nominal_steps, 0)
        self.assertEqual((args.payload_level, args.actuator_gain_level, args.force_pulse_level, args.observation_noise_level), (0, 0, 0, 0))

    @staticmethod
    def _manifest() -> dict[str, object]:
        return {
            "delay_calibration": {"anticipation_delay_steps": 6},
            "preview_calibration": {"selected_steps": 2},
            "paired_cem_seeds": [0, 1, 2, 3, 4],
            "delay_sweep_seeds": [0, 1, 2],
            "delay_sweep_steps": [0, 2, 4, 6, 8],
            "projection_common_delay_steps": 6,
            "delay_calibrations": {
                "joint_only_projection_off": {"anticipation_delay_steps": 4},
                "joint_only_full_compiled": {"anticipation_delay_steps": 6},
                "joint_only_two_stage": {"anticipation_delay_steps": 5},
            },
        }

    def test_formal_suite_sizes_match_the_registered_protocol(self) -> None:
        manifest = self._manifest()
        self.assertEqual(len(suite_cases(manifest, "main")), 84)
        self.assertEqual(len(suite_cases(manifest, "ablation")), 80)
        self.assertEqual(len(suite_cases(manifest, "delay_sweep")), 60)
        self.assertEqual(len(suite_cases(manifest, "preview")), 4)
        self.assertEqual(len(suite_cases(manifest, "oracle")), 12)
        self.assertEqual(len(suite_cases(manifest, "task_cost")), 24)
        self.assertEqual(len(suite_cases(manifest, "delay_sweep_components")), 96)
        self.assertEqual(len(suite_cases(manifest, "projection_choice")), 36)

    def test_projection_suite_freezes_joint_only_cost_and_common_d6(self) -> None:
        cases = suite_cases(self._manifest(), "projection_choice")
        self.assertTrue(all(case["exact_task_space_cost"] == "off" for case in cases))
        common = [case for case in cases if case["evaluation_set"] == "common_d"]
        self.assertEqual(len(common), 18)
        self.assertTrue(all(case["delay_steps"] == 6 for case in common))

    def test_four_stage_delay_sweep_size(self) -> None:
        self.assertEqual(len(suite_cases(self._manifest(), "delay_sweep_components")), 96)

    def test_projection_suite_size(self) -> None:
        self.assertEqual(len(suite_cases(self._manifest(), "projection_choice")), 36)

    def test_existing_main_case_cache_compatibility(self) -> None:
        self.assertEqual(
            _legacy_compatibility_backfill({}),
            {
                "residual_parameterization": "full",
                "stage_one_task_space_cost": "off",
                "stage_one_task_compile": "off",
                "mpc_preview_nominal_steps": 0,
            },
        )
        self.assertIsNone(_legacy_compatibility_backfill({"stage_one_task_space_cost": "gpu_budgeted"}))

    def test_task_cost_suite_has_fixed_and_deployed_paired_variants(self) -> None:
        cases = suite_cases(self._manifest(), "task_cost")
        fixed = [case for case in cases if case["label"].endswith("FixedD6")]
        self.assertTrue(fixed)
        self.assertTrue(all(case["delay_steps"] == 6 for case in fixed))
        self.assertEqual({case["exact_task_space_cost"] for case in cases}, {"on", "off"})

    def test_ablation_full_cases_are_exact_main_cache_reuses(self) -> None:
        manifest = self._manifest()
        ignored = {"label"}
        main = {
            tuple(sorted((key, value) for key, value in case.items() if key not in ignored))
            for case in suite_cases(manifest, "main")
            if case["label"] == "FullVirtual"
        }
        ablation = {
            tuple(sorted((key, value) for key, value in case.items() if key not in ignored))
            for case in suite_cases(manifest, "ablation")
            if case["label"] == "FullVirtual"
        }
        self.assertEqual(main, ablation)
        self.assertEqual(len(main), 20)


if __name__ == "__main__":
    unittest.main()
