"""CLI defaults for the primary threaded ASAP experiment paths."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = load_module("cli_defaults_runner", ROOT / "scripts" / "run_cem_mpc.py")
SWEEP = load_module("cli_defaults_sweep", ROOT / "scripts" / "run_cem_budget_sweep.py")


class ThreadedASAPDefaultTests(unittest.TestCase):
    def test_runner_defaults_to_threaded_asap(self) -> None:
        args = RUNNER.parse_args([])
        self.assertEqual(args.multirate_mode, "threaded_asap")
        self.assertEqual(args.model_type, "gru")
        self.assertEqual(args.delay_protocol, "full")
        self.assertEqual(args.ik_preview_steps, 0)
        self.assertEqual(args.mpc_preview_nominal_steps, 0)
        self.assertEqual(args.planner_projection, "on")
        self.assertEqual(args.planner_projection_backend, "compiled")
        self.assertEqual(args.planner_projection_strategy, "two_stage")
        self.assertEqual(args.exact_task_space_cost, "on")
        self.assertEqual(args.w_task_position, 1.0)
        self.assertEqual(args.w_task_orientation, 0.25)
        self.assertAlmostEqual(args.task_position_scale_m, 0.05)
        self.assertAlmostEqual(args.task_orientation_scale_rad, np.deg2rad(5.0))
        self.assertEqual(args.horizon, 20)
        self.assertEqual(args.residual_parameterization, "full")
        self.assertEqual(args.residual_control_points, 8)
        self.assertEqual(args.stage_one_task_space_cost, "off")
        self.assertEqual(args.residual_cost_semantics, "requested")
        self.assertEqual(args.packet_residual_semantics, "requested")
        self.assertEqual(args.residual_feasibility_semantics, "finite")
        self.assertEqual(args.nominal_command_semantics, "raw_ik")
        self.assertEqual(args.ik_command_projection, "raw")

    def test_linear_control_points_require_valid_horizon_and_std_reset(self) -> None:
        args = RUNNER.parse_args([
            "--reference_mode", "task",
            "--residual_parameterization", "linear_control_points",
            "--residual_control_points", "8",
            "--horizon", "32",
        ])
        with self.assertRaisesRegex(ValueError, "reset_std_each_step"):
            RUNNER.run_closed_loop_mpc(args)
        args = RUNNER.parse_args([
            "--reference_mode", "task",
            "--residual_parameterization", "linear_control_points",
            "--residual_control_points", "33",
            "--horizon", "32",
            "--reset_std_each_step",
        ])
        with self.assertRaisesRegex(ValueError, "in \\[2, horizon\\]"):
            RUNNER.run_closed_loop_mpc(args)

    def test_budget_sweep_defaults_to_and_accepts_threaded_asap(self) -> None:
        with mock.patch.object(sys, "argv", ["run_cem_budget_sweep.py"]):
            self.assertEqual(SWEEP.parse_args().multirate_mode, "threaded_asap")

    def test_exact_task_space_cost_rejects_non_task_reference(self) -> None:
        args = RUNNER.parse_args(
            [
                "--exact_task_space_cost",
                "on",
                "--w_task_position",
                "1.0",
                "--reference_mode",
                "joint_sine",
            ]
        )
        with self.assertRaisesRegex(ValueError, "reference_mode task"):
            RUNNER.run_closed_loop_mpc(args)

    def test_exact_task_space_cost_rejects_non_two_stage_strategy(self) -> None:
        args = RUNNER.parse_args(
            [
                "--exact_task_space_cost",
                "on",
                "--w_task_position",
                "1.0",
                "--reference_mode",
                "task",
                "--planner_projection_strategy",
                "full",
            ]
        )
        with self.assertRaisesRegex(ValueError, "two-stage MPC"):
            RUNNER.run_closed_loop_mpc(args)

    def test_mpc_preview_nominal_requires_task_mpc(self) -> None:
        args = RUNNER.parse_args([
            "--controller_mode", "ik_direct", "--reference_mode", "task",
            "--mpc_preview_nominal_steps", "7",
        ])
        with self.assertRaisesRegex(ValueError, "only valid with --controller_mode mpc"):
            RUNNER.run_closed_loop_mpc(args)
        args = RUNNER.parse_args([
            "--controller_mode", "mpc", "--reference_mode", "joint_sine",
            "--mpc_preview_nominal_steps", "7",
        ])
        with self.assertRaisesRegex(ValueError, "requires --reference_mode task"):
            RUNNER.run_closed_loop_mpc(args)

    def test_task_reference_validation_honors_execution_cap(self) -> None:
        bundle = SimpleNamespace(
            q_des=np.zeros((10, 6), dtype=np.float32),
            dq_des=np.zeros((10, 6), dtype=np.float32),
            execution_steps=8,
            task_positions_des=np.zeros((10, 3), dtype=np.float32),
            task_rotations_des=np.zeros((10, 3, 3), dtype=np.float32),
            segment_ids=np.zeros(10, dtype=np.int64),
            lap_ids=np.zeros(10, dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "too short"):
            RUNNER._validate_task_reference(bundle, 6, 3)
        RUNNER._validate_task_reference(bundle, 6, 3, execution_steps=6)
        with mock.patch.object(sys, "argv", ["run_cem_budget_sweep.py", "--multirate_mode", "threaded_asap"]):
            self.assertEqual(SWEEP.parse_args().multirate_mode, "threaded_asap")

    def test_preview_nominal_truncates_only_the_reference_tail(self) -> None:
        bundle = SimpleNamespace(
            q_des=np.zeros((10, 6), dtype=np.float32),
            dq_des=np.zeros((10, 6), dtype=np.float32),
            ddq_des=np.zeros((10, 6), dtype=np.float32),
            execution_steps=8,
            task_positions_des=np.zeros((10, 3), dtype=np.float32),
            task_rotations_des=np.zeros((10, 3, 3), dtype=np.float32),
            segment_ids=np.zeros(10, dtype=np.int64),
            lap_ids=np.zeros(10, dtype=np.int64),
        )
        args = SimpleNamespace(
            reference_file="reference.npz",
            controller_mode="mpc",
            horizon=2,
            mpc_preview_nominal_steps=2,
            ik_preview_steps=0,
            multirate_mode="virtual_asap",
            anticipation_delay_steps=1,
            max_execution_steps=None,
            n_joints=6,
        )
        with mock.patch.object(RUNNER, "load_reference_bundle", return_value=bundle):
            returned = RUNNER._load_task_reference(args)
        self.assertIs(returned, bundle)
        # 10 - (H=2 + D=1 + P=2) - 1 = 4; original execution is 8.
        self.assertEqual(bundle.execution_steps, 4)


if __name__ == "__main__":
    unittest.main()
