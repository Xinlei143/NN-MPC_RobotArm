from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("test_model_c_evaluate", ROOT / "scripts" / "model_c" / "evaluate.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ModelABCSummaryTests(unittest.TestCase):
    def test_threaded_sparse_solve_fields_are_aggregated_finitely(self) -> None:
        arrays = {
            "realized_tracking_error": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "failure_flags": np.zeros(3, dtype=np.int64),
            "planning_time": np.array([np.nan, 0.02, np.nan], dtype=np.float32),
            "best_cost": np.array([np.nan, 4.0, np.nan], dtype=np.float32),
            "mpc_replanned": np.array([0, 1, 0], dtype=np.int64),
            "actual_states": np.zeros((3, 2), dtype=np.float32),
            "q_des": np.zeros((3, 1), dtype=np.float32),
            "ee_position_errors": np.array([0.01, 0.02, 0.03], dtype=np.float32),
            "ee_orientation_errors": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "actual_control_period_s": np.array([np.nan, 0.01, 0.012], dtype=np.float32),
            "control_step_wall_time": np.array([0.001, 0.002, 0.003], dtype=np.float32),
            "control_wakeup_lateness_s": np.array([0.0, 0.001, 0.002], dtype=np.float32),
            "control_deadline_miss": np.zeros(3, dtype=np.int64),
            "packet_age": np.array([-1, 0, 1], dtype=np.int64),
            "planner_solve_count": np.asarray(5),
            "planner_actual_update_rate_hz": np.asarray(25.0),
            "planner_late_drop_count": np.asarray(1),
        }
        summary = MODULE.summarize("A", arrays, "")
        self.assertAlmostEqual(summary["planning_time_mean"], 0.02)
        self.assertAlmostEqual(summary["best_cost_mean"], 4.0)
        self.assertAlmostEqual(summary["active_packet_ratio"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["planner_late_drop_rate"], 0.2)
        self.assertTrue(np.isfinite(summary["tcp_position_rmse_m"]))


if __name__ == "__main__":
    unittest.main()
