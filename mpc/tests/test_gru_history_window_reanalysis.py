"""Unit tests for the frozen GRU true-history window reanalysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "paper_experiments" / "reanalyze_gru_history_windows.py"
SPEC = importlib.util.spec_from_file_location("gru_history_window_reanalysis_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HistoryWindowSelectionTests(unittest.TestCase):
    def test_fixed_nonoverlapping_starts_require_complete_history(self) -> None:
        self.assertEqual(
            MODULE.window_starts(200, history_len=16, max_horizon=20, stride=20),
            (16, 36, 56, 76, 96, 116, 136, 156, 176),
        )

    def test_short_rollout_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.window_starts(35, history_len=16, max_horizon=20, stride=20)

    def test_horizon_parser_deduplicates_and_sorts(self) -> None:
        self.assertEqual(MODULE.parse_horizons("20,1,5,20"), (1, 5, 20))

    def test_float32_action_std_metadata_has_stable_group_label(self) -> None:
        self.assertEqual(MODULE.action_std_label(0.800000011920929), "0.8")


if __name__ == "__main__":
    unittest.main()
