"""Regression tests for auditable UR5e manifest resolution metadata."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "paper_experiments" / "ur5e_workflow.py"
SPEC = importlib.util.spec_from_file_location("ur5e_workflow_upgrade_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
WORKFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)


class UR5eManifestUpgradeTests(unittest.TestCase):
    def test_upgrade_preserves_legacy_fingerprint_payload_and_exposes_contract(self) -> None:
        robot = WORKFLOW.load_robot_spec(WORKFLOW.ROBOT_CONFIG, validate_model=False)
        legacy_base = {
            "robot_config": "./configs/robots/ur5e.yaml",
            "model_xml": "ABB_IRB2400.xml",
            "anticipation_delay_steps": 6,
        }
        manifest = {
            "schema_version": 1,
            "kind": "ur5e_nominal_portability",
            "robot_identity": robot.artifact_identity(),
            "checkpoint": {"path": "checkpoint.pt"},
            "normalizer": {"path": "normalizer.npz"},
            "references": {"circle": {"path": "reference.npz"}},
            "delay_calibration": {"anticipation_delay_steps": 7},
            "base_run_args": legacy_base,
        }
        case = {"label": "FullVirtual", "trajectory": "circle", "seed": 0}
        before = WORKFLOW._case_fingerprint(manifest, case)
        upgraded = WORKFLOW.upgrade_manifest_payload(manifest, robot)

        self.assertEqual(upgraded["schema_version"], 2)
        self.assertEqual(upgraded["base_run_args"], legacy_base)
        self.assertEqual(WORKFLOW._case_fingerprint(upgraded, case), before)
        self.assertIn("before RobotSpec resolution", upgraded["base_run_args_semantics"])
        self.assertEqual(
            upgraded["resolved_robot_contract"]["model_xml"],
            "dynamics_modeling/robots/ur5e/ur5e_project.xml",
        )
        self.assertEqual(upgraded["formal_case_delay_steps"]["FullVirtual"], 7)
        self.assertEqual(upgraded["formal_case_delay_steps"]["ProjectedDirectIK"], 0)


if __name__ == "__main__":
    unittest.main()
