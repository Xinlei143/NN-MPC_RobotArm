from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
if str(DYNAMICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAMICS_ROOT))

from neural_dynamics.mujoco_env import MuJoCoArmEnv
from mpc.reference_pipeline import (
    ReferenceConfig,
    build_reference,
    load_reference_bundle,
    save_reference_bundle,
)
from mpc.robot_config import load_robot_spec, validate_robot_model_contract


class RobotConfigTests(unittest.TestCase):
    def test_default_profile_preserves_abb_behavior(self) -> None:
        robot = load_robot_spec(validate_model=True)
        self.assertEqual(robot.robot_id, "abb_irb2400")
        np.testing.assert_array_equal(robot.home_q, np.zeros(6, dtype=np.float32))
        self.assertEqual(robot.gravity_compensation_zero_indices, (5,))

    def test_ur5e_model_contract_and_home_hold(self) -> None:
        robot = load_robot_spec("configs/robots/ur5e.yaml", validate_model=True)
        report = validate_robot_model_contract(robot)
        self.assertEqual((report["nq"], report["nv"], report["nu"]), (6, 6, 6))
        self.assertEqual(report["control_dt"], 0.01)
        self.assertEqual(robot.gravity_compensation_zero_indices, ())

        env = MuJoCoArmEnv(
            str(robot.model_xml),
            n_joints=robot.n_joints,
            frame_skip=robot.frame_skip,
            home_q=robot.home_q,
            ee_site_name=robot.ee_site_name,
            gravity_compensation=robot.gravity_compensation,
            gravity_compensation_zero_indices=robot.gravity_compensation_zero_indices,
        )
        try:
            env.reset_to_configuration(robot.home_q)
            for _ in range(500):
                env.step(robot.home_q)
            state = env.get_state()
            self.assertTrue(np.all(np.isfinite(state)))
            np.testing.assert_allclose(state[:6], robot.home_q, atol=1e-4)
        finally:
            env.close()

    def test_ur5e_nonzero_home_reference_has_strict_identity(self) -> None:
        robot = load_robot_spec("configs/robots/ur5e.yaml", validate_model=True)
        model = mujoco.MjModel.from_xml_path(str(robot.model_xml))
        config = ReferenceConfig(
            shape_name="circle",
            repeat_count=1,
            circle_radius=0.02,
            lap_duration=2.0,
            safe_search_samples=300,
            ee_site_name=robot.ee_site_name,
            max_joint_velocity=tuple(robot.command_velocity_limit),
            max_joint_acceleration=tuple(robot.command_acceleration_limit),
        )
        bundle = build_reference(
            config,
            model,
            robot.home_q,
            robot.expected_control_dt,
            horizon=4,
            robot_spec=robot,
        )
        np.testing.assert_allclose(bundle.q_des[0], robot.home_q, atol=1e-6)
        np.testing.assert_allclose(
            bundle.q_des[bundle.execution_steps - 1],
            robot.home_q,
            atol=1e-6,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_reference_bundle(bundle, directory)
            load_reference_bundle(path, expected_robot_spec=robot, min_horizon=4)
            with self.assertRaisesRegex(ValueError, "robot identity mismatch"):
                load_reference_bundle(path, expected_robot_spec=load_robot_spec())


if __name__ == "__main__":
    unittest.main()
