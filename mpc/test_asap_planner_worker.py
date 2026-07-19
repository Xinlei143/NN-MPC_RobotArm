from __future__ import annotations

import unittest

import numpy as np

from mpc.delay_aware import project_executable_command_np


class ExecutableAnchorProjectionTests(unittest.TestCase):
    def test_projected_command_and_residual_replace_raw_packet_value(self) -> None:
        command, executed, velocity = project_executable_command_np(
            nominal_q_ref=np.array([0.50], dtype=np.float32),
            requested_correction=np.array([0.08], dtype=np.float32),
            previous_command=np.array([0.50], dtype=np.float32),
            previous_velocity=np.array([0.0], dtype=np.float32),
            joint_low=np.array([-2.0], dtype=np.float32), joint_high=np.array([2.0], dtype=np.float32),
            joint_limit_margin=0.0, velocity_limit=np.array([10.0], dtype=np.float32),
            acceleration_limit=np.array([300.0], dtype=np.float32), control_dt=0.01,
        )
        np.testing.assert_allclose(command, [0.53], atol=1e-6)
        np.testing.assert_allclose(executed, [0.03], atol=1e-6)
        np.testing.assert_allclose(velocity, [3.0], atol=1e-6)
        self.assertFalse(np.allclose(executed, [0.08]))

    def test_two_step_executed_residual_velocity_uses_projected_values(self) -> None:
        dt = 0.01
        prior_command = np.array([0.50], dtype=np.float32)
        prior_velocity = np.array([0.0], dtype=np.float32)
        commands = []
        residuals = []
        for correction in (np.array([0.02], dtype=np.float32), np.array([0.03], dtype=np.float32)):
            command, residual, velocity = project_executable_command_np(
                np.array([0.50], dtype=np.float32), correction, prior_command, prior_velocity,
                np.array([-2.0], dtype=np.float32), np.array([2.0], dtype=np.float32), 0.0,
                np.array([10.0], dtype=np.float32), np.array([100.0], dtype=np.float32), dt,
            )
            commands.append(command); residuals.append(residual)
            prior_command, prior_velocity = command, velocity
        np.testing.assert_allclose(residuals[0], [0.01], atol=1e-6)
        np.testing.assert_allclose(residuals[1], [0.03], atol=1e-6)
        np.testing.assert_allclose((residuals[1] - residuals[0]) / dt, [2.0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
