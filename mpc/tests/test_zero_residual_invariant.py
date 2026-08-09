from __future__ import annotations

import unittest

import numpy as np

from mpc.preview_nominal import nominal_command
from robot_runtime.executable_command import (
    ExecutableCommandState,
    make_executable_command_spec,
    step_executable_command_np,
)
from robot_runtime.runner import PlannerCommand, compose_requested_command


class ZeroResidualBaselineTests(unittest.TestCase):
    def test_zero_residual_uses_same_tick_nominal(self) -> None:
        reference = np.linspace(-0.4, 0.4, 20, dtype=np.float32)[:, None].repeat(5, axis=1)
        for tick in (0, 4, 11):
            nominal = nominal_command(reference, tick, 0)
            command = compose_requested_command(
                nominal, np.zeros(5, dtype=np.float32),
                PlannerCommand(np.zeros(5), 0, tick, tick, True), applied=True,
            )
            np.testing.assert_array_equal(command, reference[tick])

    def test_direct_and_zero_residual_have_identical_transmitted_raw(self) -> None:
        n = 5
        spec = make_executable_command_spec(
            joint_low=np.full(n, -1.0), joint_high=np.full(n, 1.0),
            velocity_limit=np.full(n, 3.0), acceleration_limit=np.full(n, 20.0),
            relative_limit=np.full(n, 2.0), raw_low=np.zeros(n),
            raw_high=np.full(n, 4095.0), calibration_low=np.zeros(n),
            calibration_high=np.full(n, 4095.0), control_dt=1.0 / 30.0,
        )
        state = ExecutableCommandState.anchored(np.zeros(n))
        nominal = np.full(n, 0.12, dtype=np.float32)
        direct = step_executable_command_np(nominal, np.zeros(n), state, spec)
        mpc = step_executable_command_np(
            compose_requested_command(nominal, np.zeros(n, dtype=np.float32), None, applied=False),
            np.zeros(n), state, spec,
        )
        np.testing.assert_array_equal(direct.tx_goal_position_raw, mpc.tx_goal_position_raw)
        np.testing.assert_allclose(direct.transmitted_q_ref, mpc.transmitted_q_ref, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
