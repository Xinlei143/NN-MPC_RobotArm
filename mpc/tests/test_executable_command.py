from __future__ import annotations

import unittest

import numpy as np
import torch

from robot_runtime.executable_command import (
    ExecutableCommandState,
    make_executable_command_spec,
    step_executable_command_np,
    step_executable_command_torch,
)


class ExecutableCommandParityTests(unittest.TestCase):
    def setUp(self) -> None:
        n = 5
        self.spec = make_executable_command_spec(
            joint_low=np.full(n, -1.0), joint_high=np.full(n, 1.0),
            velocity_limit=np.full(n, 1.5), acceleration_limit=np.full(n, 4.0),
            relative_limit=np.full(n, 0.8), raw_low=np.full(n, 100.0),
            raw_high=np.full(n, 3995.0), calibration_low=np.zeros(n),
            calibration_high=np.full(n, 4095.0), control_dt=1.0 / 30.0,
        )

    def test_numpy_and_torch_match_raw_counts_over_sequence(self) -> None:
        rng = np.random.default_rng(10)
        state_np = ExecutableCommandState.anchored(np.zeros(5))
        previous_q = torch.zeros((1, 5), dtype=torch.float32)
        previous_velocity = torch.zeros((1, 5), dtype=torch.float32)
        measured = torch.zeros((1, 5), dtype=torch.float32)
        for _ in range(30):
            requested = rng.uniform(-2.0, 2.0, size=5).astype(np.float32)
            np_result = step_executable_command_np(requested, measured.numpy()[0], state_np, self.spec)
            _, raw, transmitted, velocity = step_executable_command_torch(
                torch.as_tensor(requested).view(1, -1), measured,
                previous_q, previous_velocity, self.spec,
            )
            np.testing.assert_array_equal(raw.numpy()[0], np_result.tx_goal_position_raw)
            np.testing.assert_allclose(transmitted.numpy()[0], np_result.transmitted_q_ref, atol=0.0, rtol=0.0)
            np.testing.assert_allclose(velocity.numpy()[0], np_result.command_velocity, atol=2e-7, rtol=0.0)
            state_np = np_result.next_state
            previous_q = transmitted
            previous_velocity = velocity

    def test_next_velocity_uses_transmitted_quantized_position(self) -> None:
        state = ExecutableCommandState.anchored(np.zeros(5))
        result = step_executable_command_np(np.full(5, 0.2), np.zeros(5), state, self.spec)
        expected = (result.transmitted_q_ref - state.previous_transmitted_q_ref) / self.spec.control_dt
        np.testing.assert_allclose(result.command_velocity, expected, atol=1e-7, rtol=0.0)
        self.assertIn("encoder_quantization", result.projection_flags)


if __name__ == "__main__":
    unittest.main()
