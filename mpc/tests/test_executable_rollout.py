from __future__ import annotations

import unittest

import numpy as np
import torch

from neural_dynamics.models import GRUDynamics
from neural_dynamics.normalization import StandardNormalizer
from mpc.executable_rollout import ExecutableRolloutEngine
from robot_runtime.executable_command import make_executable_command_spec


def _normalizer() -> StandardNormalizer:
    normalizer = StandardNormalizer()
    normalizer.state_mean = torch.zeros(10)
    normalizer.state_std = torch.ones(10)
    normalizer.action_mean = torch.zeros(5)
    normalizer.action_std = torch.ones(5)
    normalizer.delta_mean = torch.zeros(10)
    normalizer.delta_std = torch.ones(10)
    return normalizer


def _spec():
    n = 5
    return make_executable_command_spec(
        joint_low=np.full(n, -1.0), joint_high=np.full(n, 1.0),
        velocity_limit=np.full(n, 1.5), acceleration_limit=np.full(n, 4.0),
        relative_limit=np.full(n, 0.8), raw_low=np.full(n, 100.0),
        raw_high=np.full(n, 3995.0), calibration_low=np.zeros(n),
        calibration_high=np.full(n, 4095.0), control_dt=1.0 / 30.0,
    )


class ExecutableRolloutTests(unittest.TestCase):
    def test_eager_rollout_preserves_state_machine_shapes(self) -> None:
        model = GRUDynamics(10, 5, hidden_size=8, output_dim=10).eval()
        engine = ExecutableRolloutEngine(
            model=model, normalizer=_normalizer(), model_type="gru", state_dim=10,
            target_mode="delta_state", control_dt=1.0 / 30.0, spec=_spec(), backend="eager",
        )
        output = engine.run(
            initial_history=torch.zeros(2, 16, 15),
            requested_q_ref=torch.zeros(2, 6, 5),
            previous_q_ref=torch.zeros(2, 5),
            previous_velocity=torch.zeros(2, 5),
        )
        self.assertEqual(tuple(output.q_ref_sequences.shape), (2, 6, 5))
        self.assertEqual(tuple(output.pred_states.shape), (2, 7, 10))
        self.assertEqual(tuple(output.expected_raw_sequences.shape), (2, 6, 5))
        self.assertEqual(tuple(output.final_history.shape), (2, 16, 15))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA Graph test requires CUDA")
    def test_cuda_graph_matches_eager_raw_and_state(self) -> None:
        torch.manual_seed(4)
        model = GRUDynamics(10, 5, hidden_size=8, output_dim=10).cuda().eval()
        eager = ExecutableRolloutEngine(
            model=model, normalizer=_normalizer(), model_type="gru", state_dim=10,
            target_mode="delta_state", control_dt=1.0 / 30.0, spec=_spec(), backend="eager",
        )
        graph = ExecutableRolloutEngine(
            model=model, normalizer=_normalizer(), model_type="gru", state_dim=10,
            target_mode="delta_state", control_dt=1.0 / 30.0, spec=_spec(), backend="cuda_graph",
        )
        history = torch.randn(3, 16, 15, device="cuda")
        requested = torch.randn(3, 6, 5, device="cuda") * 0.2
        previous = torch.zeros(3, 5, device="cuda")
        velocity = torch.zeros_like(previous)
        expected = torch.zeros(3, 6, 5, dtype=torch.int64, device="cuda")
        mask = torch.zeros(3, 6, dtype=torch.bool, device="cuda")
        fallback = torch.zeros_like(requested)
        expected[0, 0, 0] = 1234
        mask[0, 0] = True
        for exact, fail_closed in ((False, False), (True, False), (True, True)):
            first = eager.run(
                initial_history=history, requested_q_ref=requested,
                previous_q_ref=previous, previous_velocity=velocity,
                fallback_q_ref=fallback, expected_raw=expected,
                expected_raw_mask=mask, fail_closed=fail_closed, exact=exact,
            )
            second = graph.run(
                initial_history=history, requested_q_ref=requested,
                previous_q_ref=previous, previous_velocity=velocity,
                fallback_q_ref=fallback, expected_raw=expected,
                expected_raw_mask=mask, fail_closed=fail_closed, exact=exact,
            )
            torch.testing.assert_close(first.q_ref_sequences, second.q_ref_sequences, rtol=0.0, atol=0.0)
            torch.testing.assert_close(first.pred_states, second.pred_states, rtol=0.0, atol=0.0)
            self.assertTrue(torch.equal(first.expected_raw_sequences, second.expected_raw_sequences))
            self.assertTrue(torch.equal(first.fallback_mask, second.fallback_mask))


if __name__ == "__main__":
    unittest.main()
