from __future__ import annotations

import unittest

import numpy as np
import torch

from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.residual_parameterization import ResidualParameterization


class ResidualParameterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameterizer = ResidualParameterization.create(
            rollout_horizon=32,
            decision_horizon=8,
            mode="linear_control_points",
            device="cpu",
        )

    def test_constant_and_endpoint_semantics(self) -> None:
        points = torch.linspace(-0.8, 0.9, 8).view(1, 8, 1)
        expanded = self.parameterizer.expand(points)
        torch.testing.assert_close(expanded[:, 0], points[:, 0])
        torch.testing.assert_close(expanded[:, -1], points[:, -1])
        self.assertGreaterEqual(float(expanded.min()), float(points.min()))
        self.assertLessEqual(float(expanded.max()), float(points.max()))
        torch.testing.assert_close(
            self.parameterizer.expand(torch.ones((2, 8, 6))),
            torch.ones((2, 32, 6)),
        )

    def test_compress_expand_recovers_control_points(self) -> None:
        generator = torch.Generator().manual_seed(7)
        points = 1.6 * torch.rand((3, 8, 6), generator=generator) - 0.8
        reconstructed = self.parameterizer.compress(self.parameterizer.expand(points))
        torch.testing.assert_close(reconstructed, points, atol=2e-6, rtol=0.0)

    def test_tick_shift_holds_tail_and_pins_endpoints(self) -> None:
        points = torch.linspace(-0.7, 0.6, 8).view(8, 1)
        full = self.parameterizer.expand(points)
        shifted_points = self.parameterizer.shift(points, 5)
        shifted = self.parameterizer.expand(shifted_points)
        torch.testing.assert_close(shifted[0], full[5], atol=1e-6, rtol=0.0)
        torch.testing.assert_close(shifted[-1], full[-1], atol=1e-6, rtol=0.0)


class _ExpandedPlanner:
    def __init__(self) -> None:
        self.parameterizer = ResidualParameterization.create(
            rollout_horizon=32,
            decision_horizon=8,
            mode="linear_control_points",
            device="cpu",
        )

    def expand_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.parameterizer.expand(action)

    def shift_control_points(self, action: torch.Tensor, steps: int) -> torch.Tensor:
        return self.parameterizer.shift(action, steps)

    def evaluate(self, action: torch.Tensor, **_kwargs) -> dict[str, torch.Tensor]:
        expanded = self.expand_action(action)
        costs = expanded.square().mean(dim=(1, 2))
        batch = action.shape[0]
        return {
            "costs": costs,
            "q_ref_sequences": expanded,
            "residual_sequences": expanded,
            "normalized_residual_sequences": expanded,
            "cost_terms": {"total": costs},
            "pred_states": torch.zeros((batch, 33, 12)),
        }


class ReducedHorizonControllerTests(unittest.TestCase):
    def test_controller_keeps_latent_and_rollout_shapes_separate(self) -> None:
        controller = CEMMPCController(
            CEMMPCConfig(
                horizon=32,
                decision_horizon=8,
                action_dim=6,
                num_samples=8,
                cem_iters=1,
                reset_std_each_step=True,
                force_baseline_candidate=True,
                execute="lowest_cost",
                device="cpu",
            ),
            _ExpandedPlanner(),
            np.full(6, -2.0, dtype=np.float32),
            np.full(6, 2.0, dtype=np.float32),
        )
        result = controller.plan(np.zeros(12, dtype=np.float32), np.zeros(6, dtype=np.float32))
        self.assertFalse(result.failure, result.failure_reason)
        self.assertEqual(result.selected_control_points.shape, (8, 6))
        self.assertEqual(result.selected_action_sequence.shape, (32, 6))
        self.assertEqual(result.selected_residual_sequence.shape, (32, 6))
        self.assertEqual(result.selected_predicted_state_sequence.shape, (33, 12))

    def test_reduced_horizon_requires_std_reset(self) -> None:
        with self.assertRaisesRegex(ValueError, "reset_std_each_step"):
            CEMMPCController(
                CEMMPCConfig(
                    horizon=32,
                    decision_horizon=8,
                    action_dim=6,
                    reset_std_each_step=False,
                ),
                _ExpandedPlanner(),
                np.full(6, -2.0, dtype=np.float32),
                np.full(6, 2.0, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
