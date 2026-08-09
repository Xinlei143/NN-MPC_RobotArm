from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_dynamics.dataset import RolloutDynamicsDataset
from neural_dynamics.models import MLPDynamics
from neural_dynamics.normalization import StandardNormalizer
from neural_dynamics.rollout import load_dynamics_bundle, rollout_dynamics_batch
from neural_dynamics.train_utils import save_checkpoint


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "train_dynamics_for_gradient_tests", ROOT / "dynamics_modeling" / "scripts" / "train_dynamics.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load train_dynamics.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class DynamicsRolloutGradientTests(unittest.TestCase):
    @staticmethod
    def _normalizer(mode: str = "q_ref_minus_q") -> StandardNormalizer:
        normalizer = StandardNormalizer(action_input_mode=mode)
        states = torch.tensor([[0.0, 0.0], [0.2, -0.1], [-0.1, 0.3]])
        actions = torch.tensor([[0.1], [0.4], [-0.2]])
        normalizer.fit(
            states,
            actions,
            torch.tensor([[0.01, 0.02], [0.02, -0.01], [-0.01, 0.03]]),
        )
        return normalizer

    def test_only_rollout_loss_produces_nonzero_parameter_gradient(self) -> None:
        torch.manual_seed(7)
        model = MLPDynamics(state_dim=2, action_dim=1, hidden_size=8, output_dim=2)
        history = torch.tensor([[[0.0, 0.0, 0.1]], [[0.2, -0.1, 0.4]]])
        actions = torch.tensor([[[0.2], [0.3], [0.4]], [[0.5], [0.4], [0.3]]])
        prediction = rollout_dynamics_batch(
            model, self._normalizer(), "mlp", history, actions, 2, "delta_state", 0.1,
            track_grad=True,
        )
        torch.square(prediction[:, 1:]).sum().backward()
        self.assertTrue(any(
            parameter.grad is not None and bool(torch.any(parameter.grad != 0))
            for parameter in model.parameters()
        ))

    def test_inference_rollout_does_not_retain_autograd_graph(self) -> None:
        model = MLPDynamics(state_dim=2, action_dim=1, hidden_size=8, output_dim=2)
        prediction = rollout_dynamics_batch(
            model, self._normalizer(), "mlp", torch.zeros(1, 1, 3), torch.zeros(1, 2, 1),
            2, "delta_state", 0.1,
        )
        self.assertFalse(prediction.requires_grad)

    def test_q_ref_minus_q_encoding_and_legacy_default(self) -> None:
        normalizer = self._normalizer()
        normalizer.state_mean.zero_(); normalizer.state_std.fill_(1.0)
        normalizer.action_mean.zero_(); normalizer.action_std.fill_(1.0)
        encoded = normalizer.normalize_single_input(
            torch.tensor([[1.0, 0.2]]), torch.tensor([[1.5]])
        )
        torch.testing.assert_close(encoded[:, -1:], torch.tensor([[0.5]]))
        state = normalizer.state_dict()
        state.pop("action_input_mode")
        legacy = StandardNormalizer(action_input_mode="q_ref_minus_q")
        legacy.load_state_dict(state)
        self.assertEqual(legacy.action_input_mode, "absolute_q_ref")

    def test_checkpoint_and_normalizer_input_modes_must_match(self) -> None:
        model = MLPDynamics(state_dim=2, action_dim=1, hidden_size=256, output_dim=2)
        normalizer = self._normalizer("q_ref_minus_q")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            normalizer_path = root / "normalizer.pt"
            save_checkpoint(checkpoint, model, {
                "model_type": "mlp", "state_dim": 2, "action_dim": 1, "output_dim": 2,
                "history_len": 1, "target_mode": "delta_state", "control_dt": 0.1,
                "action_input_mode": "absolute_q_ref",
            })
            normalizer.save(normalizer_path)
            with self.assertRaisesRegex(ValueError, "action_input_mode mismatch"):
                load_dynamics_bundle(checkpoint, normalizer_path, "mlp", 1, "cpu")

    def test_micro_batch_gradient_sum_matches_full_effective_batch(self) -> None:
        states = np.linspace(-0.3, 0.4, 16, dtype=np.float32).reshape(8, 2)
        actions = np.linspace(-0.2, 0.25, 8, dtype=np.float32).reshape(8, 1)
        next_states = states + np.concatenate([0.02 * actions, -0.01 * actions], axis=1)
        dataset = RolloutDynamicsDataset(
            states, actions, next_states, model_type="mlp", target_mode="delta_state", rollout_steps=3
        )
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        normalizer = StandardNormalizer()
        normalizer.fit(
            torch.as_tensor(states), torch.as_tensor(actions), torch.as_tensor(next_states - states)
        )
        torch.manual_seed(11)
        full = MLPDynamics(state_dim=2, action_dim=1, hidden_size=8, output_dim=2)
        micro = MLPDynamics(state_dim=2, action_dim=1, hidden_size=8, output_dim=2)
        micro.load_state_dict(full.state_dict())
        common = dict(
            loader=loader, normalizer=normalizer, state_dim=2, model_type="mlp",
            device=torch.device("cpu"), target_mode="delta_state", control_dt=0.1,
            rollout_loss_weight=0.2,
        )
        TRAIN.run_epoch(full, optimizer=torch.optim.SGD(full.parameters(), lr=1e-4), **common)
        TRAIN.run_epoch(
            micro, optimizer=torch.optim.SGD(micro.parameters(), lr=1e-4), micro_batch_size=2, **common
        )
        for expected, actual in zip(full.parameters(), micro.parameters()):
            torch.testing.assert_close(expected, actual, rtol=2e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
