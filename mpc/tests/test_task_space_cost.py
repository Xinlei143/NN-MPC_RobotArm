from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
if str(DYNAMICS_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAMICS_ROOT))

from mpc.kinematics_utils import MujocoKinematics
from mpc.planner_rollout import LearnedDynamicsPlanner
from mpc.task_space_cost import (
    ExactTaskSpaceCost,
    TaskSpaceCostConfig,
    TorchSerialKinematics,
    TorchTaskSpaceCost,
)


MODEL_XML = ROOT / "dynamics_modeling" / "ABB_IRB2400.xml"


class ExactTaskSpaceCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        cls.q = np.zeros(6, dtype=np.float64)
        cls.position, cls.rotation = MujocoKinematics(cls.model).forward(cls.q)

    def scorer(self, *, discount: float = 0.95) -> ExactTaskSpaceCost:
        return ExactTaskSpaceCost(
            self.model,
            ee_site_name="ee_site",
            n_joints=6,
            config=TaskSpaceCostConfig(
                w_position=1.0,
                w_orientation=1.0,
                position_scale_m=0.05,
                orientation_scale_rad=0.1,
                temporal_discount=discount,
            ),
        )

    def test_matching_pose_has_zero_cost(self) -> None:
        terms = self.scorer().evaluate(
            torch.zeros((2, 1, 6)),
            self.position[None, :],
            self.rotation[None, :, :],
        )
        torch.testing.assert_close(terms["task_position"], torch.zeros(2), atol=1e-7, rtol=0.0)
        torch.testing.assert_close(terms["task_orientation"], torch.zeros(2), atol=1e-7, rtol=0.0)

    def test_position_and_orientation_are_normalized(self) -> None:
        angle = 0.1
        desired_rotation = self.rotation @ np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        desired_position = self.position.copy()
        desired_position[0] += 0.05
        terms = self.scorer().evaluate(
            torch.zeros((1, 1, 6)),
            desired_position[None, :],
            desired_rotation[None, :, :],
        )
        torch.testing.assert_close(terms["task_position"], torch.ones(1), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(terms["task_orientation"], torch.ones(1), atol=1e-6, rtol=0.0)

    def test_temporal_discount_weights_horizon(self) -> None:
        positions = np.repeat(self.position[None, :], 2, axis=0)
        positions[0, 0] += 0.05
        positions[1, 0] += 0.10
        rotations = np.repeat(self.rotation[None, :, :], 2, axis=0)
        terms = self.scorer(discount=0.5).evaluate(
            torch.zeros((1, 2, 6)),
            positions,
            rotations,
        )
        expected = torch.tensor([(1.0 + 0.5 * 4.0) / 1.5])
        torch.testing.assert_close(terms["task_position"], expected, atol=1e-6, rtol=0.0)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "weights"):
            TaskSpaceCostConfig(w_position=-1.0).validate()
        with self.assertRaisesRegex(ValueError, "scales"):
            TaskSpaceCostConfig(position_scale_m=0.0).validate()


class TorchTaskSpaceCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
        cls.exact_fk = MujocoKinematics(cls.model)
        cls.torch_fk = TorchSerialKinematics(cls.model)
        cls.config = TaskSpaceCostConfig(
            w_position=1.0,
            w_orientation=0.25,
            position_scale_m=0.05,
            orientation_scale_rad=0.1,
            temporal_discount=0.95,
        )

    @staticmethod
    def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
        relative = first.T @ second
        sine = 0.5 * np.linalg.norm(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
        cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
        return float(np.arctan2(sine, cosine))

    def test_batched_fk_matches_mujoco(self) -> None:
        rng = np.random.default_rng(20260726)
        q = rng.uniform(self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1], size=(128, 6))
        positions, rotations = self.torch_fk.forward(torch.as_tensor(q, dtype=torch.float32))
        max_position = 0.0
        max_rotation = 0.0
        for index, joint_position in enumerate(q):
            exact_position, exact_rotation = self.exact_fk.forward(joint_position)
            max_position = max(max_position, float(np.linalg.norm(positions[index].numpy() - exact_position)))
            max_rotation = max(
                max_rotation,
                self.rotation_distance(rotations[index].numpy(), exact_rotation),
            )
        self.assertLessEqual(max_position, 1e-5)
        self.assertLessEqual(max_rotation, 1e-4)

    def test_torch_cost_matches_exact_cost(self) -> None:
        rng = np.random.default_rng(11)
        predicted_q = rng.uniform(
            self.model.jnt_range[:6, 0],
            self.model.jnt_range[:6, 1],
            size=(3, 4, 6),
        )
        desired_q = rng.uniform(
            self.model.jnt_range[:6, 0],
            self.model.jnt_range[:6, 1],
            size=(4, 6),
        )
        poses = [self.exact_fk.forward(value) for value in desired_q]
        desired_positions = np.asarray([pose[0] for pose in poses])
        desired_rotations = np.asarray([pose[1] for pose in poses])
        exact = ExactTaskSpaceCost(
            self.model,
            ee_site_name="ee_site",
            n_joints=6,
            config=self.config,
        ).evaluate(
            torch.as_tensor(predicted_q, dtype=torch.float32),
            desired_positions,
            desired_rotations,
        )
        approximate = TorchTaskSpaceCost(
            self.model,
            ee_site_name="ee_site",
            n_joints=6,
            config=self.config,
            device="cpu",
        ).evaluate(
            torch.as_tensor(predicted_q, dtype=torch.float32),
            desired_positions,
            desired_rotations,
        )
        torch.testing.assert_close(
            approximate["task_position"], exact["task_position"], atol=2e-4, rtol=2e-5
        )
        torch.testing.assert_close(
            approximate["task_orientation"], exact["task_orientation"], atol=2e-4, rtol=2e-5
        )


class _FakeTaskScorer:
    def __init__(self) -> None:
        self.config = SimpleNamespace(w_position=2.0, w_orientation=3.0)

    def evaluate(self, predicted_q, desired_positions, desired_rotations):
        del desired_positions, desired_rotations
        batch = predicted_q.shape[0]
        return {
            "task_position": torch.arange(1, batch + 1, dtype=predicted_q.dtype),
            "task_orientation": torch.ones(batch, dtype=predicted_q.dtype),
        }


class _PlannerUnderTest(LearnedDynamicsPlanner):
    def __init__(self) -> None:
        self.exact_task_space_cost = _FakeTaskScorer()
        self.task_positions_des = np.zeros((1, 3), dtype=np.float64)
        self.task_rotations_des = np.repeat(np.eye(3)[None, :, :], 1, axis=0)

    def evaluate(self, candidate_action, *, project_kinematics_override=None):
        del project_kinematics_override
        batch = candidate_action.shape[0]
        costs = torch.full((batch,), 5.0)
        pred_states = torch.zeros((batch, 2, 12))
        return {
            "costs": costs,
            "cost_terms": {"total": costs.clone()},
            "cost_valid": torch.ones(batch, dtype=torch.bool),
            "pred_states": pred_states,
        }


class ExactPlannerCompositionTests(unittest.TestCase):
    def test_task_cost_is_added_only_by_exact_evaluation(self) -> None:
        planner = _PlannerUnderTest()
        candidates = torch.zeros((2, 1, 6))
        cheap = planner.evaluate(candidates)
        exact = planner.evaluate_exact(candidates)
        torch.testing.assert_close(cheap["costs"], torch.tensor([5.0, 5.0]))
        torch.testing.assert_close(exact["costs"], torch.tensor([10.0, 12.0]))
        torch.testing.assert_close(exact["cost_terms"]["total"], exact["costs"])
        self.assertIn("task_position", exact["cost_terms"])
        self.assertIn("task_orientation", exact["cost_terms"])


if __name__ == "__main__":
    unittest.main()
