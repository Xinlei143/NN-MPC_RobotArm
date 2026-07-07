from __future__ import annotations

import unittest

import torch

from mpc.cost_functions import JointSpaceCostConfig, joint_space_tracking_cost


def make_pred_states(q_pred: torch.Tensor, dq_pred: torch.Tensor | None = None, initial_q: torch.Tensor | None = None) -> torch.Tensor:
    if q_pred.ndim != 3:
        raise ValueError("q_pred must have shape [batch, horizon, n_joints]")
    batch, horizon, n_joints = q_pred.shape
    if dq_pred is None:
        dq_pred = torch.zeros_like(q_pred)
    if initial_q is None:
        initial_q = torch.zeros((batch, n_joints), dtype=q_pred.dtype)
    initial_dq = torch.zeros_like(initial_q)
    pred_states = torch.zeros((batch, horizon + 1, 2 * n_joints), dtype=q_pred.dtype)
    pred_states[:, 0, :n_joints] = initial_q
    pred_states[:, 0, n_joints:] = initial_dq
    pred_states[:, 1:, :n_joints] = q_pred
    pred_states[:, 1:, n_joints:] = dq_pred
    return pred_states


class CostFunctionTests(unittest.TestCase):
    def test_joint_tracking_uses_horizon_normalized_mean_error(self) -> None:
        q_des = torch.tensor([[[0.0, 0.00], [1.0, 0.02]]], dtype=torch.float32)
        q_scale = torch.tensor([[[0.2, 0.04]]], dtype=torch.float32)
        q_pred = q_des + q_scale
        pred_states = make_pred_states(q_pred)

        cost = joint_space_tracking_cost(
            pred_states=pred_states,
            q_des=q_des,
            dq_des=None,
            actuator_q_ref=q_pred,
            delta_q_ref=torch.zeros_like(q_pred),
            previous_q_ref=torch.zeros(2),
            joint_low=torch.full((2,), -10.0),
            joint_high=torch.full((2,), 10.0),
            config=JointSpaceCostConfig(
                w_q=1.0,
                w_dq=0.0,
                w_u_offset=0.0,
                w_dqref=0.0,
                w_ddqref=0.0,
                w_terminal=0.0,
                w_joint_limit=0.0,
                q_amp_fraction=0.2,
                q_tol=0.04,
            ),
        )

        self.assertTrue(torch.allclose(cost, torch.tensor([1.0])))

    def test_actuator_offset_penalizes_target_relative_to_predicted_position(self) -> None:
        q_pred = torch.tensor([[[1.0, -0.5], [1.2, -0.4]]], dtype=torch.float32)
        pred_states = make_pred_states(q_pred, initial_q=torch.zeros((1, 2), dtype=torch.float32))

        cost = joint_space_tracking_cost(
            pred_states=pred_states,
            q_des=q_pred,
            dq_des=None,
            actuator_q_ref=q_pred.clone(),
            delta_q_ref=torch.zeros_like(q_pred),
            previous_q_ref=torch.zeros(2),
            joint_low=torch.full((2,), -10.0),
            joint_high=torch.full((2,), 10.0),
            config=JointSpaceCostConfig(
                w_q=0.0,
                w_dq=0.0,
                w_u_offset=1.0,
                w_dqref=0.0,
                w_ddqref=0.0,
                w_terminal=0.0,
                w_joint_limit=0.0,
                u_offset_scale=0.2,
            ),
        )

        self.assertTrue(torch.allclose(cost, torch.tensor([0.0])))

    def test_command_smoothness_separates_first_and_second_differences(self) -> None:
        actuator_q_ref = torch.tensor([[[1.0], [3.0], [6.0]]], dtype=torch.float32)
        pred_states = make_pred_states(torch.zeros_like(actuator_q_ref))

        cost = joint_space_tracking_cost(
            pred_states=pred_states,
            q_des=torch.zeros_like(actuator_q_ref),
            dq_des=None,
            actuator_q_ref=actuator_q_ref,
            delta_q_ref=torch.zeros_like(actuator_q_ref),
            previous_q_ref=torch.zeros(1),
            joint_low=torch.full((1,), -10.0),
            joint_high=torch.full((1,), 10.0),
            config=JointSpaceCostConfig(
                w_q=0.0,
                w_dq=0.0,
                w_u_offset=0.0,
                w_dqref=1.0,
                w_ddqref=1.0,
                w_terminal=0.0,
                w_joint_limit=0.0,
                dqref_scale=1.0,
                ddqref_scale=1.0,
            ),
        )

        expected_dqref = torch.tensor((1.0**2 + 2.0**2 + 3.0**2) / 3.0)
        expected_ddqref = torch.tensor((1.0**2 + 1.0**2) / 2.0)
        self.assertTrue(torch.allclose(cost, (expected_dqref + expected_ddqref).unsqueeze(0)))

    def test_joint_limit_barrier_penalizes_close_to_limit_before_violation(self) -> None:
        q_pred = torch.tensor(
            [
                [[0.0], [0.0]],
                [[0.95], [0.96]],
            ],
            dtype=torch.float32,
        )
        pred_states = make_pred_states(q_pred)

        cost = joint_space_tracking_cost(
            pred_states=pred_states,
            q_des=q_pred,
            dq_des=None,
            actuator_q_ref=q_pred,
            delta_q_ref=torch.zeros_like(q_pred),
            previous_q_ref=torch.zeros(1),
            joint_low=torch.full((1,), -1.0),
            joint_high=torch.full((1,), 1.0),
            config=JointSpaceCostConfig(
                w_q=0.0,
                w_dq=0.0,
                w_u_offset=0.0,
                w_dqref=0.0,
                w_ddqref=0.0,
                w_terminal=0.0,
                w_joint_limit=1.0,
                joint_limit_safe_margin=0.08,
                joint_limit_temp=0.02,
            ),
        )

        self.assertGreater(float(cost[1]), float(cost[0]))
        self.assertGreater(float(cost[1]), 0.0)

    def test_velocity_damping_mode_ignores_dq_des(self) -> None:
        q_pred = torch.zeros((1, 2, 1), dtype=torch.float32)
        dq_pred = torch.full_like(q_pred, 0.5)
        pred_states = make_pred_states(q_pred, dq_pred=dq_pred)

        cost = joint_space_tracking_cost(
            pred_states=pred_states,
            q_des=q_pred,
            dq_des=torch.full_like(q_pred, 100.0),
            actuator_q_ref=q_pred,
            delta_q_ref=torch.zeros_like(q_pred),
            previous_q_ref=torch.zeros(1),
            joint_low=torch.full((1,), -10.0),
            joint_high=torch.full((1,), 10.0),
            config=JointSpaceCostConfig(
                w_q=0.0,
                w_dq=1.0,
                w_u_offset=0.0,
                w_dqref=0.0,
                w_ddqref=0.0,
                w_terminal=0.0,
                w_joint_limit=0.0,
                dq_scale=0.5,
                velocity_cost_mode="damping",
            ),
        )

        self.assertTrue(torch.allclose(cost, torch.tensor([1.0])))


if __name__ == "__main__":
    unittest.main()
