from __future__ import annotations

import numpy as np

from mpc.asap_shared import LatestSnapshotStore, PlanPacketStore
from mpc.asap_types import ASAPPlanPacket
from mpc.delay_aware import project_packet_command_sequence_np
from robot_runtime.asap_adapter import ASAPStorePlannerAdapter
from robot_runtime.runner import PlannerCommand, RealControlMode, active_mpc_gate_open, compose_requested_command


def test_real_adapter_preserves_projected_absolute_packet_command() -> None:
    packets = PlanPacketStore()
    adapter = ASAPStorePlannerAdapter(
        snapshots=LatestSnapshotStore(),
        packets=packets,
        n_joints=2,
        record_ood_tokens=False,
    )
    q_ref = np.asarray([[0.41, -0.19], [0.42, -0.18]], dtype=np.float32)
    residual = np.asarray([[0.03, 0.01], [0.02, 0.02]], dtype=np.float32)
    predicted = np.zeros((3, 4), dtype=np.float32)
    packets.publish(
        ASAPPlanPacket(
            plan_id=1,
            launch_step=0,
            launch_time_ns=0,
            activation_step=0,
            activation_time_ns=0,
            publish_time_ns=0,
            residual_sequence=residual,
            predicted_state_sequence=predicted,
            planning_time_s=0.0,
            anchor_state=np.zeros(4, dtype=np.float32),
            selection_mode="best",
            selected_cost=0.0,
            q_ref_sequence=q_ref,
            requested_residual_sequence=residual.copy(),
        )
    )

    adapter.current_tick = 0
    command = adapter.latest()

    assert command is not None
    np.testing.assert_allclose(command.residual, residual[0])
    np.testing.assert_allclose(command.absolute_q_ref, q_ref[0])


def test_real_adapter_initializes_home_velocity_and_converts_position_delta() -> None:
    adapter = ASAPStorePlannerAdapter(
        snapshots=LatestSnapshotStore(),
        packets=PlanPacketStore(),
        n_joints=1,
        control_dt=0.1,
        record_ood_tokens=False,
    )
    adapter.submit(
        tick_index=0,
        state_timestamp_ns=0,
        states=np.zeros((1, 2), dtype=np.float32),
        commands=np.asarray([[0.5]], dtype=np.float32),
        history_generation=0,
    )
    np.testing.assert_allclose(adapter.previous_q_ref, [0.5])
    np.testing.assert_allclose(adapter.previous_velocity, [0.0])

    adapter.submit(
        tick_index=1,
        state_timestamp_ns=1,
        states=np.zeros((1, 2), dtype=np.float32),
        commands=np.asarray([[0.6]], dtype=np.float32),
        history_generation=0,
    )
    np.testing.assert_allclose(adapter.previous_velocity, [1.0], atol=1e-6)


def test_runner_prefers_absolute_projected_command_over_live_nominal() -> None:
    planner_command = PlannerCommand(
        residual=np.asarray([0.03, 0.01], dtype=np.float32),
        history_generation=0,
        activation_tick=0,
        publication_tick=0,
        absolute_q_ref=np.asarray([0.41, -0.19], dtype=np.float32),
    )
    requested = compose_requested_command(
        np.asarray([0.50, -0.25], dtype=np.float32),
        planner_command.residual,
        planner_command,
        applied=True,
    )
    np.testing.assert_allclose(requested, planner_command.absolute_q_ref)


def test_active_mpc_startup_gate_blocks_static_prefix_only() -> None:
    assert not active_mpc_gate_open(RealControlMode.ACTIVE_MPC, 82, 83)
    assert active_mpc_gate_open(RealControlMode.ACTIVE_MPC, 83, 83)
    assert active_mpc_gate_open(RealControlMode.SHADOW_MPC, 0, 83)


def test_packet_projection_returns_absolute_commands_and_chains_velocity() -> None:
    commands, final_velocity = project_packet_command_sequence_np(
        nominal_q_ref=np.asarray([[0.50], [0.50]], dtype=np.float32),
        requested_residual=np.asarray([[0.08], [0.08]], dtype=np.float32),
        previous_command=np.asarray([0.50], dtype=np.float32),
        previous_velocity=np.asarray([0.0], dtype=np.float32),
        joint_low=np.asarray([-2.0], dtype=np.float32),
        joint_high=np.asarray([2.0], dtype=np.float32),
        joint_limit_margin=0.0,
        velocity_limit=np.asarray([10.0], dtype=np.float32),
        acceleration_limit=np.asarray([300.0], dtype=np.float32),
        control_dt=0.01,
    )
    np.testing.assert_allclose(commands[:, 0], [0.53, 0.58], atol=1e-6)
    np.testing.assert_allclose(final_velocity, [5.0], atol=1e-5)
