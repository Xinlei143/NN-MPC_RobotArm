from __future__ import annotations

import numpy as np

from mpc.asap_shared import PlanPacketStore
from mpc.asap_types import ASAPPlanPacket


def test_packet_store_preserves_uncertainty_telemetry() -> None:
    packet = ASAPPlanPacket(
        plan_id=7, launch_step=10, launch_time_ns=100, activation_step=12,
        activation_time_ns=120, publish_time_ns=110,
        residual_sequence=np.zeros((3, 6), dtype=np.float32),
        predicted_state_sequence=np.zeros((3, 12), dtype=np.float32),
        planning_time_s=0.042, anchor_state=np.zeros(12, dtype=np.float32),
        selection_mode="uncertainty_nominal_fallback", selected_cost=1.5,
        uncertainty_gate=True, uncertainty_score=0.125, uncertainty_max_score=0.250,
        uncertainty_evaluation_time_s=0.006, uncertainty_residual_scale=0.5,
        uncertainty_high_risk=True,
    )
    store = PlanPacketStore()
    store.publish(packet)
    active = store.activate_due(current_step=12, current_time_ns=120)
    assert active is not None
    assert active.uncertainty_gate
    assert active.uncertainty_score == 0.125
    assert active.uncertainty_residual_scale == 0.5
    assert active.uncertainty_high_risk
