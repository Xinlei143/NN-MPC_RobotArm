from __future__ import annotations

import time
import numpy as np

from mpc.asap_shared import LatestSnapshotStore, PlanPacketStore
from mpc.asap_types import PlanningSnapshot
from robot_runtime.runner import PlannerCommand
from robot_runtime.ood import RobustEnvelope


class ASAPStorePlannerAdapter:
    """Connect the real tick loop to the existing CUDA ASAP worker stores."""

    def __init__(self, snapshots: LatestSnapshotStore, packets: PlanPacketStore, n_joints: int,
                 ood_envelope: RobustEnvelope | None = None):
        self.snapshots, self.packets, self.n_joints = snapshots, packets, int(n_joints)
        self.request_id = 0
        self.current_tick = 0
        self.generation = 0
        self.previous_q_ref = np.zeros(n_joints, dtype=np.float32)
        self.previous_velocity = np.zeros(n_joints, dtype=np.float32)
        self._zeros = np.zeros(n_joints, dtype=np.float32)
        self.ood_envelope = ood_envelope
        self._executed_ood_valid = True

    def submit(self, tick_index: int, state_timestamp_ns: int, states: np.ndarray, commands: np.ndarray,
               history_generation: int) -> None:
        self.current_tick, self.generation = int(tick_index), int(history_generation)
        if commands.size:
            new_q = np.asarray(commands[-1], dtype=np.float32)
            self.previous_velocity = new_q - self.previous_q_ref
            self.previous_q_ref = new_q.copy()
        if self.ood_envelope is not None:
            token = np.concatenate((np.asarray(states[-1]), self.previous_q_ref))
            self._executed_ood_valid = bool(self.ood_envelope.contains(token))
        self.snapshots.publish(PlanningSnapshot(
            request_id=self.request_id, launch_step=self.current_tick, launch_time_ns=int(state_timestamp_ns),
            states_history=np.asarray(states, dtype=np.float32), command_history=np.asarray(commands, dtype=np.float32),
            previous_q_ref=self.previous_q_ref.copy(), previous_q_ref_velocity=self.previous_velocity.copy(),
            previous_requested_mpc_residual=self._zeros.copy(), previous_requested_mpc_residual_velocity=self._zeros.copy(),
            previous_command_nominal_offset=self._zeros.copy(), previous_command_nominal_offset_velocity=self._zeros.copy(),
            packet_schedule=self.packets.schedule(), history_generation=self.generation,
        ))
        self.request_id += 1

    def latest(self) -> PlannerCommand | None:
        packet = self.packets.activate_due(self.current_tick, time.perf_counter_ns())
        if packet is None or packet.history_generation != self.generation:
            return None
        index = packet.index_at(self.current_tick)
        if index is None:
            return None
        ood_valid = self._executed_ood_valid
        if self.ood_envelope is not None and packet.q_ref_sequence.size and packet.predicted_state_sequence.size:
            length = min(len(packet.q_ref_sequence), len(packet.predicted_state_sequence))
            future_tokens = np.concatenate((packet.predicted_state_sequence[:length], packet.q_ref_sequence[:length]), axis=1)
            ood_valid = ood_valid and bool(np.all(self.ood_envelope.contains(future_tokens)))
        return PlannerCommand(packet.residual_sequence[index].copy(), packet.history_generation,
                              packet.activation_step, packet.publication_tick, ood_valid)

    def clear(self, history_generation: int) -> None:
        self.generation = int(history_generation)
        self.packets.clear()
