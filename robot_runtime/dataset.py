from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_runtime.config import file_sha256

from robot_runtime.interfaces import CommandResult, RobotState


@dataclass(frozen=True)
class TransitionValidity:
    valid_target: bool
    command_delivery_uncertain: bool
    termination_reason: str = ""


class RealDatasetRecorder:
    """NPZ v2 recorder retaining invalid rows for audit while flagging training targets."""

    def __init__(self, session_id: str, *, source_id: int = 0, control_dt: float = 1 / 30):
        self.session_id = str(session_id)
        self.source_id = int(source_id)
        self.control_dt = float(control_dt)
        self._rows: list[dict[str, Any]] = []

    def append(self, state: RobotState, command: CommandResult, next_state: RobotState,
               *, episode_id: int, split_group_id: int, motion_mode_id: int,
               validity: TransitionValidity, requested_q_ref: np.ndarray | None = None,
               motion_variant: str | None = None,
               table_safety: dict[str, Any] | None = None) -> None:
        q_ref = command.transmitted_q_ref
        valid = bool(validity.valid_target and state.valid and next_state.valid and command.tx_local_success
                     and not validity.command_delivery_uncertain
                     and state.history_generation == next_state.history_generation
                     and state.estimator_generation == next_state.estimator_generation)
        def diagnostic(name: str, length: int = 6) -> np.ndarray:
            return np.asarray(state.diagnostics.get(name, np.full(length, np.nan)))
        table = {} if table_safety is None else dict(table_safety)
        self._rows.append({
            "states": state.vector, "actions": q_ref.copy(), "next_states": next_state.vector,
            "q_ref": q_ref.copy(), "requested_q_ref": command.requested_q_ref.copy(),
            "projected_q_ref": command.projected_q_ref.copy(), "transmitted_q_ref": q_ref.copy(),
            "delta_q_ref": q_ref - state.q_ctrl, "tx_goal_position_raw": command.tx_goal_position_raw.copy(),
            "measured_q": state.q_ctrl.copy(), "estimated_dq": state.dq_ctrl.copy(),
            "raw_dq": np.asarray(state.diagnostics.get("raw_dq", np.full_like(state.q_ctrl, np.nan))),
            "raw_present_velocity": diagnostic("raw_present_velocity", 5),
            "state_timestamp_ns": state.timestamp_ns, "state_timestamp_uncertainty_ns": state.timestamp_uncertainty_ns,
            "action_write_start_ns": command.write_start_ns, "action_write_end_ns": command.write_end_ns,
            "next_state_timestamp_ns": next_state.timestamp_ns,
            "actual_dt": (next_state.timestamp_ns - state.timestamp_ns) * 1e-9,
            "read_latency_s": (state.read_end_ns - state.read_start_ns) * 1e-9,
            "write_latency_s": (command.write_end_ns - command.write_start_ns) * 1e-9,
            "deadline_miss": bool((command.write_end_ns - state.read_start_ns) * 1e-9 > self.control_dt),
            "motor_current_raw": diagnostic("motor_current_raw"),
            "motor_load_raw": diagnostic("motor_load_raw"),
            "motor_voltage": diagnostic("motor_voltage_v"),
            "motor_temperature": diagnostic("motor_temperature"),
            "motor_temperature_filtered": diagnostic("motor_temperature_filtered"),
            "temperature_filter_status": str(state.diagnostics.get("temperature_filter_status", "")),
            # Goal Position is sampled at the low-rate diagnostic cadence.
            # Retain both its latest six-motor readback and its explicit
            # comparison result so gripper latch and uncertain-command
            # intervals remain independently auditable offline.
            "goal_position_readback_raw": diagnostic("goal_position_readback_raw"),
            "goal_readback_mismatch": bool(state.diagnostics.get("goal_readback_mismatch", False)),
            "read_retry_count": int(state.diagnostics.get("read_retry_count", -1)),
            "diagnostic_update_timestamp_ns": int(state.diagnostics.get("diagnostic_update_timestamp_ns", 0)),
            "diagnostic_sample_age_s": float(state.diagnostics.get("diagnostic_sample_age_s", np.inf)),
            "diagnostic_sample_count": int(state.diagnostics.get("diagnostic_sample_count", 0)),
            "valid_target": valid, "command_delivery_uncertain": validity.command_delivery_uncertain,
            "tx_local_success": command.tx_local_success, "state_validity_flags": "|".join(state.validity_flags),
            "next_state_validity_flags": "|".join(next_state.validity_flags),
            "projection_flags": "|".join(command.projection_flags), "termination_reasons": validity.termination_reason,
            "command_projection_active": bool(command.projection_flags),
            "safety_flags": "|".join((*state.validity_flags, *command.projection_flags)),
            "episode_ids": int(episode_id), "split_group_ids": int(split_group_id), "source_ids": self.source_id,
            "session_ids": self.session_id, "motion_mode_ids": int(motion_mode_id),
            "motion_variant": "" if motion_variant is None else str(motion_variant),
            "tick_index": state.tick_index, "command_tick_index": command.command_tick_index,
            "history_generation": state.history_generation, "next_history_generation": next_state.history_generation,
            "estimator_generation": state.estimator_generation, "next_estimator_generation": next_state.estimator_generation,
            "state_estimator_version": "backward_difference_causal_lpf_4hz_v1",
            "table_predicted_clearance_m": float(table.get("predicted_clearance_m", np.nan)),
            "table_effective_clearance_m": float(table.get("effective_clearance_m", np.nan)),
            "table_nearest_component": str(table.get("nearest_component", "")),
            "table_safety_identity": str(table.get("identity", "")),
            "table_safety_violation": str(table.get("violation", "")),
        })

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._rows:
            return {}
        return {key: np.asarray([row[key] for row in self._rows]) for key in self._rows[0]}

    def mark_command_uncertain_since_tick(self, first_tick: int) -> None:
        """Retroactively invalidate the interval bounded by Goal_Position readbacks."""
        for row in self._rows:
            if int(row["command_tick_index"]) >= int(first_tick):
                row["command_delivery_uncertain"] = True
                row["valid_target"] = False

    def save(self, output: str | Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **self.arrays())
        manifest_path = target.with_suffix(".manifest.json")
        payload = {"schema_version": 2, "action_semantics": "software_transmitted_absolute_position_target",
                   "motor_acknowledged": False, "sample_count": len(self._rows), **manifest}
        payload["dataset"] = {"path": str(target), "sha256": file_sha256(target)}
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return target, manifest_path
