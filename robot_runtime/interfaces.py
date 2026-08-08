from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


DiagnosticValue = np.ndarray | float | int | bool | str


@dataclass(frozen=True)
class RobotState:
    """One tick-start measurement in the hardware control coordinate system."""

    q_ctrl: np.ndarray
    dq_ctrl: np.ndarray
    timestamp_ns: int
    timestamp_uncertainty_ns: int
    read_start_ns: int
    read_end_ns: int
    tick_index: int
    history_generation: int
    estimator_generation: int
    valid: bool
    validity_flags: tuple[str, ...] = ()
    diagnostics: dict[str, DiagnosticValue] = field(default_factory=dict)

    @property
    def q(self) -> np.ndarray:
        return self.q_ctrl

    @property
    def dq(self) -> np.ndarray:
        return self.dq_ctrl

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate((self.q_ctrl, self.dq_ctrl)).astype(np.float32)


@dataclass(frozen=True)
class CommandResult:
    """A locally successful bus transmission, not a per-motor acknowledgement."""

    requested_q_ref: np.ndarray
    projected_q_ref: np.ndarray
    transmitted_q_ref: np.ndarray
    tx_goal_position_raw: np.ndarray
    tx_local_success: bool
    command_tick_index: int
    write_start_ns: int
    write_end_ns: int
    projection_flags: tuple[str, ...] = ()
    command_delivery_uncertain: bool = False
    diagnostics: dict[str, DiagnosticValue] = field(default_factory=dict)

    @property
    def sent_q_ref(self) -> np.ndarray:
        """Compatibility alias; callers should prefer transmitted_q_ref."""
        return self.transmitted_q_ref


@runtime_checkable
class RobotBackend(Protocol):
    n_joints: int
    control_dt: float
    joint_low: np.ndarray
    joint_high: np.ndarray

    def connect(self) -> None: ...
    def read_state(self, *, tick_index: int = 0) -> RobotState: ...
    def send_joint_targets(self, q_ref: np.ndarray, *, tick_index: int = 0) -> CommandResult: ...
    def move_to_configuration(self, target_q: np.ndarray, duration_s: float) -> None: ...
    def disable_torque(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class HardwareEnvelopeBackend(RobotBackend, Protocol):
    """Optional direct-only wide envelope used by audited workspace collection."""

    def prepare_hardware_motion(self) -> RobotState: ...
    def send_hardware_joint_targets(self, q_ref: np.ndarray, *, tick_index: int = 0) -> CommandResult: ...
    def configure_workspace_projector(self, workspace_low: np.ndarray, workspace_high: np.ndarray) -> None: ...
    def send_workspace_joint_targets(self, q_ref: np.ndarray, *, tick_index: int = 0) -> CommandResult: ...
