"""Building blocks for latency-aware multi-rate residual MPC.

The helpers deliberately separate the Direct-IK nominal from the slower MPC
correction.  This makes ``correction == 0`` an exact Direct-IK command, which
is important when a delayed plan expires or the planner fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mpc.constraints import clip_to_joint_limits, project_position_command_sequence


@dataclass(frozen=True)
class DelayedPlanPacket:
    """A CEM solution scheduled to become valid at a virtual future step."""

    launch_step: int
    activation_step: int
    residual_sequence: np.ndarray
    predicted_state_sequence: np.ndarray
    planning_time_s: float
    mode: str

    @property
    def horizon(self) -> int:
        return int(self.residual_sequence.shape[0])

    def index_at(self, step: int) -> int | None:
        index = int(step) - self.activation_step
        return index if 0 <= index < self.horizon else None


def corrected_direct_ik_command(
    nominal_q_des: torch.Tensor,
    correction: torch.Tensor,
    previous_q_ref: torch.Tensor,
    previous_q_ref_velocity: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    joint_limit_margin: float,
    velocity_limit: torch.Tensor,
    acceleration_limit: torch.Tensor,
    control_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a bounded correction while preserving exact Direct IK at zero.

    A zero correction bypasses the command-rate projection entirely, so a
    fallback cannot inherit a stale MPC command.  Nonzero corrections are
    projected with the *physical* limits supplied by the caller.
    """
    nominal = clip_to_joint_limits(nominal_q_des, joint_low, joint_high, joint_limit_margin)
    requested = clip_to_joint_limits(nominal + correction, joint_low, joint_high, joint_limit_margin)
    if bool(torch.all(torch.abs(correction) <= 1e-8)):
        return nominal, torch.zeros_like(correction)
    command = project_position_command_sequence(
        requested.view(1, 1, -1),
        previous_q_ref=previous_q_ref,
        previous_q_ref_velocity=previous_q_ref_velocity,
        control_dt=control_dt,
        velocity_limit=velocity_limit,
        acceleration_limit=acceleration_limit,
        joint_low=joint_low,
        joint_high=joint_high,
        joint_limit_margin=joint_limit_margin,
    )[0, 0]
    return command, command - nominal


def feedback_correction(
    predicted_state: np.ndarray,
    measured_state: np.ndarray,
    kq: float,
    kdq: float,
    max_abs: np.ndarray,
) -> np.ndarray:
    """Small position-command correction used by ASAP/tube feedback."""
    predicted = np.asarray(predicted_state, dtype=np.float32)
    measured = np.asarray(measured_state, dtype=np.float32)
    n_joints = max_abs.shape[0]
    if predicted.shape != measured.shape or predicted.shape != (2 * n_joints,):
        raise ValueError("predicted_state and measured_state must have shape [2 * n_joints]")
    correction = float(kq) * (predicted[:n_joints] - measured[:n_joints])
    correction += float(kdq) * (predicted[n_joints:] - measured[n_joints:])
    return np.clip(correction, -max_abs, max_abs).astype(np.float32)
