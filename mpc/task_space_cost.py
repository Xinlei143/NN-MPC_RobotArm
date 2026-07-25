from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mpc.kinematics_utils import MujocoKinematics, rotation_log_vector


@dataclass(frozen=True)
class TaskSpaceCostConfig:
    """Normalized TCP pose costs used only for exact final-pool selection."""

    w_position: float = 0.0
    w_orientation: float = 0.0
    position_scale_m: float = 0.05
    orientation_scale_rad: float = np.deg2rad(5.0)
    temporal_discount: float = 0.95

    def validate(self) -> None:
        values = (
            self.w_position,
            self.w_orientation,
            self.position_scale_m,
            self.orientation_scale_rad,
            self.temporal_discount,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("task-space cost configuration must contain only finite values")
        if self.w_position < 0.0 or self.w_orientation < 0.0:
            raise ValueError("task-space cost weights must be non-negative")
        if self.position_scale_m <= 0.0 or self.orientation_scale_rad <= 0.0:
            raise ValueError("task-space cost scales must be positive")
        if not 0.0 < self.temporal_discount <= 1.0:
            raise ValueError("task-space temporal_discount must be in (0, 1]")


class ExactTaskSpaceCost:
    """CPU MuJoCo FK scorer for the small CEM exact final pool."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        ee_site_name: str,
        n_joints: int,
        config: TaskSpaceCostConfig,
    ) -> None:
        config.validate()
        self.config = config
        self.kinematics = MujocoKinematics(
            model=model,
            ee_site_name=ee_site_name,
            n_joints=n_joints,
        )

    def evaluate(
        self,
        predicted_q: torch.Tensor,
        desired_positions: np.ndarray,
        desired_rotations: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Return one normalized position/orientation cost per candidate."""

        if predicted_q.ndim != 3:
            raise ValueError(
                "predicted_q must have shape [batch, horizon, n_joints], "
                f"got {tuple(predicted_q.shape)}"
            )
        batch_size, horizon, n_joints = predicted_q.shape
        if n_joints != self.kinematics.n_joints:
            raise ValueError(
                f"predicted_q has {n_joints} joints, expected {self.kinematics.n_joints}"
            )
        positions = np.asarray(desired_positions, dtype=np.float64)
        rotations = np.asarray(desired_rotations, dtype=np.float64)
        if positions.shape != (horizon, 3):
            raise ValueError(
                f"desired_positions must have shape ({horizon}, 3), got {positions.shape}"
            )
        if rotations.shape != (horizon, 3, 3):
            raise ValueError(
                f"desired_rotations must have shape ({horizon}, 3, 3), got {rotations.shape}"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rotations)):
            raise ValueError("task-space targets must contain only finite values")

        q_values = predicted_q.detach().cpu().numpy().astype(np.float64, copy=False)
        position_error_sq = np.empty((batch_size, horizon), dtype=np.float64)
        orientation_error_sq = np.empty((batch_size, horizon), dtype=np.float64)
        for candidate_idx in range(batch_size):
            for step_idx in range(horizon):
                actual_position, actual_rotation = self.kinematics.forward(
                    q_values[candidate_idx, step_idx]
                )
                position_error = (
                    actual_position - positions[step_idx]
                ) / self.config.position_scale_m
                orientation_error = rotation_log_vector(
                    rotations[step_idx].T @ actual_rotation
                ) / self.config.orientation_scale_rad
                position_error_sq[candidate_idx, step_idx] = float(
                    np.dot(position_error, position_error)
                )
                orientation_error_sq[candidate_idx, step_idx] = float(
                    np.dot(orientation_error, orientation_error)
                )

        weights = np.power(
            self.config.temporal_discount,
            np.arange(horizon, dtype=np.float64),
        )
        weights /= np.sum(weights)
        position_cost = position_error_sq @ weights
        orientation_cost = orientation_error_sq @ weights
        return {
            "task_position": torch.as_tensor(
                position_cost, device=predicted_q.device, dtype=predicted_q.dtype
            ),
            "task_orientation": torch.as_tensor(
                orientation_cost, device=predicted_q.device, dtype=predicted_q.dtype
            ),
        }
