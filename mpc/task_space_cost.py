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


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


class TorchSerialKinematics:
    """Batched Torch FK for the MuJoCo serial chain ending at a TCP site."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        ee_site_name: str = "ee_site",
        n_joints: int = 6,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        if site_id < 0:
            raise ValueError(f"MuJoCo model does not contain TCP site {ee_site_name!r}")
        controlled_joint_ids = list(range(n_joints))
        controlled_bodies = {int(model.jnt_bodyid[joint_id]): joint_id for joint_id in controlled_joint_ids}
        if len(controlled_bodies) != n_joints:
            raise ValueError("Torch FK requires one controlled hinge joint per serial-chain body")
        chain: list[int] = []
        body_id = int(model.site_bodyid[site_id])
        while body_id != 0:
            chain.append(body_id)
            body_id = int(model.body_parentid[body_id])
        chain.reverse()
        if not set(controlled_bodies).issubset(chain):
            raise ValueError("all controlled joints must lie on the TCP serial chain")

        self.n_joints = int(n_joints)
        self.device = torch.device(device)
        self.dtype = dtype
        self.body_positions = [
            torch.as_tensor(model.body_pos[body], device=self.device, dtype=dtype)
            for body in chain
        ]
        self.body_rotations = [
            torch.as_tensor(_quat_to_matrix(model.body_quat[body]), device=self.device, dtype=dtype)
            for body in chain
        ]
        self.body_joint_indices: list[int] = []
        self.joint_positions: list[torch.Tensor | None] = []
        self.joint_axes: list[torch.Tensor | None] = []
        for body in chain:
            joint_id = controlled_bodies.get(body)
            if joint_id is None:
                self.body_joint_indices.append(-1)
                self.joint_positions.append(None)
                self.joint_axes.append(None)
                continue
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError("Torch FK currently supports hinge controlled joints only")
            self.body_joint_indices.append(int(joint_id))
            self.joint_positions.append(
                torch.as_tensor(model.jnt_pos[joint_id], device=self.device, dtype=dtype)
            )
            self.joint_axes.append(
                torch.as_tensor(model.jnt_axis[joint_id], device=self.device, dtype=dtype)
            )
        self.site_position = torch.as_tensor(
            model.site_pos[site_id], device=self.device, dtype=dtype
        )
        self.site_rotation = torch.as_tensor(
            _quat_to_matrix(model.site_quat[site_id]), device=self.device, dtype=dtype
        )

    @staticmethod
    def _axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
        x, y, z = axis.unbind()
        zero = torch.zeros_like(x)
        skew = torch.stack(
            [
                torch.stack([zero, -z, y]),
                torch.stack([z, zero, -x]),
                torch.stack([-y, x, zero]),
            ]
        )
        outer = axis[:, None] * axis[None, :]
        identity = torch.eye(3, device=angle.device, dtype=angle.dtype)
        cosine = torch.cos(angle)[..., None, None]
        sine = torch.sin(angle)[..., None, None]
        return cosine * identity + (1.0 - cosine) * outer + sine * skew

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim < 2 or q.shape[-1] != self.n_joints:
            raise ValueError(f"q must have shape [..., {self.n_joints}], got {tuple(q.shape)}")
        values = q.to(device=self.device, dtype=self.dtype)
        batch_shape = values.shape[:-1]
        rotation = torch.eye(3, device=values.device, dtype=values.dtype).expand(
            *batch_shape, 3, 3
        ).clone()
        position = torch.zeros(*batch_shape, 3, device=values.device, dtype=values.dtype)
        for body_position, body_rotation, joint_index, joint_position, joint_axis in zip(
            self.body_positions,
            self.body_rotations,
            self.body_joint_indices,
            self.joint_positions,
            self.joint_axes,
            strict=True,
        ):
            position = position + torch.matmul(rotation, body_position[..., None]).squeeze(-1)
            rotation = torch.matmul(rotation, body_rotation)
            if joint_index >= 0:
                assert joint_position is not None and joint_axis is not None
                joint_rotation = self._axis_angle(joint_axis, values[..., joint_index])
                rotated_anchor = torch.matmul(rotation, joint_position[..., None]).squeeze(-1)
                next_rotation = torch.matmul(rotation, joint_rotation)
                position = (
                    position
                    + rotated_anchor
                    - torch.matmul(next_rotation, joint_position[..., None]).squeeze(-1)
                )
                rotation = next_rotation
        position = position + torch.matmul(rotation, self.site_position[..., None]).squeeze(-1)
        rotation = torch.matmul(rotation, self.site_rotation)
        return position, rotation


class TorchTaskSpaceCost:
    """GPU-compatible task-space scorer used by the full CEM population."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        ee_site_name: str,
        n_joints: int,
        config: TaskSpaceCostConfig,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        config.validate()
        self.config = config
        self.kinematics = TorchSerialKinematics(
            model,
            ee_site_name=ee_site_name,
            n_joints=n_joints,
            device=device,
            dtype=dtype,
        )

    def evaluate(
        self,
        predicted_q: torch.Tensor,
        desired_positions: np.ndarray | torch.Tensor,
        desired_rotations: np.ndarray | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if predicted_q.ndim != 3:
            raise ValueError("predicted_q must have shape [batch, horizon, n_joints]")
        positions = torch.as_tensor(
            desired_positions, device=predicted_q.device, dtype=predicted_q.dtype
        )
        rotations = torch.as_tensor(
            desired_rotations, device=predicted_q.device, dtype=predicted_q.dtype
        )
        horizon = predicted_q.shape[1]
        if positions.shape != (horizon, 3):
            raise ValueError(f"desired_positions must have shape ({horizon}, 3)")
        if rotations.shape != (horizon, 3, 3):
            raise ValueError(f"desired_rotations must have shape ({horizon}, 3, 3)")
        actual_positions, actual_rotations = self.kinematics.forward(predicted_q)
        position_error = (
            actual_positions - positions.unsqueeze(0)
        ) / float(self.config.position_scale_m)
        relative = torch.matmul(rotations.transpose(-1, -2).unsqueeze(0), actual_rotations)
        cosine = ((torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(
            -1.0, 1.0
        )
        sine_vector = 0.5 * torch.stack(
            [
                relative[..., 2, 1] - relative[..., 1, 2],
                relative[..., 0, 2] - relative[..., 2, 0],
                relative[..., 1, 0] - relative[..., 0, 1],
            ],
            dim=-1,
        )
        sine = torch.linalg.vector_norm(sine_vector, dim=-1)
        angle = torch.atan2(sine, cosine) / float(self.config.orientation_scale_rad)
        weights = torch.pow(
            torch.as_tensor(
                self.config.temporal_discount,
                device=predicted_q.device,
                dtype=predicted_q.dtype,
            ),
            torch.arange(horizon, device=predicted_q.device, dtype=predicted_q.dtype),
        )
        weights = weights / weights.sum()
        return {
            "task_position": torch.sum(torch.sum(position_error.square(), dim=-1) * weights, dim=1),
            "task_orientation": torch.sum(angle.square() * weights, dim=1),
        }
