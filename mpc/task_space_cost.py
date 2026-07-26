from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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


class TorchSerialKinematics(torch.nn.Module):
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
        super().__init__()
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
        self.chain_length = len(chain)
        self._joint_mask_static: tuple[bool, ...] = tuple(body in controlled_bodies for body in chain)
        body_positions = []
        body_rotations = []
        joint_positions = []
        joint_axes = []
        joint_indices = []
        for body in chain:
            body_positions.append(model.body_pos[body])
            body_rotations.append(_quat_to_matrix(model.body_quat[body]))
            joint_id = controlled_bodies.get(body)
            if joint_id is None:
                joint_indices.append(0)
                joint_positions.append(np.zeros(3, dtype=np.float64))
                joint_axes.append(np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
                continue
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError("Torch FK currently supports hinge controlled joints only")
            joint_indices.append(int(joint_id))
            joint_positions.append(model.jnt_pos[joint_id])
            joint_axes.append(model.jnt_axis[joint_id])
        tensor = lambda value: torch.as_tensor(value, device=device, dtype=dtype)
        self.register_buffer("body_positions", tensor(np.asarray(body_positions)))
        self.register_buffer("body_rotations", tensor(np.asarray(body_rotations)))
        self.register_buffer("joint_positions", tensor(np.asarray(joint_positions)))
        self.register_buffer("joint_axes", tensor(np.asarray(joint_axes)))
        self.register_buffer("joint_indices", torch.as_tensor(joint_indices, device=device, dtype=torch.long))
        self._joint_indices_static: tuple[int, ...] = tuple(joint_indices)
        self.register_buffer("joint_mask", torch.as_tensor(self._joint_mask_static, device=device, dtype=torch.bool))
        self.register_buffer("site_position", tensor(model.site_pos[site_id]))
        self.register_buffer("site_rotation", tensor(_quat_to_matrix(model.site_quat[site_id])))
        self.register_buffer("identity", torch.eye(3, device=device, dtype=dtype))

    @staticmethod
    def _axis_angle(axis: torch.Tensor, angle: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
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
        cosine = torch.cos(angle)[..., None, None]
        sine = torch.sin(angle)[..., None, None]
        return cosine * identity + (1.0 - cosine) * outer + sine * skew

    def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim < 2 or q.shape[-1] != self.n_joints:
            raise ValueError(f"q must have shape [..., {self.n_joints}], got {tuple(q.shape)}")
        values = q.to(device=self.body_positions.device, dtype=self.body_positions.dtype)
        batch_shape = values.shape[:-1]
        rotation = self.identity.expand(
            *batch_shape, 3, 3
        ).clone()
        position = torch.zeros(*batch_shape, 3, device=values.device, dtype=values.dtype)
        for body_index in range(self.chain_length):
            body_position = self.body_positions[body_index]
            body_rotation = self.body_rotations[body_index]
            position = position + torch.matmul(rotation, body_position[..., None]).squeeze(-1)
            rotation = torch.matmul(rotation, body_rotation)
            if self._joint_mask_static[body_index]:
                joint_position = self.joint_positions[body_index]
                joint_axis = self.joint_axes[body_index]
                joint_index = self._joint_indices_static[body_index]
                joint_rotation = self._axis_angle(joint_axis, values[..., joint_index], self.identity)
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


class TorchTaskSpaceCost(torch.nn.Module):
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
        step_indices: Sequence[int] | None = None,
        rollout_horizon: int | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.kinematics = TorchSerialKinematics(
            model,
            ee_site_name=ee_site_name,
            n_joints=n_joints,
            device=device,
            dtype=dtype,
        )
        if step_indices is None:
            indices: tuple[int, ...] = ()
            weights = torch.empty(0, device=device, dtype=dtype)
            self.sparse_weighting = "full"
        else:
            if rollout_horizon is None or rollout_horizon <= 0:
                raise ValueError("rollout_horizon is required for sparse Torch task cost")
            indices = tuple(int(index) for index in step_indices)
            if not indices or tuple(sorted(set(indices))) != indices:
                raise ValueError("step_indices must be non-empty, unique, and strictly increasing")
            if indices[0] < 0 or indices[-1] >= rollout_horizon:
                raise ValueError("step_indices must lie in [0, rollout_horizon)")
            full_weights = torch.pow(
                torch.as_tensor(config.temporal_discount, device=device, dtype=dtype),
                torch.arange(rollout_horizon, device=device, dtype=dtype),
            )
            full_weights = full_weights / full_weights.sum()
            steps = torch.arange(rollout_horizon, device=device)
            sparse = torch.as_tensor(indices, device=device)
            nearest = torch.argmin(torch.abs(steps[:, None] - sparse[None, :]), dim=1)
            weights = torch.zeros(len(indices), device=device, dtype=dtype)
            weights.scatter_add_(0, nearest, full_weights)
            self.sparse_weighting = "nearest_interval"
        self.register_buffer("step_indices", torch.as_tensor(indices, device=device, dtype=torch.long))
        self.register_buffer("temporal_weights", weights)
        self._compiled_terms = None
        self._profile_enabled = False
        self._last_profile_ms = float("nan")

    @property
    def is_sparse(self) -> bool:
        return self.step_indices.numel() > 0

    @property
    def compile_enabled(self) -> bool:
        return self._compiled_terms is not None

    def enable_compile(self) -> None:
        if self._compiled_terms is None:
            self._compiled_terms = torch.compile(
                self._evaluate_terms,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,
            )

    @property
    def last_profile_ms(self) -> float:
        return self._last_profile_ms

    def set_profile_enabled(self, enabled: bool) -> None:
        self._profile_enabled = bool(enabled)

    def warm_up(self, batch_size: int, *, repetitions: int = 3) -> None:
        if not self.is_sparse:
            raise ValueError("warm_up is only supported for fixed-shape sparse scorers")
        if batch_size <= 0 or repetitions <= 0:
            raise ValueError("batch_size and repetitions must be positive")
        device = self.site_device
        dtype = self.kinematics.body_positions.dtype
        q = torch.zeros((batch_size, self.step_indices.numel(), self.kinematics.n_joints), device=device, dtype=dtype)
        positions = torch.zeros((self.step_indices.numel(), 3), device=device, dtype=dtype)
        rotations = self.kinematics.identity.expand(self.step_indices.numel(), -1, -1).clone()
        for _ in range(repetitions):
            self(q, positions, rotations)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @property
    def site_device(self) -> torch.device:
        return self.kinematics.site_position.device

    def _weights(self, horizon: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.is_sparse:
            if horizon != self.step_indices.numel():
                raise ValueError(
                    f"sparse scorer expects {self.step_indices.numel()} steps, got {horizon}"
                )
            return self.temporal_weights.to(device=device, dtype=dtype)
        weights = torch.pow(
            torch.as_tensor(self.config.temporal_discount, device=device, dtype=dtype),
            torch.arange(horizon, device=device, dtype=dtype),
        )
        return weights / weights.sum()

    def _evaluate_terms(
        self,
        predicted_q: torch.Tensor,
        desired_positions: torch.Tensor,
        desired_rotations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, horizon, joints = predicted_q.shape
        if joints != self.kinematics.n_joints:
            raise ValueError(f"predicted_q has {joints} joints, expected {self.kinematics.n_joints}")
        flat_q = predicted_q.reshape(batch * horizon, joints)
        flat_position, flat_rotation = self.kinematics(flat_q)
        actual_positions = flat_position.reshape(batch, horizon, 3)
        actual_rotations = flat_rotation.reshape(batch, horizon, 3, 3)
        position_error = (actual_positions - desired_positions.unsqueeze(0)) / float(self.config.position_scale_m)
        relative = torch.matmul(desired_rotations.transpose(-1, -2).unsqueeze(0), actual_rotations)
        cosine = ((torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
        sine_vector = 0.5 * torch.stack(
            [
                relative[..., 2, 1] - relative[..., 1, 2],
                relative[..., 0, 2] - relative[..., 2, 0],
                relative[..., 1, 0] - relative[..., 0, 1],
            ],
            dim=-1,
        )
        angle = torch.atan2(torch.linalg.vector_norm(sine_vector, dim=-1), cosine)
        weights = self._weights(horizon, device=predicted_q.device, dtype=predicted_q.dtype)
        position_cost = torch.sum(torch.sum(position_error.square(), dim=-1) * weights, dim=1)
        orientation_cost = torch.sum(
            (angle / float(self.config.orientation_scale_rad)).square() * weights,
            dim=1,
        )
        return position_cost, orientation_cost

    def forward(
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
        terms = self._evaluate_terms if self._compiled_terms is None else self._compiled_terms
        if self._profile_enabled and predicted_q.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            position_cost, orientation_cost = terms(predicted_q, positions, rotations)
            end.record()
            torch.cuda.synchronize(predicted_q.device)
            self._last_profile_ms = float(start.elapsed_time(end))
        else:
            position_cost, orientation_cost = terms(predicted_q, positions, rotations)
            self._last_profile_ms = float("nan")
        return {
            "task_position": position_cost,
            "task_orientation": orientation_cost,
        }

    evaluate = forward
