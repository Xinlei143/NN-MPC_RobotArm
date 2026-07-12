from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mpc.kinematics_utils import MujocoKinematics, orientation_error


@dataclass(frozen=True)
class IKConfig:
    max_iterations: int = 100
    position_tolerance: float = 1e-4
    orientation_tolerance: float = 1e-3
    damping: float = 1e-2
    step_gain: float = 0.5
    max_joint_step: float = 0.05
    orientation_weight: float = 0.3
    joint_limit_margin: float = 0.05
    sigma_warning: float = 0.02
    max_backtracking_steps: int = 8

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.position_tolerance <= 0.0 or self.orientation_tolerance <= 0.0:
            raise ValueError("IK tolerances must be positive")
        if self.damping <= 0.0:
            raise ValueError("damping must be positive")
        if not 0.0 < self.step_gain <= 1.0:
            raise ValueError("step_gain must be in (0, 1]")
        if self.max_joint_step <= 0.0:
            raise ValueError("max_joint_step must be positive")
        if self.orientation_weight <= 0.0:
            raise ValueError("orientation_weight must be positive")
        if self.joint_limit_margin < 0.0:
            raise ValueError("joint_limit_margin must be non-negative")
        if self.sigma_warning < 0.0:
            raise ValueError("sigma_warning must be non-negative")
        if self.max_backtracking_steps < 0:
            raise ValueError("max_backtracking_steps must be non-negative")


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float
    sigma_min: float
    joint_limit_margin: float
    message: str = ""


@dataclass(frozen=True)
class TrajectoryIKResult:
    q_des: np.ndarray
    success: bool
    failure_index: int | None
    position_errors: np.ndarray
    orientation_errors: np.ndarray
    iteration_counts: np.ndarray
    sigma_min: np.ndarray
    joint_limit_margins: np.ndarray
    failure_message: str = ""


class MujocoDLSIKSolver:
    """Continuous, bounded damped-least-squares pose IK over a MuJoCo TCP site."""

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_site_name: str = "ee_site",
        config: IKConfig | None = None,
        n_joints: int = 6,
    ) -> None:
        self.config = IKConfig() if config is None else config
        self.kinematics = MujocoKinematics(model=model, ee_site_name=ee_site_name, n_joints=n_joints)
        # This is private to offline IK and never aliases the live environment's data.
        self.ik_data = self.kinematics.data
        self.n_joints = int(n_joints)
        self.joint_low = self.kinematics.joint_low.copy()
        self.joint_high = self.kinematics.joint_high.copy()
        self._safe_low = self.joint_low + float(self.config.joint_limit_margin)
        self._safe_high = self.joint_high - float(self.config.joint_limit_margin)
        if np.any(self._safe_low >= self._safe_high):
            raise ValueError("joint_limit_margin leaves no valid IK joint range")

    def _validate_target(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        q_seed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position = np.asarray(target_position, dtype=np.float64)
        rotation = np.asarray(target_rotation, dtype=np.float64)
        seed = np.asarray(q_seed, dtype=np.float64)
        if position.shape != (3,):
            raise ValueError(f"target_position must have shape (3,), got {position.shape}")
        if rotation.shape != (3, 3):
            raise ValueError(f"target_rotation must have shape (3, 3), got {rotation.shape}")
        if seed.shape != (self.n_joints,):
            raise ValueError(f"q_seed must have shape ({self.n_joints},), got {seed.shape}")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(seed)):
            raise ValueError("IK targets and q_seed must contain only finite values")
        return position, rotation, seed

    def _evaluate(
        self,
        q: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        position, rotation, jacobian = self.kinematics.pose_jacobian(q)
        position_vector = target_position - position
        orientation_vector = orientation_error(target_rotation, rotation)
        position_norm = float(np.linalg.norm(position_vector))
        orientation_norm = float(np.linalg.norm(orientation_vector))
        weighted_error_norm = float(
            np.linalg.norm(np.concatenate([position_vector, self.config.orientation_weight * orientation_vector]))
        )
        return (
            position_vector,
            orientation_vector,
            jacobian,
            position_norm,
            orientation_norm,
            weighted_error_norm,
        )

    def _result(
        self,
        q: np.ndarray,
        success: bool,
        iterations: int,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        message: str = "",
    ) -> IKResult:
        position, rotation, jacobian = self.kinematics.pose_jacobian(q)
        position_error = float(np.linalg.norm(target_position - position))
        orientation_error_norm = float(np.linalg.norm(orientation_error(target_rotation, rotation)))
        return IKResult(
            q=np.asarray(q, dtype=np.float64).copy(),
            success=bool(success),
            iterations=int(iterations),
            position_error=position_error,
            orientation_error=orientation_error_norm,
            sigma_min=float(np.linalg.svd(jacobian, compute_uv=False)[-1]),
            joint_limit_margin=self.kinematics.joint_limit_margin(q),
            message=message,
        )

    def solve_pose(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        q_seed: np.ndarray,
    ) -> IKResult:
        """Solve one pose target, retaining no state beyond the supplied seed."""

        target_position, target_rotation, q = self._validate_target(target_position, target_rotation, q_seed)
        if np.any(q < self._safe_low) or np.any(q > self._safe_high):
            return self._result(
                q,
                success=False,
                iterations=0,
                target_position=target_position,
                target_rotation=target_rotation,
                message="q_seed violates the configured joint-limit margin",
            )

        for iteration in range(self.config.max_iterations + 1):
            (
                position_vector,
                orientation_vector,
                jacobian,
                position_norm,
                orientation_norm,
                weighted_error_norm,
            ) = self._evaluate(q, target_position, target_rotation)
            if (
                position_norm <= self.config.position_tolerance
                and orientation_norm <= self.config.orientation_tolerance
            ):
                return self._result(q, True, iteration, target_position, target_rotation)
            if iteration == self.config.max_iterations:
                break

            weighted_jacobian = jacobian.copy()
            weighted_jacobian[3:] *= float(self.config.orientation_weight)
            weighted_error = np.concatenate(
                [position_vector, float(self.config.orientation_weight) * orientation_vector]
            )
            normal_matrix = weighted_jacobian @ weighted_jacobian.T
            normal_matrix += float(self.config.damping) ** 2 * np.eye(weighted_jacobian.shape[0])
            try:
                delta_q = weighted_jacobian.T @ np.linalg.solve(normal_matrix, weighted_error)
            except np.linalg.LinAlgError:
                return self._result(
                    q,
                    False,
                    iteration,
                    target_position,
                    target_rotation,
                    "DLS normal system was singular",
                )

            delta_q *= float(self.config.step_gain)
            max_delta = float(np.max(np.abs(delta_q)))
            if max_delta > self.config.max_joint_step:
                delta_q *= float(self.config.max_joint_step) / max_delta
            if float(np.max(np.abs(delta_q))) < 1e-12:
                return self._result(
                    q,
                    False,
                    iteration,
                    target_position,
                    target_rotation,
                    "DLS update became numerically zero before convergence",
                )

            accepted = False
            for backtracking_step in range(self.config.max_backtracking_steps + 1):
                scale = 0.5**backtracking_step
                candidate = q + scale * delta_q
                if np.any(candidate < self._safe_low) or np.any(candidate > self._safe_high):
                    continue
                candidate_eval = self._evaluate(candidate, target_position, target_rotation)
                candidate_error_norm = candidate_eval[5]
                if candidate_error_norm <= weighted_error_norm + 1e-12:
                    q = candidate
                    accepted = True
                    break
            if not accepted:
                return self._result(
                    q,
                    False,
                    iteration,
                    target_position,
                    target_rotation,
                    "DLS backtracking could not reduce pose error within joint limits",
                )

        return self._result(
            q,
            False,
            self.config.max_iterations,
            target_position,
            target_rotation,
            "DLS exceeded max_iterations before convergence",
        )

    def solve_trajectory(
        self,
        target_positions: np.ndarray,
        target_rotations: np.ndarray,
        initial_q: np.ndarray,
    ) -> TrajectoryIKResult:
        """Solve targets in order, using each successful solution as the next seed."""

        positions = np.asarray(target_positions, dtype=np.float64)
        rotations = np.asarray(target_rotations, dtype=np.float64)
        initial = np.asarray(initial_q, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError(f"target_positions must have shape [N, 3], got {positions.shape}")
        if rotations.shape != (positions.shape[0], 3, 3):
            raise ValueError(
                "target_rotations must have shape [N, 3, 3] matching target_positions, "
                f"got {rotations.shape}"
            )
        if initial.shape != (self.n_joints,):
            raise ValueError(f"initial_q must have shape ({self.n_joints},), got {initial.shape}")

        count = positions.shape[0]
        q_des = np.full((count, self.n_joints), np.nan, dtype=np.float64)
        position_errors = np.full(count, np.nan, dtype=np.float64)
        orientation_errors = np.full(count, np.nan, dtype=np.float64)
        iteration_counts = np.full(count, -1, dtype=np.int64)
        sigma_min = np.full(count, np.nan, dtype=np.float64)
        joint_limit_margins = np.full(count, np.nan, dtype=np.float64)

        q_seed = initial.copy()
        for index in range(count):
            result = self.solve_pose(positions[index], rotations[index], q_seed)
            q_des[index] = result.q
            position_errors[index] = result.position_error
            orientation_errors[index] = result.orientation_error
            iteration_counts[index] = result.iterations
            sigma_min[index] = result.sigma_min
            joint_limit_margins[index] = result.joint_limit_margin
            if not result.success:
                return TrajectoryIKResult(
                    q_des=q_des,
                    success=False,
                    failure_index=index,
                    position_errors=position_errors,
                    orientation_errors=orientation_errors,
                    iteration_counts=iteration_counts,
                    sigma_min=sigma_min,
                    joint_limit_margins=joint_limit_margins,
                    failure_message=result.message,
                )
            q_seed = result.q

        return TrajectoryIKResult(
            q_des=q_des,
            success=True,
            failure_index=None,
            position_errors=position_errors,
            orientation_errors=orientation_errors,
            iteration_counts=iteration_counts,
            sigma_min=sigma_min,
            joint_limit_margins=joint_limit_margins,
        )
