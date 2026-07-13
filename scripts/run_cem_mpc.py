from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from neural_dynamics.mujoco_env import MuJoCoArmEnv
from neural_dynamics.paths import DEFAULT_MODEL_XML
from neural_dynamics.rollout import load_dynamics_bundle
from neural_dynamics.train_utils import set_seed
from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.cost_functions import JointSpaceCostConfig
from mpc.kinematics_utils import site_pose
from mpc.logging import save_mpc_run
from mpc.planner_rollout import LearnedDynamicsPlanner, PlannerRolloutConfig
from mpc.reference import finite_difference_dq, generate_joint_reference
from mpc.reference_pipeline import ReferenceBundle, load_reference_bundle
from mpc.utils import build_history_tensor


MPC_HOME_Q = np.zeros(6, dtype=np.float32)


def resolve_runtime_path(path: str) -> Path:
    expanded = Path(path).expanduser()
    if expanded.is_absolute():
        return expanded
    root_path = ROOT / expanded
    if root_path.exists() or (expanded.parts and expanded.parts[0] == "dynamics_modeling"):
        return root_path
    dynamics_path = DYNAMICS_ROOT / expanded
    if dynamics_path.exists():
        return dynamics_path
    if expanded.parts and expanded.parts[0] == "outputs":
        return root_path
    return root_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run learned CEM-MPC in closed-loop MuJoCo simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_xml", default=DEFAULT_MODEL_XML, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--normalizer", required=True, type=str)
    parser.add_argument("--model_type", choices=["mlp", "gru", "transformer"], default="transformer")
    parser.add_argument("--history_len", default=None, type=int)
    parser.add_argument("--n_joints", default=6, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--episode_len", default=200, type=int)
    parser.add_argument("--settle_steps", default=50, type=int)
    parser.add_argument(
        "--reference_mode",
        choices=["hold", "step", "joint_sine", "multi_joint_sine", "task"],
        default="multi_joint_sine",
    )
    parser.add_argument(
        "--reference_file",
        default=None,
        type=str,
        help="Validated task-space ReferenceBundle .npz file. Required when --reference_mode task.",
    )
    parser.add_argument("--ee_site_name", default="ee_site", type=str)
    parser.add_argument("--reference_amplitude", default=0.15, type=float)
    parser.add_argument("--save_dir", default="outputs/mpc/cem_run", type=str)
    parser.add_argument("--fail_on_limit_violation", action="store_true")

    parser.add_argument("--horizon", default=20, type=int)
    parser.add_argument("--num_samples", default=1024, type=int)
    parser.add_argument("--num_elites", default=None, type=int)
    parser.add_argument("--elite_ratio", default=0.08, type=float)
    parser.add_argument("--cem_iters", default=4, type=int)
    parser.add_argument("--init_std", default=0.12, type=float)
    parser.add_argument("--min_std", default=0.01, type=float)
    parser.add_argument("--smoothing_alpha", default=0.2, type=float)
    parser.add_argument("--temporal_noise_alpha", default=0.8, type=float)
    parser.add_argument("--rollout_batch_size", default=256, type=int)

    parser.add_argument("--ref_mode", choices=["delta", "absolute"], default="delta")
    parser.add_argument("--delta_base", choices=["previous_q_ref", "current_q"], default="previous_q_ref")
    parser.add_argument("--delta_q_ref_max", default=0.08, type=float)
    parser.add_argument("--q_ref_rate_limit", default=0.08, type=float)
    parser.add_argument("--delta_rate_limit", default=None, type=float)
    parser.add_argument("--joint_limit_margin", default=0.02, type=float)

    parser.add_argument("--w_q", default=1.0, type=float)
    parser.add_argument("--w_dq", default=0.05, type=float)
    parser.add_argument("--w_u_offset", default=0.05, type=float)
    parser.add_argument("--w_dqref", default=0.05, type=float)
    parser.add_argument("--w_ddqref", default=0.02, type=float)
    parser.add_argument("--w_terminal", default=0.5, type=float)
    parser.add_argument("--w_joint_limit", default=2.0, type=float)
    parser.add_argument("--q_amp_fraction", default=0.2, type=float)
    parser.add_argument("--q_tol", default=0.04, type=float)
    parser.add_argument("--dq_scale", default=0.5, type=float)
    parser.add_argument("--u_offset_scale", default=0.2, type=float)
    parser.add_argument("--dqref_scale", default=0.08, type=float)
    parser.add_argument("--ddqref_scale", default=0.05, type=float)
    parser.add_argument("--joint_limit_safe_margin", default=0.08, type=float)
    parser.add_argument("--joint_limit_temp", default=0.02, type=float)
    parser.add_argument("--velocity_cost_mode", choices=["track", "damping"], default="track")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _stack_records(records: list[np.ndarray], dtype: np.dtype = np.float32) -> np.ndarray:
    if not records:
        return np.empty((0,), dtype=dtype)
    return np.asarray(records, dtype=dtype)


def _validate_task_reference(bundle: ReferenceBundle, n_joints: int, horizon: int) -> None:
    """Validate the subset of ReferenceBundle invariants required by receding-horizon MPC."""
    q_des = np.asarray(bundle.q_des)
    dq_des = np.asarray(bundle.dq_des)
    if q_des.ndim != 2 or q_des.shape[1] != n_joints:
        raise ValueError(f"Task reference q_des must have shape [N, {n_joints}], got {q_des.shape}")
    if dq_des.shape != q_des.shape:
        raise ValueError(f"Task reference dq_des must match q_des shape {q_des.shape}, got {dq_des.shape}")
    if bundle.execution_steps <= 0:
        raise ValueError(f"Task reference execution_steps must be positive, got {bundle.execution_steps}")
    minimum_length = int(bundle.execution_steps) + int(horizon) + 1
    if q_des.shape[0] < minimum_length:
        raise ValueError(
            "Task reference is too short for the requested MPC horizon: "
            f"need at least execution_steps + horizon + 1 = {minimum_length} points, got {q_des.shape[0]}"
        )

    expected_task_shapes = {
        "task_positions_des": (q_des.shape[0], 3),
        "task_rotations_des": (q_des.shape[0], 3, 3),
        "segment_ids": (q_des.shape[0],),
        "lap_ids": (q_des.shape[0],),
    }
    for name, expected_shape in expected_task_shapes.items():
        value = getattr(bundle, name, None)
        if value is None or np.asarray(value).shape != expected_shape:
            actual_shape = None if value is None else np.asarray(value).shape
            raise ValueError(f"Task reference {name} must have shape {expected_shape}, got {actual_shape}")


def _load_task_reference(args: argparse.Namespace) -> ReferenceBundle:
    if not args.reference_file:
        raise ValueError("--reference_file is required when --reference_mode task")
    bundle = load_reference_bundle(
        resolve_runtime_path(args.reference_file),
        expected_n_joints=args.n_joints,
        min_horizon=args.horizon,
    )
    _validate_task_reference(bundle, args.n_joints, args.horizon)
    return bundle


def _reference_for_run(
    args: argparse.Namespace,
    state: np.ndarray,
    env: MuJoCoArmEnv,
    control_dt: float,
) -> tuple[np.ndarray, np.ndarray, int, ReferenceBundle | None]:
    """Return the reference arrays and the number of closed-loop control steps."""
    if args.reference_mode == "task":
        bundle = _load_task_reference(args)
        expected_initial_q = MPC_HOME_Q[: args.n_joints]
        if not np.allclose(bundle.q_des[0], expected_initial_q, atol=1e-6, rtol=0.0):
            raise ValueError(
                "Task reference must start at the fixed MPC home pose "
                f"{expected_initial_q.tolist()}, got {np.asarray(bundle.q_des[0]).tolist()}"
            )
        return (
            np.asarray(bundle.q_des, dtype=np.float32),
            np.asarray(bundle.dq_des, dtype=np.float32),
            int(bundle.execution_steps),
            bundle,
        )

    reference = generate_joint_reference(
        args.reference_mode,
        state[: args.n_joints],
        env.joint_low + args.joint_limit_margin,
        env.joint_high - args.joint_limit_margin,
        args.episode_len + args.horizon + 1,
        control_dt,
        seed=args.seed,
        amplitude=args.reference_amplitude,
    )
    return reference, finite_difference_dq(reference, control_dt), args.episode_len, None


def _orientation_error(desired_rotation: np.ndarray, actual_rotation: np.ndarray) -> float:
    """Return the geodesic orientation error in radians."""
    relative_rotation = np.asarray(desired_rotation, dtype=np.float64) @ np.asarray(actual_rotation, dtype=np.float64).T
    cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def run_closed_loop_mpc(args: argparse.Namespace) -> dict[str, Any]:
    if args.reference_mode != "task" and args.episode_len <= 0:
        raise ValueError(f"episode_len must be positive, got {args.episode_len}")
    if args.horizon <= 0:
        raise ValueError(f"horizon must be positive, got {args.horizon}")
    set_seed(args.seed)
    device = resolve_device(args.device)
    bundle = load_dynamics_bundle(
        checkpoint_path=resolve_runtime_path(args.checkpoint),
        normalizer_path=resolve_runtime_path(args.normalizer),
        model_type=args.model_type,
        n_joints=args.n_joints,
        device=device,
        history_len=args.history_len,
    )
    env = MuJoCoArmEnv(str(resolve_runtime_path(args.model_xml)), n_joints=args.n_joints, seed=args.seed)

    states_history: list[np.ndarray] = []
    q_ref_history: list[np.ndarray] = []
    actual_states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    selected_q_refs: list[np.ndarray] = []
    selected_delta_q_refs: list[np.ndarray] = []
    q_des_records: list[np.ndarray] = []
    dq_des_records: list[np.ndarray] = []
    planning_times: list[float] = []
    best_costs: list[float] = []
    elite_costs: list[float] = []
    failures: list[int] = []
    failure_reasons: list[str] = []
    joint_limit_violations: list[int] = []
    realized_tracking_errors: list[float] = []
    predicted_real_error_gaps: list[float] = []
    torque_records: dict[str, list[np.ndarray]] = {"tau_actuator": [], "tau_gravity": [], "tau_total": []}
    desired_ee_positions: list[np.ndarray] = []
    desired_ee_rotations: list[np.ndarray] = []
    actual_ee_positions: list[np.ndarray] = []
    actual_ee_rotations: list[np.ndarray] = []
    ee_position_errors: list[float] = []
    ee_orientation_errors: list[float] = []
    segment_ids: list[int] = []
    lap_ids: list[int] = []
    rows: list[dict[str, Any]] = []
    task_reference: ReferenceBundle | None = None
    execution_steps = args.episode_len

    try:
        if args.n_joints > MPC_HOME_Q.shape[0]:
            raise ValueError(f"MPC home pose supports at most {MPC_HOME_Q.shape[0]} joints, got {args.n_joints}")
        state = env.reset_to_configuration(MPC_HOME_Q[: args.n_joints])
        previous_q_ref = np.asarray(state[: args.n_joints], dtype=np.float32).copy()
        for _ in range(args.settle_steps):
            state = env.step(previous_q_ref)
        states_history.append(state.copy())
        q_ref_history.append(previous_q_ref.copy())

        reference, dq_reference, execution_steps, task_reference = _reference_for_run(
            args=args,
            state=state,
            env=env,
            control_dt=bundle.control_dt,
        )
        cost_config = JointSpaceCostConfig(
            w_q=args.w_q,
            w_dq=args.w_dq,
            w_u_offset=args.w_u_offset,
            w_dqref=args.w_dqref,
            w_ddqref=args.w_ddqref,
            w_terminal=args.w_terminal,
            w_joint_limit=args.w_joint_limit,
            q_amp_fraction=args.q_amp_fraction,
            q_tol=args.q_tol,
            dq_scale=args.dq_scale,
            u_offset_scale=args.u_offset_scale,
            dqref_scale=args.dqref_scale,
            ddqref_scale=args.ddqref_scale,
            joint_limit_safe_margin=args.joint_limit_safe_margin,
            joint_limit_temp=args.joint_limit_temp,
            velocity_cost_mode=args.velocity_cost_mode,
        )
        rollout_config = PlannerRolloutConfig(
            mode=args.ref_mode,
            delta_base=args.delta_base,
            delta_q_ref_max=args.delta_q_ref_max,
            q_ref_rate_limit=args.q_ref_rate_limit,
            delta_rate_limit=args.delta_rate_limit,
            joint_limit_margin=args.joint_limit_margin,
            rollout_batch_size=args.rollout_batch_size,
        )
        controller: CEMMPCController | None = None

        for step_idx in range(execution_steps):
            initial_history = build_history_tensor(states_history, q_ref_history, bundle.history_len, device)
            planner = LearnedDynamicsPlanner(
                model=bundle.model,
                normalizer=bundle.normalizer,
                model_type=bundle.model_type,
                state_dim=bundle.state_dim,
                target_mode=bundle.target_mode,
                control_dt=bundle.control_dt,
                initial_history=initial_history,
                q_des=torch.as_tensor(reference[step_idx + 1 : step_idx + 1 + args.horizon], dtype=torch.float32, device=device),
                dq_des=torch.as_tensor(dq_reference[step_idx + 1 : step_idx + 1 + args.horizon], dtype=torch.float32, device=device),
                previous_q_ref=torch.as_tensor(previous_q_ref, dtype=torch.float32, device=device),
                joint_low=torch.as_tensor(env.joint_low, dtype=torch.float32, device=device),
                joint_high=torch.as_tensor(env.joint_high, dtype=torch.float32, device=device),
                cost_config=cost_config,
                rollout_config=rollout_config,
            )
            if controller is None:
                controller = CEMMPCController(
                    config=CEMMPCConfig(
                        horizon=args.horizon,
                        action_dim=args.n_joints,
                        num_samples=args.num_samples,
                        num_elites=args.num_elites,
                        elite_ratio=args.elite_ratio,
                        cem_iters=args.cem_iters,
                        init_std=args.init_std,
                        min_std=args.min_std,
                        smoothing_alpha=args.smoothing_alpha,
                        temporal_noise_alpha=args.temporal_noise_alpha,
                        seed=args.seed,
                        device=str(device),
                    ),
                    planner=planner,
                    joint_low=env.joint_low,
                    joint_high=env.joint_high,
                )
            else:
                controller.planner = planner

            result = controller.plan(current_state=state, previous_q_ref=previous_q_ref)
            q_ref_command = result.q_ref.astype(np.float32)
            torque = env.compute_torque_components(q_ref_command)
            actual_states.append(state.copy())
            q_des_records.append(reference[step_idx].copy())
            dq_des_records.append(dq_reference[step_idx].copy())
            if task_reference is not None:
                desired_position = np.asarray(task_reference.task_positions_des[step_idx], dtype=np.float32)
                desired_rotation = np.asarray(task_reference.task_rotations_des[step_idx], dtype=np.float32)
                actual_position, actual_rotation = site_pose(env.model, env.data, args.ee_site_name)
                actual_position = np.asarray(actual_position, dtype=np.float32)
                actual_rotation = np.asarray(actual_rotation, dtype=np.float32)
                desired_ee_positions.append(desired_position)
                desired_ee_rotations.append(desired_rotation)
                actual_ee_positions.append(actual_position)
                actual_ee_rotations.append(actual_rotation)
                ee_position_errors.append(float(np.linalg.norm(actual_position - desired_position)))
                ee_orientation_errors.append(_orientation_error(desired_rotation, actual_rotation))
                segment_ids.append(int(task_reference.segment_ids[step_idx]))
                lap_ids.append(int(task_reference.lap_ids[step_idx]))
            selected_q_refs.append(q_ref_command.copy())
            selected_delta_q_refs.append(result.delta_q_ref.copy())
            planning_times.append(result.planning_time)
            best_costs.append(result.best_cost)
            elite_costs.append(result.elite_mean_cost)
            failures.append(int(result.failure))
            failure_reasons.append(result.failure_reason)
            for key, target_key in (("actuator_tau", "tau_actuator"), ("gravity_tau", "tau_gravity"), ("total_tau", "tau_total")):
                torque_records[target_key].append(torque[key].astype(np.float32))

            try:
                state = env.step(q_ref_command)
                joint_limit_violations.append(0)
            except RuntimeError:
                joint_limit_violations.append(1)
                if args.fail_on_limit_violation:
                    raise
                break

            next_states.append(state.copy())
            previous_q_ref = q_ref_command.copy()
            states_history.append(state.copy())
            q_ref_history.append(previous_q_ref.copy())
            realized_error = float(np.linalg.norm(state[: args.n_joints] - reference[step_idx + 1]))
            realized_tracking_errors.append(realized_error)
            predicted_real_error_gaps.append(float(result.best_cost - realized_error))
            row: dict[str, Any] = {
                "step": step_idx,
                "tracking_error": realized_error,
                "planning_time": result.planning_time,
                "best_cost": result.best_cost,
                "elite_mean_cost": result.elite_mean_cost,
                "failure": int(result.failure),
                "failure_reason": result.failure_reason,
                "joint_limit_violation": joint_limit_violations[-1],
                "predicted_real_error_gap": predicted_real_error_gaps[-1],
            }
            if task_reference is not None:
                row.update(
                    {
                        "ee_position_error": ee_position_errors[-1],
                        "ee_orientation_error": ee_orientation_errors[-1],
                        "segment_id": segment_ids[-1],
                        "lap_id": lap_ids[-1],
                    }
                )
            rows.append(row)
    finally:
        env.close()

    arrays: dict[str, np.ndarray] = {
        "actual_states": _stack_records(actual_states),
        "next_states": _stack_records(next_states),
        "q_des": _stack_records(q_des_records),
        "dq_des": _stack_records(dq_des_records),
        "actuator_q_ref": _stack_records(selected_q_refs),
        "delta_q_ref": _stack_records(selected_delta_q_refs),
        "planning_time": np.asarray(planning_times, dtype=np.float32),
        "best_cost": np.asarray(best_costs, dtype=np.float32),
        "elite_mean_cost": np.asarray(elite_costs, dtype=np.float32),
        "failure_flags": np.asarray(failures, dtype=np.int64),
        "joint_limit_violation_flags": np.asarray(joint_limit_violations, dtype=np.int64),
        "realized_tracking_error": np.asarray(realized_tracking_errors, dtype=np.float32),
        "predicted_real_error_gap": np.asarray(predicted_real_error_gaps, dtype=np.float32),
        **{key: _stack_records(value) for key, value in torque_records.items()},
    }
    if task_reference is not None:
        arrays.update(
            {
                "desired_ee_positions": _stack_records(desired_ee_positions),
                "desired_ee_rotations": _stack_records(desired_ee_rotations),
                "actual_ee_positions": _stack_records(actual_ee_positions),
                "actual_ee_rotations": _stack_records(actual_ee_rotations),
                "ee_position_errors": np.asarray(ee_position_errors, dtype=np.float32),
                "ee_orientation_errors": np.asarray(ee_orientation_errors, dtype=np.float32),
                "segment_ids": _stack_records(segment_ids, dtype=np.int64),
                "lap_ids": _stack_records(lap_ids, dtype=np.int64),
                "execution_steps": np.asarray(execution_steps, dtype=np.int64),
            }
        )
    return {"arrays": arrays, "rows": rows, "failure_reasons": failure_reasons}


def main() -> None:
    args = parse_args()
    result = run_closed_loop_mpc(args)
    save_dir = resolve_runtime_path(args.save_dir)
    save_mpc_run(save_dir, result["arrays"], result["rows"])
    print(f"Saved CEM-MPC rollout to {save_dir}")


if __name__ == "__main__":
    main()
