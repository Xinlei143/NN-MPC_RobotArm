"""Calibrate real threaded E2E delay for a residual-MPC planner variant.

The filename is retained for compatibility with the original H20 workflow.  The
calibration itself is horizon-agnostic: it measures the configured threaded
planner, including its residual parameterization and optional GPU stage-one
task-space cost, then derives the deployment delay from E2E P95.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from scripts.experiment_utils.hashing import file_identity
from scripts.robustness._runtime import ensure_import_paths, load_runner
from scripts.robustness.evaluate_direct_ik import load_task_cases

ensure_import_paths()


RUNNER = load_runner("planner_projection_delay_calibration_runner")


def decision_horizon(args: argparse.Namespace) -> int:
    """Return the CEM latent horizon, distinct from the dynamics rollout horizon."""
    if args.residual_parameterization == "full":
        return int(args.horizon)
    return int(args.residual_control_points)


def _apply_variant_configuration(run_args: argparse.Namespace, args: argparse.Namespace) -> None:
    """Restore planner settings after a benchmark case supplies its reference.

    Benchmark manifests record the historical H20 settings.  They should supply
    trajectory/reference metadata here, but must not silently overwrite the
    variant whose latency is being calibrated.
    """
    fields = (
        "horizon",
        "num_samples",
        "num_elites",
        "elite_ratio",
        "cem_iters",
        "init_std",
        "min_std",
        "smoothing_alpha",
        "temporal_noise_alpha",
        "reset_std_each_step",
        "uniform_sample_ratio",
        "rollout_batch_size",
        "cem_execute",
        "residual_parameterization",
        "residual_control_points",
        "planner_guard_ms",
        "planner_min_interval_ms",
        "planner_projection",
        "planner_projection_backend",
        "planner_projection_strategy",
        "exact_task_space_cost",
        "stage_one_task_space_cost",
        "w_task_position",
        "w_task_orientation",
        "task_position_scale_m",
        "task_orientation_scale_rad",
        "temporal_discount",
        "mpc_preview_nominal_steps",
        "residual_cost_semantics",
        "packet_residual_semantics",
        "residual_feasibility_semantics",
        "nominal_command_semantics",
        "asap_history_mode",
        "asap_snapshot_mode",
        "feedback_kq",
        "feedback_kdq",
        "feedback_max",
        "mpc_warmup_plans",
    )
    for field in fields:
        setattr(run_args, field, getattr(args, field))


def parse_args() -> argparse.Namespace:
    parser = RUNNER.build_arg_parser()
    parser.add_argument(
        "--manifest",
        default="outputs/robustness/benchmark.json",
        help="Task-reference benchmark manifest; it supplies cases, not the calibrated MPC configuration.",
    )
    parser.add_argument("--case_ids", default="circle_00,figure8_00")
    parser.add_argument("--plans", default=500, type=int)
    parser.add_argument("--calibration_delay", default=10, type=int)
    parser.add_argument("--output_path", required=True)
    parser.set_defaults(
        checkpoint="dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt",
        normalizer="dynamics_modeling/outputs/checkpoints/gru_20260717_182930/normalizer.pt",
        model_type="gru",
        history_len=16,
        horizon=20,
        num_samples=128,
        cem_iters=2,
        rollout_batch_size=128,
        multirate_mode="threaded_asap",
        delay_protocol="full",
        device="cuda",
        max_execution_steps=500,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plans <= 0 or args.calibration_delay <= 0:
        raise ValueError("plans and calibration_delay must be positive")
    if args.horizon <= 0:
        raise ValueError("horizon must be positive")
    if args.residual_parameterization == "linear_control_points":
        if not 2 <= args.residual_control_points <= args.horizon:
            raise ValueError("--residual_control_points must be in [2, horizon]")
        if not args.reset_std_each_step:
            raise ValueError("linear_control_points calibration requires --reset_std_each_step")
    case_ids = [value.strip() for value in args.case_ids.split(",") if value.strip()]
    if not case_ids:
        raise ValueError("--case_ids must contain at least one case")
    manifest_path = RUNNER.resolve_runtime_path(args.manifest)
    _, cases = load_task_cases(manifest_path, case_ids)
    solve: list[float] = []
    e2e: list[float] = []
    episodes = 0
    late_count = 0
    while len(e2e) < args.plans:
        case = cases[episodes % len(cases)]
        run_args = deepcopy(args)
        for key, value in case["run_args"].items():
            if hasattr(run_args, key):
                setattr(run_args, key, value)
        _apply_variant_configuration(run_args, args)
        run_args.checkpoint = args.checkpoint
        run_args.normalizer = args.normalizer
        run_args.model_type = "gru"
        run_args.history_len = 16
        run_args.device = args.device
        run_args.controller_mode = "mpc"
        run_args.mpc_policy = "residual"
        run_args.reference_mode = "task"
        run_args.multirate_mode = "threaded_asap"
        run_args.delay_protocol = "full"
        run_args.anticipation_delay_steps = args.calibration_delay
        run_args.max_execution_steps = 500
        run_args.planner_projection = args.planner_projection
        run_args.planner_projection_backend = args.planner_projection_backend
        run_args.planner_projection_strategy = args.planner_projection_strategy
        run_args.seed = 20260723 + episodes
        run_args.visualize = False
        result = RUNNER.run_closed_loop_mpc(run_args)
        arrays = result["arrays"]
        replanned = np.asarray(arrays["mpc_replanned"], dtype=bool)
        solve_values = np.asarray(arrays["planning_time"], dtype=np.float64)
        e2e_values = np.asarray(arrays["planner_end_to_end_latency_s"], dtype=np.float64)
        valid = replanned & np.isfinite(e2e_values)
        solve.extend(solve_values[valid & np.isfinite(solve_values)].tolist())
        e2e.extend(e2e_values[valid].tolist())
        late = np.asarray(arrays["packet_late_dropped"], dtype=np.int64)
        late_count += int(np.sum(late[valid] != 0))
        episodes += 1
        print(f"episode={episodes} collected={min(len(e2e), args.plans)}/{args.plans}")
    solve_array = np.asarray(solve[: args.plans], dtype=np.float64)
    e2e_array = np.asarray(e2e[: args.plans], dtype=np.float64)
    guard_s = float(args.planner_guard_ms) / 1000.0
    control_dt = 0.01
    delay = int(np.ceil((float(np.percentile(e2e_array, 95)) + guard_s) / control_dt))

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "p50_s": float(np.percentile(values, 50)),
            "p95_s": float(np.percentile(values, 95)),
            "p99_s": float(np.percentile(values, 99)),
            "max_s": float(np.max(values)),
        }

    payload = {
        "schema_version": 2,
        "kind": "threaded_e2e_delay_calibration",
        # Keep horizon for old consumers; rollout_horizon/decision_horizon make
        # Control-point variants unambiguous for new consumers.
        "horizon": args.horizon,
        "rollout_horizon": args.horizon,
        "decision_horizon": decision_horizon(args),
        "residual_parameterization": args.residual_parameterization,
        "residual_control_points": (
            None if args.residual_parameterization == "full" else args.residual_control_points
        ),
        "control_point_interpolation": (
            "identity" if args.residual_parameterization == "full" else "linear_align_corners"
        ),
        "control_point_tail_mode": "hold",
        "cem_reset_std_each_step": args.reset_std_each_step,
        "num_samples": args.num_samples,
        "cem_iters": args.cem_iters,
        "plans": args.plans,
        "episodes": episodes,
        "case_ids": case_ids,
        "planner_projection": args.planner_projection,
        "planner_projection_backend": args.planner_projection_backend,
        "planner_projection_strategy": args.planner_projection_strategy,
        "exact_task_space_cost": args.exact_task_space_cost,
        "stage_one_task_space_cost": args.stage_one_task_space_cost,
        "w_task_position": args.w_task_position,
        "w_task_orientation": args.w_task_orientation,
        "task_position_scale_m": args.task_position_scale_m,
        "task_orientation_scale_rad": args.task_orientation_scale_rad,
        "temporal_discount": args.temporal_discount,
        "mpc_preview_nominal_steps": args.mpc_preview_nominal_steps,
        "calibration_delay_steps": args.calibration_delay,
        "solve_latency": stats(solve_array),
        "end_to_end_latency": stats(e2e_array),
        "guard_s": guard_s,
        "control_dt_s": control_dt,
        "anticipation_delay_steps": delay,
        "calibration_late_count": late_count,
        "manifest": file_identity(manifest_path),
        "checkpoint": file_identity(RUNNER.resolve_runtime_path(args.checkpoint)),
        "normalizer": file_identity(RUNNER.resolve_runtime_path(args.normalizer)),
    }
    output = RUNNER.resolve_runtime_path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"H={args.horizon}, K={decision_horizon(args)}, "
        f"stage1_task={args.stage_one_task_space_cost}; "
        f"E2E p95={payload['end_to_end_latency']['p95_s'] * 1e3:.2f} ms; "
        f"D={delay}; saved {output}"
    )


if __name__ == "__main__":
    main()
