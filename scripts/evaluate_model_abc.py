from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from mpc.logging import save_mpc_run


def load_run_cem_mpc_module():
    spec = importlib.util.spec_from_file_location("local_run_cem_mpc", ROOT / "scripts" / "run_cem_mpc.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load scripts/run_cem_mpc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_CEM_MPC = load_run_cem_mpc_module()


def parse_model_spec(value: str) -> dict[str, str]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) not in {4, 5}:
        raise ValueError(
            "--model_spec must be label,checkpoint,normalizer,model_type[,dataset_path], "
            f"got {value!r}"
        )
    return {
        "label": parts[0],
        "checkpoint": parts[1],
        "normalizer": parts[2],
        "model_type": parts[3],
        "dataset_path": "" if len(parts) == 4 else parts[4],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = RUN_CEM_MPC.build_arg_parser()
    parser.description = "Evaluate Model A/B/C checkpoints with common CEM-MPC settings."
    for action in parser._actions:
        if action.dest in {"checkpoint", "normalizer"}:
            action.required = False
    parser.add_argument(
        "--model_spec",
        action="append",
        required=True,
        help="Repeat as label,checkpoint,normalizer,model_type[,dataset_path].",
    )
    parser.set_defaults(save_dir="outputs/mpc/model_abc")
    return parser.parse_args(argv)


def summarize(label: str, arrays: dict[str, np.ndarray], dataset_path: str) -> dict[str, float | str | int]:
    tracking = arrays["realized_tracking_error"]
    failures = arrays["failure_flags"]
    planning_time = arrays["planning_time"]
    predicted_q_error = arrays.get("predicted_next_q_error", np.empty(0))
    predicted_dq_error = arrays.get("predicted_next_dq_error", np.empty(0))
    replay_q_error = np.asarray(arrays.get("replay_q_error_norm", np.empty((0, 0))), dtype=np.float64)
    replay_dq_error = np.asarray(arrays.get("replay_dq_error_norm", np.empty((0, 0))), dtype=np.float64)
    replay_q_first = replay_q_error[:, 0] if replay_q_error.ndim == 2 and replay_q_error.shape[1] else np.empty(0)
    replay_q_terminal = replay_q_error[:, -1] if replay_q_error.ndim == 2 and replay_q_error.shape[1] else np.empty(0)
    replay_dq_first = replay_dq_error[:, 0] if replay_dq_error.ndim == 2 and replay_dq_error.shape[1] else np.empty(0)
    replay_dq_terminal = replay_dq_error[:, -1] if replay_dq_error.ndim == 2 and replay_dq_error.shape[1] else np.empty(0)
    finite_mean = lambda values: float(np.mean(values[np.isfinite(values)])) if np.any(np.isfinite(values)) else float("nan")
    finite_percentile = lambda values, q: float(np.percentile(values[np.isfinite(values)], q)) if np.any(np.isfinite(values)) else float("nan")
    replanned = np.asarray(arrays.get("mpc_replanned", np.empty(0)), dtype=bool)
    solve_mask = replanned if replanned.shape == planning_time.shape else np.isfinite(planning_time)
    solve_planning_time = planning_time[solve_mask] if solve_mask.size else planning_time
    best_cost = np.asarray(arrays.get("best_cost", np.empty(0)), dtype=np.float64)
    solve_best_cost = best_cost[solve_mask] if solve_mask.shape == best_cost.shape else best_cost
    actual_states = np.asarray(arrays.get("actual_states", np.empty((0, 0))), dtype=np.float64)
    q_des = np.asarray(arrays.get("q_des", np.empty((0, 0))), dtype=np.float64)
    joint_length = min(actual_states.shape[0], q_des.shape[0]) if actual_states.ndim == q_des.ndim == 2 else 0
    joint_rmse = float(np.sqrt(np.mean(np.square(actual_states[:joint_length, : q_des.shape[1]] - q_des[:joint_length])))) if joint_length and actual_states.shape[1] >= q_des.shape[1] else float("nan")
    position_error = np.asarray(arrays.get("ee_position_errors", np.empty(0)), dtype=np.float64)
    orientation_error = np.asarray(arrays.get("ee_orientation_errors", np.empty(0)), dtype=np.float64)
    packet_age = np.asarray(arrays.get("packet_age", np.empty(0)), dtype=np.float64)
    solve_count = int(np.asarray(arrays.get("planner_solve_count", np.sum(replanned))).reshape(-1)[0])
    late_drop_count = int(np.asarray(arrays.get("planner_late_drop_count", 0)).reshape(-1)[0])
    return {
        "label": label,
        "dataset_path": dataset_path,
        "steps": int(len(tracking)),
        "tracking_error_mean": float(np.mean(tracking)) if len(tracking) else float("nan"),
        "tracking_error_final": float(tracking[-1]) if len(tracking) else float("nan"),
        "failure_rate": float(np.mean(failures)) if len(failures) else float("nan"),
        "planning_time_mean": finite_mean(solve_planning_time),
        "best_cost_mean": finite_mean(solve_best_cost),
        "predicted_next_q_error_mean": finite_mean(predicted_q_error),
        "predicted_next_dq_error_mean": finite_mean(predicted_dq_error),
        "tcp_position_rmse_m": float(np.sqrt(np.mean(np.square(position_error[np.isfinite(position_error)])))) if np.any(np.isfinite(position_error)) else float("nan"),
        "orientation_rmse_rad": float(np.sqrt(np.mean(np.square(orientation_error[np.isfinite(orientation_error)])))) if np.any(np.isfinite(orientation_error)) else float("nan"),
        "joint_position_rmse_rad": joint_rmse,
        "control_period_p99_s": finite_percentile(np.asarray(arrays.get("actual_control_period_s", np.empty(0)), dtype=np.float64), 99.0),
        "control_wakeup_lateness_p99_s": finite_percentile(np.asarray(arrays.get("control_wakeup_lateness_s", np.empty(0)), dtype=np.float64), 99.0),
        "control_compute_p99_s": finite_percentile(np.asarray(arrays.get("control_step_wall_time", np.empty(0)), dtype=np.float64), 99.0),
        "control_deadline_miss_count": int(np.sum(np.asarray(arrays.get("control_deadline_miss", np.empty(0))) != 0)),
        "planner_solve_count": solve_count,
        "planner_update_rate_hz": float(np.asarray(arrays.get("planner_actual_update_rate_hz", np.nan)).reshape(-1)[0]),
        "planner_late_drop_rate": float(late_drop_count / solve_count) if solve_count else float("nan"),
        "active_packet_ratio": float(np.mean(packet_age >= 0.0)) if packet_age.size else float("nan"),
        "replay_q_error_k1_mean": finite_mean(replay_q_first),
        "replay_q_error_kH_mean": finite_mean(replay_q_terminal),
        "replay_dq_error_k1_mean": finite_mean(replay_dq_first),
        "replay_dq_error_kH_mean": finite_mean(replay_dq_terminal),
    }


def write_summary(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    save_root = RUN_CEM_MPC.resolve_runtime_path(args.save_dir)
    rows: list[dict[str, float | str | int]] = []
    for spec_text in args.model_spec:
        spec = parse_model_spec(spec_text)
        run_args = deepcopy(args)
        run_args.checkpoint = spec["checkpoint"]
        run_args.normalizer = spec["normalizer"]
        run_args.model_type = spec["model_type"]
        run_args.save_dir = str(save_root / spec["label"])
        result = RUN_CEM_MPC.run_closed_loop_mpc(run_args)
        save_mpc_run(Path(run_args.save_dir), result["arrays"], result["rows"])
        rows.append(summarize(spec["label"], result["arrays"], spec["dataset_path"]))
    write_summary(save_root / "model_abc_summary.csv", rows)
    print(f"Saved Model A/B/C summary to {save_root / 'model_abc_summary.csv'}")


if __name__ == "__main__":
    main()
