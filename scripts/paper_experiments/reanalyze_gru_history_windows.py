"""Reanalyse frozen GRU validation rollouts with complete recurrent history.

The original held-out validation starts at rollout step zero.  For a recurrent
model this requires padding the missing history with the initial token.  This
tool leaves those frozen results untouched and evaluates separate, later windows
whose first token has at least ``history_len`` preceding ground-truth tokens.
It performs learned-model inference only: no data collection, training, or
MuJoCo rollout is rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for candidate in (ROOT, DYNAMICS_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dynamics_modeling.scripts.eval_dynamics import predict_open_loop_segment, summarize_prediction
from mpc.robot_config import file_sha256, load_robot_spec
from neural_dynamics.rollout import load_dynamics_bundle
from scripts.paper_experiments.evaluation import write_csv, write_json


DEFAULT_INPUT = ROOT / "outputs" / "paper_final" / "diagnostics" / "gru_validation"
DEFAULT_OUTPUT = ROOT / "outputs" / "paper_model_validation_history_windows_v1"
DEFAULT_ROBOT_CONFIG = ROOT / "configs" / "robots" / "abb_irb2400.yaml"


def parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--horizons must be a comma-separated list of integers") from exc
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise argparse.ArgumentTypeError("--horizons must contain positive integers")
    return horizons


def action_std_label(value: float) -> str:
    """Canonicalize float32 archive metadata for stable aggregate grouping."""
    return format(float(value), ".6g")


def window_starts(
    rollout_length: int,
    *,
    history_len: int,
    max_horizon: int,
    stride: int,
) -> tuple[int, ...]:
    """Return common starts that have true history and fit every requested horizon."""
    if history_len <= 0 or max_horizon <= 0 or stride <= 0:
        raise ValueError("history_len, max_horizon, and stride must be positive")
    last_start = rollout_length - max_horizon
    if last_start < history_len:
        raise ValueError(
            "Rollout is too short for the requested history and horizon: "
            f"length={rollout_length}, history_len={history_len}, max_horizon={max_horizon}"
        )
    return tuple(range(history_len, last_start + 1, stride))


def _mean_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "p95": float(np.percentile(array, 95)),
        "worst": float(np.max(array)),
        "n_rollouts": int(array.size),
    }


def _input_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size": path.stat().st_size,
    }


def reanalyze(
    input_dir: Path,
    output_dir: Path,
    robot_config: Path,
    horizons: tuple[int, ...],
    stride: int,
    overwrite: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing reanalysis: {output_dir}")
    validation_manifest_path = input_dir / "validation_manifest.json"
    if not validation_manifest_path.is_file():
        raise FileNotFoundError(validation_manifest_path)
    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(validation_manifest["checkpoint"]["path"])
    normalizer = Path(validation_manifest["normalizer"]["path"])
    if not checkpoint.is_file() or not normalizer.is_file():
        raise FileNotFoundError("Frozen checkpoint or normalizer is unavailable")

    robot = load_robot_spec(robot_config, validate_model=True)
    bundle = load_dynamics_bundle(
        checkpoint,
        normalizer,
        "gru",
        robot.n_joints,
        "cpu",
        16,
        expected_robot_spec=robot,
    )
    raw_rollouts = sorted(input_dir.glob("evaluation_rollout_*.npz"))
    if not raw_rollouts:
        raise FileNotFoundError(f"No frozen evaluation rollouts found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"q{index}" for index in range(robot.n_joints)] + [
        f"dq{index}" for index in range(robot.n_joints)
    ]
    window_rows: list[dict[str, Any]] = []
    rollout_metric_values: dict[tuple[int, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    starts_by_rollout: dict[int, tuple[int, ...]] = {}

    for rollout_index, rollout_path in enumerate(raw_rollouts):
        with np.load(rollout_path, allow_pickle=False) as archive:
            true_states = np.asarray(archive["true_states"], dtype=np.float32)
            actions = np.asarray(archive["actions"], dtype=np.float32)
            true_next_states = np.asarray(archive["true_next_states"], dtype=np.float32)
            action_std = float(np.asarray(archive["action_std"]).item())
            action_std_text = action_std_label(action_std)
            motion_mode = str(np.asarray(archive["motion_mode"]).item())
        if not (len(true_states) == len(actions) == len(true_next_states)):
            raise ValueError(f"Inconsistent frozen rollout lengths in {rollout_path}")
        starts = window_starts(
            len(true_states),
            history_len=bundle.history_len,
            max_horizon=max(horizons),
            stride=stride,
        )
        starts_by_rollout[rollout_index] = starts
        for start in starts:
            for horizon in horizons:
                prediction = predict_open_loop_segment(
                    bundle.model,
                    bundle.normalizer,
                    "gru",
                    true_states,
                    actions,
                    start,
                    horizon,
                    bundle.history_len,
                    2 * robot.n_joints,
                    torch.device("cpu"),
                    bundle.target_mode,
                    bundle.control_dt,
                    record_next_states=True,
                )
                summary = summarize_prediction(
                    true_next_states[start : start + horizon],
                    prediction,
                    labels,
                )
                row = {
                    "rollout": rollout_index,
                    "motion_mode": motion_mode,
                    "action_std": action_std_text,
                    "window_start": start,
                    "horizon": horizon,
                    **summary,
                }
                window_rows.append(row)
                bucket = rollout_metric_values[(rollout_index, action_std_text, horizon)]
                for metric in ("q_rmse", "dq_rmse", "rmse", "divergence_rate"):
                    bucket[metric].append(float(summary[metric]))

    rollout_rows: list[dict[str, Any]] = []
    for (rollout, action_std, horizon), metrics in sorted(rollout_metric_values.items()):
        row: dict[str, Any] = {
            "rollout": rollout,
            "action_std": action_std,
            "horizon": horizon,
            "n_windows": len(metrics["q_rmse"]),
        }
        # All windows use the same horizon and joint count, so RMS aggregation
        # over window RMSEs equals pooling squared errors across their samples.
        for metric in ("q_rmse", "dq_rmse", "rmse"):
            values = np.asarray(metrics[metric], dtype=np.float64)
            row[metric] = float(np.sqrt(np.mean(np.square(values))))
        row["divergence_rate"] = float(np.mean(metrics["divergence_rate"]))
        rollout_rows.append(row)

    aggregate_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rollout_rows:
        grouped[(str(row["action_std"]), int(row["horizon"]))].append(row)
    for (action_std, horizon), rows in sorted(grouped.items()):
        for metric in ("q_rmse", "dq_rmse", "rmse", "divergence_rate"):
            aggregate_rows.append(
                {
                    "mode": "open_loop_true_history_windows",
                    "action_std": action_std,
                    "horizon": horizon,
                    "metric": metric,
                    **_mean_summary([float(row[metric]) for row in rows]),
                }
            )

    write_csv(output_dir / "window_metrics.csv", window_rows)
    write_csv(output_dir / "rollout_metrics.csv", rollout_rows)
    write_csv(output_dir / "aggregate.csv", aggregate_rows)
    manifest = {
        "schema_version": 1,
        "kind": "frozen_gru_true_history_window_reanalysis",
        "description": (
            "Offline reanalysis of saved held-out rollouts. Starts have complete "
            "ground-truth recurrent history; no MuJoCo rollout, data collection, "
            "or training was rerun."
        ),
        "source_validation_manifest": _input_identity(validation_manifest_path),
        "checkpoint": _input_identity(checkpoint),
        "normalizer": _input_identity(normalizer),
        "robot_identity": robot.artifact_identity(),
        "history_len": bundle.history_len,
        "horizons": list(horizons),
        "window_stride": stride,
        "starts_by_rollout": {str(key): list(value) for key, value in starts_by_rollout.items()},
        "raw_rollouts": [_input_identity(path) for path in raw_rollouts],
        "aggregation": (
            "For each rollout and horizon, pool equal-length windows by RMS; "
            "then report the mean and distribution over rollouts within each action-std group."
        ),
    }
    write_json(output_dir / "analysis_manifest.json", manifest)
    return {
        "output": str(output_dir),
        "rollouts": len(raw_rollouts),
        "windows": len(window_rows),
        "aggregate_rows": len(aggregate_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--horizons", type=parse_horizons, default=(1, 5, 10, 20))
    parser.add_argument("--window-stride", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            reanalyze(
                args.input_dir,
                args.output_dir,
                args.robot_config,
                args.horizons,
                args.window_stride,
                args.overwrite,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
