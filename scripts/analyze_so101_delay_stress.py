#!/usr/bin/env python3
"""Analyze the six-run SO101 ThreadedAsync delay-stress matrix offline."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAD_TO_DEG = 180.0 / np.pi
SHAPE_LOOP_SEGMENT = 3


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) if path.suffix in {".yaml", ".yml"} else json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _scalar(arrays: dict[str, np.ndarray], name: str, default: Any = None) -> Any:
    if name not in arrays:
        return default
    value = np.asarray(arrays[name]).reshape(-1)
    return value[0].item() if value.size else default


def _rms(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return None if not values.size else float(np.sqrt(np.mean(np.square(values))))


def _count_true(arrays: dict[str, np.ndarray], name: str, limit: int) -> int:
    values = np.asarray(arrays.get(name, np.zeros(limit, dtype=bool)), dtype=bool)
    return int(np.count_nonzero(values.reshape(-1)[:limit]))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _trial_metrics(stress: dict[str, Any], base: dict[str, Any], trial_id: str,
                   condition: str, output_root: Path) -> dict[str, Any]:
    output = output_root / f"{trial_id}__{condition}"
    manifest_path = output / "trial_manifest.json"
    rollout_path = output / "rollout.npz"
    row: dict[str, Any] = {
        "trial_id": trial_id,
        "condition": condition,
        "output_dir": str(output),
        "status": "missing",
    }
    if not rollout_path.exists():
        return row
    manifest = _load(manifest_path) if manifest_path.exists() else {}
    arrays = _load_npz(rollout_path)
    actual = np.asarray(arrays.get("actual_states", arrays.get("states")), dtype=np.float64)
    q_des = np.asarray(arrays["q_des"], dtype=np.float64)
    if actual.ndim != 2 or q_des.ndim != 2 or q_des.shape[1] != 5:
        row.update({"status": "analysis_error", "reason": "invalid actual_states/q_des shape"})
        return row
    label = trial_id.rsplit("_nn_mpc", 1)[0]
    manifest_path_ref = _path(base["heldout_reference_manifest"])
    ref_manifest = json.loads(manifest_path_ref.read_text(encoding="utf-8"))
    entry = ref_manifest["artifacts"][label]
    task_path = _path(base["heldout_reference_root"]) / label / "reference.npz"
    task = _load_npz(task_path)
    recorded = min(actual.shape[0], q_des.shape[0])
    execution_steps = int(entry["execution_steps"])
    n = min(recorded, execution_steps, len(task["segment_ids"]))
    mask = np.zeros(recorded, dtype=bool)
    mask[:n] = np.asarray(task["segment_ids"][:n], dtype=np.int64) == SHAPE_LOOP_SEGMENT
    if not np.any(mask[:n]):
        mask[:n] = True
    error_deg = (actual[:recorded, :5] - q_des[:recorded]) * RAD_TO_DEG
    velocity = np.asarray(arrays.get("command_velocity", np.zeros((recorded, 5))), dtype=np.float64)[:recorded]
    acceleration = np.asarray(arrays.get("command_acceleration", np.zeros((recorded, 5))), dtype=np.float64)[:recorded]
    latency = np.asarray(arrays.get("planner_end_to_end_latency_s", np.empty(0)), dtype=np.float64)
    latency = latency[np.isfinite(latency)] * 1000.0
    tx_success = np.asarray(
        arrays.get("tx_local_success", np.ones(recorded, dtype=bool)), dtype=bool
    ).reshape(-1)[:recorded]
    command_acceleration_flags = _count_true(
        arrays, "command_acceleration_violation_flags", recorded
    )
    safety = {
        "tx_failure_count": int(recorded - np.count_nonzero(tx_success)),
        "control_deadline_miss_count": _count_true(
            arrays, "control_deadline_miss", recorded
        ),
        "command_velocity_violation_count": _count_true(
            arrays, "command_velocity_violation_flags", recorded
        ),
        "command_acceleration_violation_count": command_acceleration_flags,
        "command_acceleration_quantization_exceedance_count": _count_true(
            arrays, "command_acceleration_quantization_exceedance_flags", recorded
        ),
        "planner_failure_count": int(_scalar(arrays, "planner_failure_count", 0) or 0),
        "planner_late_drop_count": int(_scalar(arrays, "planner_late_drop_count", 0) or 0),
        "packet_expiration_count": int(_scalar(arrays, "packet_expiration_count", 0) or 0),
    }
    safety["safety_violation_count"] = int(sum(safety[name] for name in (
        "tx_failure_count", "command_velocity_violation_count",
        "command_acceleration_violation_count", "planner_failure_count",
    )))
    row.update({
        "status": "complete" if int(manifest.get("return_code", 0)) == 0 and recorded >= execution_steps else "incomplete",
        "recorded_steps": int(recorded),
        "expected_steps": execution_steps,
        "joint_rmse_deg": _rms(error_deg[mask]),
        "command_velocity_rms_rad_s": _rms(velocity[mask]),
        "command_acceleration_rms_rad_s2": _rms(acceleration[mask]),
        "planner_latency_mean_ms": None if not latency.size else float(np.mean(latency)),
        "planner_latency_p95_ms": None if not latency.size else float(np.percentile(latency, 95.0)),
        "planner_latency_p99_ms": None if not latency.size else float(np.percentile(latency, 99.0)),
        "planner_latency_max_ms": None if not latency.size else float(np.max(latency)),
        "injected_planner_delay_ms": float(_scalar(arrays, "injected_planner_delay_ms", np.nan)),
    })
    row.update(safety)
    return row


def _paired(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_trial: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_trial.setdefault(str(row["trial_id"]), {})[str(row["condition"])] = row
    values = []
    records = []
    for trial_id in sorted(by_trial):
        pair = by_trial[trial_id]
        if pair.get("baseline", {}).get("status") != "complete" or pair.get("delay33", {}).get("status") != "complete":
            continue
        difference = float(pair["delay33"]["joint_rmse_deg"] - pair["baseline"]["joint_rmse_deg"])
        values.append(difference)
        records.append({
            "trial_id": trial_id,
            "baseline_joint_rmse_deg": pair["baseline"]["joint_rmse_deg"],
            "delay33_joint_rmse_deg": pair["delay33"]["joint_rmse_deg"],
            "delay33_minus_baseline_deg": difference,
        })
    array = np.asarray(values, dtype=np.float64)
    mean = None if not array.size else float(np.mean(array))
    std = None if array.size < 2 else float(np.std(array, ddof=1))
    critical = {1: 0.0, 2: 12.7062047, 3: 4.3026527}.get(len(array), 1.96)
    se = 0.0 if array.size < 2 else float(std / np.sqrt(array.size))
    return {
        "n": int(array.size),
        "mean_delay33_minus_baseline_deg": mean,
        "std_deg": std,
        "paired_t_ci95_deg": None if mean is None else [mean - critical * se, mean + critical * se],
        "delay33_worse_count": None if not array.size else int(np.sum(array > 0.0)),
        "pairs": records,
    }


def _condition_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return pooled planner-event timing plus run-level tracking aggregates."""
    aggregates: dict[str, Any] = {}
    for condition in ("baseline", "delay33"):
        condition_rows = [
            row for row in rows
            if row.get("condition") == condition and row.get("status") == "complete"
        ]
        latencies: list[np.ndarray] = []
        for row in condition_rows:
            rollout_path = Path(str(row["output_dir"])) / "rollout.npz"
            if not rollout_path.exists():
                continue
            arrays = _load_npz(rollout_path)
            values = np.asarray(
                arrays.get("planner_end_to_end_latency_s", np.empty(0)),
                dtype=np.float64,
            )
            values = values[np.isfinite(values)] * 1000.0
            if values.size:
                latencies.append(values)
        pooled = np.concatenate(latencies) if latencies else np.empty(0, dtype=np.float64)
        aggregates[condition] = {
            "complete_run_count": len(condition_rows),
            "event_count": int(pooled.size),
            "joint_rmse_mean_deg": None if not condition_rows else float(
                np.mean([row["joint_rmse_deg"] for row in condition_rows])
            ),
            "command_velocity_rms_mean_rad_s": None if not condition_rows else float(
                np.mean([row["command_velocity_rms_rad_s"] for row in condition_rows])
            ),
            "command_acceleration_rms_mean_rad_s2": None if not condition_rows else float(
                np.mean([row["command_acceleration_rms_rad_s2"] for row in condition_rows])
            ),
            "planner_latency_mean_ms": None if not pooled.size else float(np.mean(pooled)),
            "planner_latency_p95_ms": None if not pooled.size else float(np.percentile(pooled, 95.0)),
            "planner_latency_p99_ms": None if not pooled.size else float(np.percentile(pooled, 99.0)),
            "planner_latency_max_ms": None if not pooled.size else float(np.max(pooled)),
        }
    return aggregates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="configs/experiments/so101_threaded_delay_stress_20260810_v2.yaml",
    )
    parser.add_argument("--output", default=None, help="JSON output; defaults to the active root summary.")
    args = parser.parse_args()
    stress = _load(_path(args.protocol))
    base = _load(_path(stress["base_protocol"]))
    output_root = _path(stress["active_output_root"])
    rows = [
        _trial_metrics(stress, base, trial_id, condition, output_root)
        for trial_id in stress["active_trial_ids"]
        for condition in ("baseline", "delay33")
    ]
    summary = {
        "schema_version": 2,
        "protocol_id": stress["protocol_id"],
        "base_protocol_id": base["protocol_id"],
        "injected_planner_delay_ms": float(stress["injected_planner_delay_ms"]),
        "rows": rows,
        "paired": _paired(rows),
        "condition_aggregates": _condition_aggregates(rows),
        "interpretation_boundary": "ThreadedAsync physical stress/deployability validation; no NaiveDelayed hardware ablation.",
    }
    output = _path(args.output) if args.output else output_root / "delay_stress_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
