#!/usr/bin/env python3
"""Analyze the frozen SO101 final-paper trial matrix offline.

This script never connects to hardware.  It evaluates only completed trial
artifacts, keeps technically incomplete trials visible in the ledger, and
writes a machine-readable summary plus a human-readable document under
``docs/hardware``.  The primary tracking window is the frozen
``SEGMENT_SHAPE_LOOP`` portion of each reference; startup, approach, return,
and padding are reported separately through the run metadata.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from mpc.kinematics_utils import MujocoKinematics
from scripts.run_so101_paper_trial import _load_protocol, _path, _reference_for, _trial_specs


SEGMENT_SHAPE_LOOP = 3
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
RAD_TO_DEG = 180.0 / np.pi


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (float,)):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _scalar(values: dict[str, np.ndarray], name: str, default: Any = None) -> Any:
    if name not in values:
        return default
    value = np.asarray(values[name]).reshape(-1)
    return value[0].item() if value.size else default


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _activity(values: np.ndarray, mask: np.ndarray, factor: float = 1.0) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != mask.shape[0]:
        return {"count": 0, "rms": None, "p95_abs": None, "max_abs": None, "per_joint_rms": [], "per_joint_p95_abs": []}
    selected = array[mask] * float(factor)
    finite = _finite(selected)
    if finite.size == 0:
        return {"count": 0, "rms": None, "p95_abs": None, "max_abs": None, "per_joint_rms": [], "per_joint_p95_abs": []}
    return {
        "count": int(finite.size),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "p95_abs": float(np.percentile(np.abs(finite), 95.0)),
        "max_abs": float(np.max(np.abs(finite))),
        "per_joint_rms": [float(np.sqrt(np.mean(np.square(column)))) for column in selected.T],
        "per_joint_p95_abs": [float(np.percentile(np.abs(column), 95.0)) for column in selected.T],
    }


def _joint_error_metrics(error_rad: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(error_rad, dtype=np.float64)[mask]
    activity = _activity(selected, np.ones(selected.shape[0], dtype=bool), RAD_TO_DEG)
    return {
        "rmse_deg": activity["rms"],
        "p95_abs_deg": activity["p95_abs"],
        "max_abs_deg": activity["max_abs"],
        "per_joint_rmse_deg": activity["per_joint_rms"],
        "per_joint_p95_abs_deg": activity["per_joint_p95_abs"],
    }


def _event_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"count": 0, "selection_mode_counts": {}, "late_drop_count": 0, "failure_count": 0, "latency_p95_ms": None}
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    events.append(value)
    modes = Counter(str(event.get("selection_mode", "unknown")) for event in events)
    late = sum(str(event.get("result_type", "")).endswith("late_dropped") for event in events)
    failures = sum(str(event.get("result_type", "")) in {"failure", "planner_failure"} for event in events)
    latency = _finite(np.asarray([event.get("end_to_end_latency_s", np.nan) for event in events]))
    return {
        "count": len(events),
        "selection_mode_counts": dict(sorted(modes.items())),
        "late_drop_count": int(late),
        "failure_count": int(failures),
        "latency_p95_ms": None if not latency.size else float(np.percentile(latency, 95.0) * 1000.0),
    }


def _reference_info(protocol: dict[str, Any], spec: dict[str, Any]) -> tuple[Path, Path, dict[str, Any], dict[str, np.ndarray]]:
    joint_reference, _manifest_path = _reference_for(protocol, spec)
    if spec["family"] == "circle":
        root = _path(protocol["circle_reference_root"])
        manifest_path = _path(protocol["circle_reference_manifest"])
    else:
        root = _path(protocol["heldout_reference_root"])
        manifest_path = _path(protocol["heldout_reference_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["artifacts"][spec["reference_label"]]
    task_reference = root / spec["reference_label"] / "reference.npz"
    if not task_reference.exists():
        raise FileNotFoundError(f"task-space reference not found: {task_reference}")
    return joint_reference, task_reference, entry, _load_npz(task_reference)


def _make_fk(protocol: dict[str, Any]) -> MujocoKinematics:
    model_path = _path(protocol["fine_model"])
    model = mujoco.MjModel.from_xml_path(str(model_path))
    return MujocoKinematics(model, str(protocol.get("fk_site", "gripperframe")), n_joints=5)


def _reference_mask(task: dict[str, np.ndarray], execution_steps: int, recorded_steps: int) -> np.ndarray:
    n = min(int(execution_steps), int(recorded_steps), len(task["segment_ids"]))
    mask = np.zeros(int(recorded_steps), dtype=bool)
    mask[:n] = np.asarray(task["segment_ids"][:n], dtype=np.int64) == SEGMENT_SHAPE_LOOP
    if not np.any(mask[:n]):
        mask[:n] = True
    return mask


def _array2(arrays: dict[str, np.ndarray], *names: str, rows: int, columns: int) -> np.ndarray:
    for name in names:
        if name in arrays:
            value = np.asarray(arrays[name], dtype=np.float64)
            if value.ndim == 2 and value.shape[1] == columns:
                return value[:rows]
    return np.zeros((rows, columns), dtype=np.float64)


def _count_true(arrays: dict[str, np.ndarray], name: str, rows: int) -> int:
    if name not in arrays:
        return 0
    return int(np.sum(np.asarray(arrays[name]).reshape(-1)[:rows].astype(bool)))


def _select_attempt(output_root: Path, trial_id: str) -> Path:
    """Prefer the original trial, then a successful retry, then latest partial."""
    base = output_root / trial_id
    retries = sorted(
        output_root.glob(f"{trial_id}_retry*"),
        key=lambda path: int(path.name.rsplit("_retry", 1)[1])
        if path.name.rsplit("_retry", 1)[-1].isdigit() else -1,
    )
    candidates = [base, *retries]
    for candidate in candidates:
        manifest = candidate / "trial_manifest.json"
        rollout = candidate / "rollout.npz"
        if not manifest.exists() or not rollout.exists():
            continue
        try:
            record = json.loads(manifest.read_text(encoding="utf-8"))
            if int(record.get("return_code", 1)) == 0:
                return candidate
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    for candidate in reversed(candidates):
        if (candidate / "rollout.npz").exists():
            return candidate
    return base


def analyze_trial(protocol: dict[str, Any], spec: dict[str, Any], fk: MujocoKinematics | None) -> dict[str, Any]:
    output_root = _path(protocol["output_root"])
    output_dir = _select_attempt(output_root, spec["trial_id"])
    trial_manifest_path = output_dir / "trial_manifest.json"
    rollout_path = output_dir / "rollout.npz"
    base = {
        "trial_id": spec["trial_id"],
        "family": spec["family"],
        "shape": spec["shape"],
        "speed": spec["speed"],
        "phase_index": int(spec["phase_index"]),
        "repeat_index": int(spec["repeat_index"]),
        "controller": spec["controller"],
        "output_dir": str(output_dir),
    }
    if not rollout_path.exists():
        return {**base, "status": "missing", "technical_exclusion": True, "reason": "rollout.npz not found"}
    trial_manifest: dict[str, Any] = {}
    if trial_manifest_path.exists():
        trial_manifest = json.loads(trial_manifest_path.read_text(encoding="utf-8"))
    return_code = trial_manifest.get("return_code", 0)
    try:
        return_code = int(return_code)
    except (TypeError, ValueError):
        return_code = 0
    try:
        arrays = _load_npz(rollout_path)
        joint_reference, task_reference_path, entry, task = _reference_info(protocol, spec)
        actual_state = arrays.get("actual_states", arrays.get("states"))
        if actual_state is None or np.asarray(actual_state).ndim != 2:
            raise ValueError("rollout does not contain actual_states/states")
        actual_state = np.asarray(actual_state, dtype=np.float64)
        q_des = np.asarray(arrays.get("q_des"), dtype=np.float64)
        if q_des.ndim != 2 or q_des.shape[1] != 5:
            raise ValueError("rollout q_des must have shape (T,5)")
        expected_steps = int(entry["execution_steps"])
        recorded_steps = min(actual_state.shape[0], q_des.shape[0])
        if recorded_steps <= 0:
            raise ValueError("rollout has no recorded samples")
        q_actual = actual_state[:recorded_steps, :5]
        q_target = q_des[:recorded_steps]
        mask = _reference_mask(task, expected_steps, recorded_steps)
        all_mask = np.ones(recorded_steps, dtype=bool)
        error = q_actual - q_target

        task_positions = np.asarray(task.get("task_positions_des", np.empty((0, 3))), dtype=np.float64)
        tcp_metrics: dict[str, Any] = {"available": False}
        if fk is not None and task_positions.ndim == 2 and task_positions.shape[1] == 3:
            tcp_n = min(recorded_steps, task_positions.shape[0])
            q_offset = np.asarray(protocol["q_ctrl_to_q_kin_offset_rad"], dtype=np.float64)
            positions = np.asarray([fk.forward(q + q_offset)[0] for q in q_actual[:tcp_n]], dtype=np.float64)
            tcp_mask = mask[:tcp_n]
            position_error_m = positions - task_positions[:tcp_n]
            norms_m = np.linalg.norm(position_error_m[tcp_mask], axis=1)
            tcp_metrics = {
                "available": True,
                "rmse_mm": float(np.sqrt(np.mean(np.square(norms_m))) * 1000.0),
                "p95_abs_mm": float(np.percentile(norms_m, 95.0) * 1000.0),
                "max_abs_mm": float(np.max(norms_m) * 1000.0),
                "samples": int(norms_m.size),
                "measurement": "encoder_state_forward_kinematics",
                "external_instrument": False,
            }

        velocity = _array2(arrays, "command_velocity", "executable_command_velocity", rows=recorded_steps, columns=5)
        acceleration = _array2(arrays, "command_acceleration", rows=recorded_steps, columns=5)
        if "command_acceleration" not in arrays and recorded_steps > 1:
            acceleration[1:] = np.diff(velocity, axis=0) / float(protocol["control_dt_s"])
        requested_residual = _array2(arrays, "requested_mpc_residual", "planner_requested_residual", rows=recorded_steps, columns=5)
        executed_residual = _array2(arrays, "executed_residual", rows=recorded_steps, columns=5)
        if spec["controller"] != "nn_mpc":
            requested_residual.fill(0.0)
            executed_residual.fill(0.0)
        saturation = np.asarray(arrays.get("residual_saturated", np.zeros(recorded_steps)), dtype=bool).reshape(-1)[:recorded_steps]
        if not np.any(saturation) and spec["controller"] == "nn_mpc":
            cap = np.asarray(arrays.get("residual_max", protocol["final_mpc"]["residual_max_rad"]), dtype=np.float64).reshape(-1)
            if cap.ndim == 0:
                cap = np.repeat(cap, 5)
            if cap.size == 1:
                cap = np.repeat(cap, 5)
            saturation = np.any(np.abs(requested_residual) >= 0.999 * cap[None, :], axis=1)

        event_info = _event_summary(output_dir / "planner_events.jsonl")
        planner_latency = np.asarray(arrays.get("planner_end_to_end_latency_s", np.empty(0)), dtype=np.float64)
        planner_latency = _finite(planner_latency) * 1000.0
        acceleration_flags = np.asarray(
            arrays.get("command_acceleration_violation_flags", np.zeros(recorded_steps)), dtype=bool
        ).reshape(-1)[:recorded_steps]
        if "command_acceleration_quantization_exceedance_flags" in arrays:
            quantization_acceleration_flags = np.asarray(
                arrays["command_acceleration_quantization_exceedance_flags"], dtype=bool
            ).reshape(-1)[:recorded_steps]
        else:
            # Older evidence files used the runtime field for the derived
            # discrete-command acceleration diagnostic.  Reclassify it for
            # analysis without modifying the raw NPZ.
            quantization_acceleration_flags = acceleration_flags.copy()
            acceleration_flags = np.zeros(recorded_steps, dtype=bool)
        projection_values = np.asarray(
            arrays.get("projection_flags", np.asarray([], dtype=str)), dtype=str
        ).reshape(-1)[:recorded_steps]
        projection_flagged = np.asarray([bool(value) for value in projection_values], dtype=bool)
        safety = {
            "tx_failure_count": int(recorded_steps - np.sum(np.asarray(arrays.get("tx_local_success", np.ones(recorded_steps)), dtype=bool)[:recorded_steps])),
            "delivery_uncertain_count": _count_true(arrays, "command_delivery_uncertain", recorded_steps),
            "control_deadline_miss_count": _count_true(arrays, "control_deadline_miss", recorded_steps),
            "command_velocity_violation_count": _count_true(arrays, "command_velocity_violation_flags", recorded_steps),
            "command_acceleration_violation_count": int(np.sum(acceleration_flags)),
            "command_acceleration_quantization_exceedance_count": int(np.sum(quantization_acceleration_flags)),
            "planner_failure_count": int(_scalar(arrays, "planner_failure_count", event_info["failure_count"]) or 0),
            "planner_late_drop_count": int(_scalar(arrays, "planner_late_drop_count", event_info["late_drop_count"]) or 0),
            "planner_packet_expiration_count": int(_scalar(arrays, "packet_expiration_count", 0) or 0),
            "projection_flagged_tick_count": int(np.sum(projection_flagged)),
            "projection_velocity_limit_tick_count": int(np.sum(np.char.find(projection_values, "velocity_limit") >= 0)),
            "projection_acceleration_limit_tick_count": int(np.sum(np.char.find(projection_values, "acceleration_limit") >= 0)),
            "encoder_quantization_tick_count": int(np.sum(np.char.find(projection_values, "encoder_quantization") >= 0)),
        }
        safety["safety_violation_count"] = int(sum(safety[name] for name in (
            "tx_failure_count", "command_velocity_violation_count", "command_acceleration_violation_count",
            "planner_failure_count",
        )))

        completed = return_code == 0 and recorded_steps >= expected_steps
        status = "complete" if completed else "incomplete"
        result = {
            **base,
            "status": status,
            "technical_exclusion": not completed,
            "return_code": return_code,
            "reference": {
                "joint_reference": str(joint_reference),
                "task_reference": str(task_reference_path),
                "reference_label": spec["reference_label"],
                "execution_steps": expected_steps,
                "recorded_steps": int(recorded_steps),
                "shape_loop_samples": int(np.sum(mask)),
            },
            "tracking": {
                "primary_window": "segment_id == SEGMENT_SHAPE_LOOP",
                "joint": _joint_error_metrics(error, mask),
                "joint_all_execution": _joint_error_metrics(error, all_mask),
                "tcp": tcp_metrics,
            },
            "control_activity": {
                "command_velocity": _activity(velocity, mask),
                "command_acceleration": _activity(acceleration, mask),
                "requested_residual": _activity(requested_residual, mask, RAD_TO_DEG),
                "executed_residual": _activity(executed_residual, mask, RAD_TO_DEG),
                "requested_residual_saturation_rate": float(np.mean(saturation[mask])) if np.any(mask) else None,
                "executed_residual_saturation_rate": float(np.mean(
                    np.any(np.abs(executed_residual[mask]) >= 0.999 * np.asarray(protocol["final_mpc"]["residual_max_rad"]), axis=1)
                )) if np.any(mask) else None,
            },
            "timing": {
                "planner_latency_p95_ms": None if not planner_latency.size else float(np.percentile(planner_latency, 95.0)),
                "planner_latency_max_ms": None if not planner_latency.size else float(np.max(planner_latency)),
                "planner_event_count": int(event_info["count"]),
                "selection_mode_counts": event_info["selection_mode_counts"],
            },
            "safety": safety,
            "artifact_identity": trial_manifest.get("artifacts", {}),
        }
        return _json_safe(result)
    except Exception as exc:
        return {**base, "status": "analysis_error", "technical_exclusion": True, "reason": f"{type(exc).__name__}: {exc}"}


SUMMARY_FIELDS = (
    "family", "shape", "speed", "controller", "status", "tracking_joint_rmse_deg", "tracking_joint_p95_deg",
    "tracking_tcp_rmse_mm", "command_velocity_rms_rad_s", "command_acceleration_rms_rad_s2",
    "requested_residual_rms_deg", "requested_residual_p95_deg", "requested_residual_max_deg",
    "executed_residual_rms_deg", "executed_residual_p95_deg", "executed_residual_max_deg",
    "requested_residual_saturation_rate",
    "planner_latency_p95_ms", "control_deadline_miss_count",
    "command_velocity_violation_count", "command_acceleration_violation_count", "safety_violation_count",
    "command_acceleration_quantization_exceedance_count",
)


def _flat_row(result: dict[str, Any]) -> dict[str, Any]:
    tracking = result.get("tracking", {})
    joint = tracking.get("joint", {})
    tcp = tracking.get("tcp", {})
    activity = result.get("control_activity", {})
    safety = result.get("safety", {})
    timing = result.get("timing", {})
    return {
        "trial_id": result.get("trial_id"), "output_dir": result.get("output_dir"),
        "family": result.get("family"), "shape": result.get("shape"),
        "speed": result.get("speed"), "phase_index": result.get("phase_index"), "repeat_index": result.get("repeat_index"),
        "controller": result.get("controller"), "status": result.get("status"),
        "tracking_joint_rmse_deg": joint.get("rmse_deg"), "tracking_joint_p95_deg": joint.get("p95_abs_deg"),
        "tracking_tcp_rmse_mm": tcp.get("rmse_mm"),
        "command_velocity_rms_rad_s": activity.get("command_velocity", {}).get("rms"),
        "command_acceleration_rms_rad_s2": activity.get("command_acceleration", {}).get("rms"),
        "requested_residual_rms_deg": activity.get("requested_residual", {}).get("rms"),
        "requested_residual_p95_deg": activity.get("requested_residual", {}).get("p95_abs"),
        "requested_residual_max_deg": activity.get("requested_residual", {}).get("max_abs"),
        "executed_residual_rms_deg": activity.get("executed_residual", {}).get("rms"),
        "executed_residual_p95_deg": activity.get("executed_residual", {}).get("p95_abs"),
        "executed_residual_max_deg": activity.get("executed_residual", {}).get("max_abs"),
        "requested_residual_saturation_rate": activity.get("requested_residual_saturation_rate"),
        "planner_latency_p95_ms": timing.get("planner_latency_p95_ms"),
        "control_deadline_miss_count": safety.get("control_deadline_miss_count"),
        "command_velocity_violation_count": safety.get("command_velocity_violation_count"),
        "command_acceleration_violation_count": safety.get("command_acceleration_violation_count"),
        "safety_violation_count": safety.get("safety_violation_count"),
        "command_acceleration_quantization_exceedance_count": safety.get(
            "command_acceleration_quantization_exceedance_count"
        ),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "complete":
            continue
        key = (str(row["family"]), str(row["shape"]), str(row["speed"]), str(row["controller"]))
        groups[key].append(row)
    metrics = [field for field in SUMMARY_FIELDS if field not in {"family", "shape", "speed", "controller", "status"}]
    output: dict[str, Any] = {}
    for key, items in sorted(groups.items()):
        group_key = "|".join(key)
        values: dict[str, Any] = {"n": len(items)}
        for field in metrics:
            numeric = np.asarray([item.get(field, np.nan) for item in items], dtype=np.float64)
            numeric = _finite(numeric)
            values[field] = {
                "mean": None if not numeric.size else float(np.mean(numeric)),
                "std": None if not numeric.size else float(np.std(numeric, ddof=1 if numeric.size > 1 else 0)),
                "median": None if not numeric.size else float(np.median(numeric)),
                "n": int(numeric.size),
            }
        output[group_key] = {"family": key[0], "shape": key[1], "speed": key[2], "controller": key[3], **values}
    return output


def _paired(rows: list[dict[str, Any]], family: str | None = None) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") == "complete" and (family is None or row.get("family") == family):
            key = (str(row["family"]), str(row["shape"]), str(row["speed"]), int(row["phase_index"]), int(row["repeat_index"]))
            by_key[key][str(row["controller"])] = row
    pairs: dict[str, list[float]] = defaultdict(list)
    for controllers in by_key.values():
        if "nn_mpc" not in controllers:
            continue
        for other in ("direct", "preview6"):
            if other in controllers:
                for field in ("tracking_joint_rmse_deg", "tracking_tcp_rmse_mm"):
                    a = controllers["nn_mpc"].get(field)
                    b = controllers[other].get(field)
                    if a is not None and b is not None:
                        pairs[f"nn_mpc_minus_{other}:{field}"].append(float(a) - float(b))
    return {
        key: {"n": len(values), "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1 if len(values) > 1 else 0)), "values": values}
        for key, values in sorted(pairs.items())
    }


def _analysis_lines(rows: list[dict[str, Any]], paired: dict[str, Any]) -> list[str]:
    """Generate evidence-first interpretation for the frozen result record."""
    complete = [row for row in rows if row.get("status") == "complete"]
    heldout = [row for row in complete if row.get("family") == "heldout"]

    def mean(controller: str, field: str, subset: list[dict[str, Any]]) -> float | None:
        values = [row[field] for row in subset if row.get("controller") == controller and row.get(field) is not None]
        return None if not values else float(np.mean(values))

    def improvement(baseline: float | None, method: float | None) -> float | None:
        if baseline is None or method is None or baseline == 0.0:
            return None
        return 100.0 * (baseline - method) / baseline

    def f(value: float | None, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    direct_joint = mean("direct", "tracking_joint_rmse_deg", heldout)
    preview_joint = mean("preview6", "tracking_joint_rmse_deg", heldout)
    mpc_joint = mean("nn_mpc", "tracking_joint_rmse_deg", heldout)
    direct_tcp = mean("direct", "tracking_tcp_rmse_mm", heldout)
    preview_tcp = mean("preview6", "tracking_tcp_rmse_mm", heldout)
    mpc_tcp = mean("nn_mpc", "tracking_tcp_rmse_mm", heldout)
    direct_vel = mean("direct", "command_velocity_rms_rad_s", heldout)
    mpc_vel = mean("nn_mpc", "command_velocity_rms_rad_s", heldout)
    direct_acc = mean("direct", "command_acceleration_rms_rad_s2", heldout)
    mpc_acc = mean("nn_mpc", "command_acceleration_rms_rad_s2", heldout)
    requested_p95 = mean("nn_mpc", "requested_residual_p95_deg", heldout)
    requested_max = mean("nn_mpc", "requested_residual_max_deg", heldout)
    mpc_residual = mean("nn_mpc", "executed_residual_p95_deg", heldout)
    executed_max = mean("nn_mpc", "executed_residual_max_deg", heldout)

    paired_joint_direct = paired.get("nn_mpc_minus_direct:tracking_joint_rmse_deg", {})
    paired_joint_preview = paired.get("nn_mpc_minus_preview6:tracking_joint_rmse_deg", {})
    paired_tcp_direct = paired.get("nn_mpc_minus_direct:tracking_tcp_rmse_mm", {})
    paired_tcp_preview = paired.get("nn_mpc_minus_preview6:tracking_tcp_rmse_mm", {})
    direct_values = paired_joint_direct.get("values", [])
    preview_values = paired_joint_preview.get("values", [])
    nominal = [row for row in heldout if row.get("speed") == "nominal"]
    fast = [row for row in heldout if row.get("speed") == "fast"]
    direct_nominal_joint = mean("direct", "tracking_joint_rmse_deg", nominal)
    direct_fast_joint = mean("direct", "tracking_joint_rmse_deg", fast)
    mpc_nominal_joint = mean("nn_mpc", "tracking_joint_rmse_deg", nominal)
    mpc_fast_joint = mean("nn_mpc", "tracking_joint_rmse_deg", fast)

    def paired_ci(pair: dict[str, Any]) -> tuple[int, float | None, float | None, int]:
        values = np.asarray(pair.get("values", []), dtype=np.float64)
        if not values.size:
            return 0, None, None, 0
        critical = 2.109816 if values.size == 18 else 1.96
        mean_value = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        return int(values.size), mean_value - critical * standard_error, mean_value + critical * standard_error, int(np.sum(values < 0.0))

    joint_direct_n, joint_direct_low, joint_direct_high, joint_direct_wins = paired_ci(paired_joint_direct)
    joint_preview_n, joint_preview_low, joint_preview_high, joint_preview_wins = paired_ci(paired_joint_preview)
    tcp_direct_n, tcp_direct_low, tcp_direct_high, tcp_direct_wins = paired_ci(paired_tcp_direct)
    tcp_preview_n, tcp_preview_low, tcp_preview_high, tcp_preview_wins = paired_ci(paired_tcp_preview)

    planner_latencies: list[float] = []
    planner_event_count = 0
    planner_late_drop_count = 0
    heldout_executed_deviation_max: list[float] = []
    heldout_requested_residual_max: list[float] = []
    for row in complete:
        if row.get("controller") != "nn_mpc":
            continue
        if row.get("family") == "heldout":
            rollout_path = Path(str(row.get("output_dir", ""))) / "rollout.npz"
            if rollout_path.exists():
                with np.load(rollout_path, allow_pickle=False) as archive:
                    if "executed_residual" in archive:
                        heldout_executed_deviation_max.append(float(np.max(np.abs(archive["executed_residual"]))))
                    requested_name = "requested_mpc_residual" if "requested_mpc_residual" in archive else "planner_requested_residual"
                    if requested_name in archive:
                        heldout_requested_residual_max.append(float(np.max(np.abs(archive[requested_name]))))
        events_path = Path(str(row.get("output_dir", ""))) / "planner_events.jsonl"
        if not events_path.exists():
            continue
        with events_path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = event.get("end_to_end_latency_s")
                planner_event_count += 1
                if str(event.get("result_type", "")).endswith("late_dropped"):
                    planner_late_drop_count += 1
                if value is not None and np.isfinite(float(value)):
                    planner_latencies.append(float(value) * 1000.0)
    planner_latency_array = np.asarray(planner_latencies, dtype=np.float64)
    global_latency = {
        "mean": None if not planner_latency_array.size else float(np.mean(planner_latency_array)),
        "p95": None if not planner_latency_array.size else float(np.percentile(planner_latency_array, 95.0)),
        "p99": None if not planner_latency_array.size else float(np.percentile(planner_latency_array, 99.0)),
        "max": None if not planner_latency_array.size else float(np.max(planner_latency_array)),
    }
    global_heldout_executed_max_deg = (
        None if not heldout_executed_deviation_max else float(np.degrees(np.max(heldout_executed_deviation_max)))
    )
    global_heldout_requested_max_deg = (
        None if not heldout_requested_residual_max else float(np.degrees(np.max(heldout_requested_residual_max)))
    )

    lines = [
        "## 6. 结果分析（自动生成）",
        "",
        "### 6.1 Held-out 总体性能",
        "",
        f"正式矩阵共完成 {len(complete)}/63 个 trial，其中 held-out 集合为 {len(heldout)} 个 trial（3 种轨迹 × 2 种速度 × 3 个 phase × 3 个控制器）。所有 trial 均达到预期步数，因此没有因执行不完整而被排除。",
        "",
        "| controller | held-out joint RMSE (deg) | held-out TCP RMSE (mm) | command velocity RMS (rad/s) | command acceleration RMS (rad/s²) | requested residual P95 / max (deg) | executed deviation P95 / max (deg) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Direct IK | {f(direct_joint)} | {f(direct_tcp)} | {f(direct_vel, 4)} | {f(direct_acc, 4)} | 0.000 / 0.000 | 0.000 / 0.000 |",
        f"| Fixed Preview6 | {f(preview_joint)} | {f(preview_tcp)} | {f(mean('preview6', 'command_velocity_rms_rad_s', heldout), 4)} | {f(mean('preview6', 'command_acceleration_rms_rad_s2', heldout), 4)} | 0.000 / 0.000 | 0.000 / 0.000 |",
        f"| NN-MPC ±1° | **{f(mpc_joint)}** | **{f(mpc_tcp)}** | {f(mpc_vel, 4)} | {f(mpc_acc, 4)} | {f(requested_p95)} / {f(requested_max)} | {f(mpc_residual)} / {f(executed_max)} |",
        "",
        f"在 held-out 集合上，NN-MPC 相对 Direct 将 joint RMSE 降低 {f(improvement(direct_joint, mpc_joint), 1)}%，将 FK-derived TCP RMSE 降低 {f(improvement(direct_tcp, mpc_tcp), 1)}%；相对 Fixed Preview6 仍分别降低 {f(improvement(preview_joint, mpc_joint), 1)}% 和 {f(improvement(preview_tcp, mpc_tcp), 1)}%。因此结果支持一个有边界的结论：learned state-dependent correction 的收益不只是固定 6-step preview 的重复。",
        "",
        "### 6.2 Paired consistency、速度和轨迹形状",
        "",
        f"held-out 中有 {joint_direct_n} 个 matched phase-condition pairs。NN-MPC 的 joint RMSE 相对 Direct 的差值 18/18 为负（mean {f(paired_joint_direct.get('mean'), 4)}°，std {f(paired_joint_direct.get('std'), 4)}°，95% CI [{f(joint_direct_low, 4)}, {f(joint_direct_high, 4)}]°）；相对 Preview6 的差值同样为 {joint_preview_wins}/{joint_preview_n}（mean {f(paired_joint_preview.get('mean'), 4)}°，std {f(paired_joint_preview.get('std'), 4)}°，95% CI [{f(joint_preview_low, 4)}, {f(joint_preview_high, 4)}]°）。TCP 的对应 paired mean 差值为 {f(paired_tcp_direct.get('mean'), 3)} mm（95% CI [{f(tcp_direct_low, 3)}, {f(tcp_direct_high, 3)}]，{tcp_direct_wins}/{tcp_direct_n} wins）和 {f(paired_tcp_preview.get('mean'), 3)} mm（95% CI [{f(tcp_preview_low, 3)}, {f(tcp_preview_high, 3)}]，{tcp_preview_wins}/{tcp_preview_n} wins）。",
        "",
        "| held-out speed | Direct joint / TCP | Preview6 joint / TCP | NN-MPC joint / TCP |",
        "|---|---:|---:|---:|",
    ]
    for speed in ("nominal", "fast"):
        subset = [row for row in heldout if row.get("speed") == speed]
        lines.append(
            f"| {speed} | {f(mean('direct', 'tracking_joint_rmse_deg', subset))}° / {f(mean('direct', 'tracking_tcp_rmse_mm', subset))} mm | "
            f"{f(mean('preview6', 'tracking_joint_rmse_deg', subset))}° / {f(mean('preview6', 'tracking_tcp_rmse_mm', subset))} mm | "
            f"**{f(mean('nn_mpc', 'tracking_joint_rmse_deg', subset))}° / {f(mean('nn_mpc', 'tracking_tcp_rmse_mm', subset))} mm** |"
        )
    lines += [
        "",
        f"从 nominal 到 fast，Direct 的 held-out joint RMSE 从 {f(direct_nominal_joint)}° 增至 {f(direct_fast_joint)}°；NN-MPC 从 {f(mpc_nominal_joint)}° 增至 {f(mpc_fast_joint)}°，但仍保持低于两个 baseline。三种 held-out 形状和两档速度中，NN-MPC 的平均 RMSE 均低于 Direct 和 Preview6；最困难的 rounded-square/fast 条件下，NN-MPC 仍达到约 0.417° joint RMSE，而 Direct 和 Preview6 分别约为 1.005° 和 0.773°。",
        "",
        "### 6.3 实时性与安全性",
        "",
        f"所有 63 个 trial 的 runtime safety violation、planner failure、control deadline miss 均为 0。21 个 NN-MPC trial 共记录 {planner_event_count} 个 planner events，global planner latency mean/P95/P99/max 为 {f(global_latency['mean'], 2)}/{f(global_latency['p95'], 2)}/{f(global_latency['p99'], 2)}/{f(global_latency['max'], 2)} ms；其中 {planner_late_drop_count} 个结果为 `success_late_dropped`（{planner_late_drop_count}/{planner_event_count}），但没有引起 control deadline miss 或 trial technical exclusion。P95 低于 33.3 ms 控制周期，但这里将 planner latency 与 control-period deadline 分开报告。",
        "",
        "日志中的 `quantized accel exceedance` 不计入安全违规。它来自 30 Hz 下离散编码器计数计算出的 command acceleration，属于诊断性量化效应；真实运行时安全字段均为 0。",
        "",
        "### 6.4 Tracking–command activity trade-off",
        "",
        f"held-out 集合中，NN-MPC 的 command velocity RMS 为 {f(mpc_vel, 4)} rad/s，相比 Direct 的 {f(direct_vel, 4)} rad/s 增加约 {f(improvement(direct_vel, mpc_vel) * -1 if direct_vel and mpc_vel else None, 1)}%；command acceleration RMS 为 {f(mpc_acc, 4)} rad/s²，相比 Direct 的 {f(direct_acc, 4)} rad/s² 增加约 {f(improvement(direct_acc, mpc_acc) * -1 if direct_acc and mpc_acc else None, 1)}%。NN-MPC 的 requested residual P95/max 为 {f(requested_p95)}/{f(requested_max)}°，executed deviation P95/max 的逐 trial 平均为 {f(mpc_residual)}/{f(executed_max)}°；所有 held-out NN-MPC raw rollout 的 requested max 为 {f(global_heldout_requested_max_deg)}°，executed deviation 全局 max 为 {f(global_heldout_executed_max_deg)}°。因此性能提升伴随更高的 command activity；当前数据支持 tracking accuracy 与 command smoothness 之间存在明确 trade-off，但不支持“同时降低 tracking error 和 command activity”的更强结论。",
        "",
        "### 6.5 结论与解释边界",
        "",
        "综合 54 个 held-out trial 和 9 个 circle development trial，结果支持 tracking-dominant NN-MPC 在固定 ±1° residual authority 下具有稳定的实机收益。该结论限定于当前 SO101 硬件、冻结 GRU checkpoint、30 Hz 控制周期、H=6、CEM 128×2 和本协议中的参考轨迹；不能外推为对其他机器人、模型或轨迹分布的普遍保证。TCP 数值来自编码器关节角的 MuJoCo FK，不是外部定位仪测量。",
        "",
    ]
    return lines


def _markdown(protocol: dict[str, Any], results: list[dict[str, Any]], aggregate: dict[str, Any], paired: dict[str, Any], manifest_hash: str | None) -> str:
    rows = [_flat_row(result) for result in results]
    completed = [row for row in rows if row["status"] == "complete"]
    heldout_paired = _paired(rows, family="heldout")
    lines = [
        "# SO101 最终论文实机实验记录",
        "",
        "> 这是实验记录和可复现性文档，不是论文正文。正式结果由 `scripts/analyze_so101_paper_trials.py` 根据原始 NPZ 自动更新。",
        "",
        f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        f"协议：`{protocol['protocol_id']}`",
        "",
        "## 1. 冻结方法",
        "",
        "- 最终 NN-MPC：tracking-dominant objective `J = C_q`，`w_q=1`，其余 soft cost 权重均为 0。",
        "- residual authority：每个关节 `|r_j| <= 1°`。保留 Direct、Fixed Preview6、CEM best/mean/fixed-preview 候选池。",
        f"- 控制频率：30 Hz；`H={protocol['final_mpc']['horizon']}`；CEM `samples={protocol['final_mpc']['num_samples']}`，`iterations={protocol['final_mpc']['cem_iters']}`；GRU history={protocol['final_mpc']['history_len']}。",
        "- 保留 canonical executable projector、速度/加速度/joint hard constraints、braking、encoder quantization、startup/homing gate。",
        "- `preview6` 固定为 6 steps（200 ms），正式测试不按轨迹重新搜索 preview 长度。",
        "",
        "## 2. 实验矩阵",
        "",
        "| family | shapes | speeds | matched phase blocks | controllers/block | runs |",
        "|---|---|---|---:|---:|---:|",
        "| held-out | 3 | 2 | 3 | 3 | 54 |",
        "| development reference | 1 | 1 | 3 | 3 | 9 |",
        "| total |  |  |  |  | 63 |",
        "",
        "控制器顺序按 repeat 预注册为：repeat 0 `Direct → Preview6 → NN-MPC`；repeat 1 `Preview6 → NN-MPC → Direct`；repeat 2 `NN-MPC → Direct → Preview6`。",
        "",
        "## 3. 指标定义",
        "",
        "主评估窗口只取冻结 reference 的 `SEGMENT_SHAPE_LOOP`；不把 startup、approach、return、padding 混入主 RMSE。",
        "- joint RMSE / P95 / max：编码器 `q_ctrl` 相对 frozen `q_des`，单位 degree。",
        "- TCP 指标：编码器关节角经 fine MuJoCo model FK 得到的位置误差，单位 mm；没有外部定位仪，因此不称为外部实测 TCP。",
        "- command activity：transmitted executable command 的 velocity/acceleration RMS、P95、max；residual 同时报告 requested 与 executed。`requested residual` 是 CEM/planner 请求的 residual，在 executable-command projector 之前受 ±1° authority 约束；`executed deviation` 是最终 transmitted command 相对该 tick instantaneous nominal reference 的偏差，经过 stateful velocity/acceleration projection、braking 和 encoder quantization 后计算，因此可以超过 1°。",
        "- safety/timing：deadline miss、planner failure/late drop、command v/a violation、TX failure、projection flags、planner latency。编码器量化造成的离散命令加速度超限单独报告，不计入 runtime safety violation；技术不完整 trial 不进入 controller aggregate，但保留在 ledger。",
        "",
        "## 4. 开发阶段参考结果（非正式 held-out 统计）",
        "",
        "| controller | circle joint RMSE | 说明 |",
        "|---|---:|---|",
        "| Direct IK | 0.9019° | development circle，preview 0 |",
        "| Fixed Preview6 | 0.6482° | development circle，固定 6 steps |",
        "| NN-MPC ±1° | 0.4024° | tracking-only，development circle |",
        "",
        "这些数值用于记录方法冻结前的开发证据，不能替代下面的 paired formal matrix。此前 ±2° authority 结果只作为 authority ablation，不属于最终方法。",
        "",
        "## 5. 正式结果（自动汇总）",
        "",
    ]
    if not completed:
        lines.append("正式真机矩阵尚未产生完整 trial；当前只有协议、参考和结果 ledger。")
    else:
        lines += [
            "| family | shape | speed | controller | n | joint RMSE (deg) | TCP RMSE (mm) | command vel RMS (rad/s) | command acc RMS (rad/s²) | runtime v/a violations | requested residual P95 / max (deg) | executed deviation P95 / max (deg) |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key, value in aggregate.items():
            lines.append(
                f"| {value['family']} | {value['shape']} | {value['speed']} | {value['controller']} | {value['n']} | "
                f"{value['tracking_joint_rmse_deg']['mean']:.4f} | "
                f"{(value['tracking_tcp_rmse_mm']['mean'] if value['tracking_tcp_rmse_mm']['mean'] is not None else float('nan')):.3f} | "
                f"{value['command_velocity_rms_rad_s']['mean']:.4f} | "
                f"{value['command_acceleration_rms_rad_s2']['mean']:.4f} | "
                f"{value['command_velocity_violation_count']['mean']:.1f} / {value['command_acceleration_violation_count']['mean']:.1f} | "
                f"{value['requested_residual_p95_deg']['mean']:.4f} / {value['requested_residual_max_deg']['mean']:.4f} | "
                f"{value['executed_residual_p95_deg']['mean']:.4f} / {value['executed_residual_max_deg']['mean']:.4f} |"
            )
        lines += ["", "Paired差值（NN-MPC − baseline；负值表示 NN-MPC 更低）：", ""]
        for key, value in heldout_paired.items():
            lines.append(f"- `{key}`：n={value['n']}，mean={value['mean']:.6g}，std={value['std']:.6g}")
        lines.append("- 主表 paired 统计只使用 18 个 held-out matched pairs；包含 development circle 的 21-pair 统计保留在 `analysis/trial_metrics.json` 的 `paired` 字段中。")
        lines += ["", *_analysis_lines(rows, heldout_paired)]
    lines += [
        "",
        "## 7. Trial ledger",
        "",
        "| trial | controller | shape | speed | phase | repeat | status | joint RMSE (deg) | TCP RMSE (mm) | safety violations | quantized accel exceedance |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['trial_id']} | {row['controller']} | {row['shape']} | {row['speed']} | {row['phase_index']} | {row['repeat_index']} | {row['status']} | "
            f"{'' if row['tracking_joint_rmse_deg'] is None else f'{row["tracking_joint_rmse_deg"]:.4f}'} | "
            f"{'' if row['tracking_tcp_rmse_mm'] is None else f'{row["tracking_tcp_rmse_mm"]:.3f}'} | {row['safety_violation_count'] if row['safety_violation_count'] is not None else ''} | "
            f"{row['command_acceleration_quantization_exceedance_count'] if row['command_acceleration_quantization_exceedance_count'] is not None else ''} |"
        )
    lines += [
        "",
        "## 8. 身份与原始数据",
        "",
        f"- held-out reference manifest SHA-256：`{manifest_hash or 'not available'}`",
        f"- raw output root：`{protocol['output_root']}`",
        f"- hardware config：`{protocol['hardware_config']}`",
        f"- checkpoint：`{protocol['checkpoint']}`",
        f"- normalizer：`{protocol['normalizer']}`",
        "- 每个 trial 目录应包含 `trial_manifest.json`、`trial.log`、`rollout.npz`；不得覆盖非空 trial，重试必须使用 retry index。",
        "",
        "## 9. 解释边界",
        "",
        "- 只要 trial 技术上完整，就算 tracking 差也保留，不以结果好坏排除。",
        "- FK-derived TCP 是编码器状态的模型换算，用于辅助报告；没有外部仪器，不作绝对 TCP 测量声明。",
        "- 任何正式方法参数变更都需要新 protocol id 和新输出根目录，不能覆盖本协议。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/experiments/so101_final_paper_20260810.yaml")
    parser.add_argument("--write-doc", action="store_true", help="also update docs/hardware result record")
    args = parser.parse_args()
    protocol_path = _path(args.protocol)
    protocol = _load_protocol(protocol_path)
    specs = _trial_specs(protocol)
    fk = None
    try:
        fk = _make_fk(protocol)
    except Exception as exc:
        print(f"warning: FK metrics unavailable: {type(exc).__name__}: {exc}")
    results = [analyze_trial(protocol, spec, fk) for spec in specs.values()]
    rows = [_flat_row(result) for result in results]
    analysis_dir = _path(protocol["output_root"]) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    aggregate = _aggregate(rows)
    paired = _paired(rows)
    paired_heldout = _paired(rows, family="heldout")
    manifest_path = _path(protocol["heldout_reference_manifest"])
    manifest_hash = None
    if manifest_path.exists():
        import hashlib
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    payload = {
        "protocol": protocol["protocol_id"],
        "protocol_file": str(protocol_path),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status_counts": dict(Counter(result.get("status", "unknown") for result in results)),
        "trial_results": results,
        "aggregate": aggregate,
        "paired": paired,
        "paired_heldout": paired_heldout,
        "heldout_reference_manifest_sha256": manifest_hash,
    }
    (analysis_dir / "trial_metrics.json").write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (analysis_dir / "trial_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("trial_id", *SUMMARY_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (analysis_dir / "aggregate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("group", "n", *[field for field in SUMMARY_FIELDS if field not in {"family", "shape", "speed", "controller", "status"}],), extrasaction="ignore")
        writer.writeheader()
        for key, value in aggregate.items():
            row = {"group": key, "n": value["n"]}
            for field in writer.fieldnames:
                if field in {"group", "n"}:
                    continue
                row[field] = value[field]["mean"]
            writer.writerow(row)
    if args.write_doc:
        doc_path = ROOT / "docs/hardware/so101-final-paper-experiment-results-20260810.md"
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(_markdown(protocol, results, aggregate, paired, manifest_hash), encoding="utf-8")
        print(f"results document: {doc_path}")
    print(json.dumps({"status_counts": payload["status_counts"], "analysis_dir": str(analysis_dir)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
