from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _percentile(values: Any, q: float) -> float:
    finite = _finite(values)
    return float(np.percentile(finite, q)) if finite.size else float("nan")


def _rms(values: Any) -> float:
    finite = _finite(values)
    return float(np.sqrt(np.mean(np.square(finite)))) if finite.size else float("nan")


def _max(values: Any, *, absolute: bool = False) -> float:
    finite = _finite(values)
    if absolute:
        finite = np.abs(finite)
    return float(np.max(finite)) if finite.size else float("nan")


def _longest_true_run(values: Any) -> int:
    array = np.asarray(values, dtype=bool).reshape(-1)
    longest = current = 0
    for value in array:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _event_metric(events: list[dict[str, Any]], key: str) -> np.ndarray:
    return _finite([event.get("candidate_diagnostics", {}).get(key, np.nan) for event in events])


def summarize_arrays(label: str, arrays: dict[str, np.ndarray], planner_events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    actual = np.asarray(arrays.get("actual_states", np.empty((0, 0))), dtype=np.float64)
    desired = np.asarray(arrays.get("q_des", np.empty((0, 0))), dtype=np.float64)
    n = min(len(actual), len(desired)) if actual.ndim == desired.ndim == 2 else 0
    joint_error = actual[:n, : desired.shape[1]] - desired[:n] if n and actual.shape[1] >= desired.shape[1] else np.empty(0)
    tcp = _finite(arrays.get("ee_position_errors", np.empty(0)))
    tcp_all = np.asarray(arrays.get("ee_position_errors", np.empty(0)), dtype=np.float64)
    lap_ids = np.asarray(arrays.get("lap_ids", np.empty(0)), dtype=np.int64)
    lap_tcp = tcp_all[lap_ids >= 0] if tcp_all.shape == lap_ids.shape else np.empty(0)
    planning = np.asarray(arrays.get("planning_time", np.empty(0)), dtype=np.float64)
    replanned = np.asarray(arrays.get("mpc_replanned", np.empty(0)), dtype=bool)
    solves = planning[replanned] if replanned.shape == planning.shape else planning
    packet_age = np.asarray(arrays.get("packet_age", np.empty(0)), dtype=np.float64)
    fallback = np.asarray(arrays.get("fallback_active", packet_age < 0), dtype=bool)
    scalar = lambda key, default="": np.asarray(arrays.get(key, default)).reshape(-1)[0]
    solve_count = int(np.asarray(arrays.get("planner_solve_count", np.sum(replanned))).reshape(-1)[0])
    late_count = int(np.asarray(arrays.get("planner_late_drop_count", 0)).reshape(-1)[0])
    failure_flags = np.asarray(arrays.get("failure_flags", np.empty(0))) != 0
    planner_failure_events = np.asarray(arrays.get("planner_failure_event", np.empty(0))) != 0
    planner_failure_count = int(scalar("planner_failure_count", np.sum(planner_failure_events)))
    any_planner_failure = planner_failure_count > 0 or bool(np.any(failure_flags))
    events = planner_events or []
    exact_pool_time = _event_metric(events, "exact_final_pool_time_s")
    exact_count = _event_metric(events, "exact_validation_count")
    exact_valid = _event_metric(events, "exact_valid_count")
    exact_changed = _event_metric(events, "exact_selection_changed")
    baseline_selected = np.asarray([str(event.get("selection_mode", "")) == "baseline" for event in events], dtype=bool)
    return {
        "label": label,
        "delay_protocol": str(scalar("delay_protocol", "not_applicable")),
        "multirate_mode": str(scalar("multirate_mode", "synchronous")),
        "delay_steps": int(scalar("anticipation_delay_steps", 0)),
        "steps": int(len(np.asarray(arrays.get("actuator_q_ref", np.empty(0))))),
        "tcp_rmse_m": float(np.sqrt(np.mean(np.square(tcp)))) if tcp.size else float("nan"),
        "lap_tcp_rmse_m": _rms(lap_tcp),
        "tcp_p95_m": float(np.percentile(tcp, 95)) if tcp.size else float("nan"),
        "orientation_rmse_rad": _rms(arrays.get("ee_orientation_errors", np.empty(0))),
        "joint_rmse_rad": _rms(joint_error),
        # A row is one trajectory-seed case.  Keep the paper failure metric
        # binary and expose transient planner fallbacks separately; 10 ms
        # ticks are not independent trials.
        "failure_rate": float(any_planner_failure),
        "planner_failure_step_rate": float(np.sum(planner_failure_events) / max(len(np.asarray(arrays.get("actuator_q_ref", np.empty(0)))), 1)) if planner_failure_events.size else float(np.mean(failure_flags)) if failure_flags.size else 0.0,
        "command_acceleration_rms_rad_s2": _rms(arrays.get("command_acceleration", np.empty(0))),
        "command_acceleration_max_abs_rad_s2": float(np.max(np.abs(np.asarray(arrays.get("command_acceleration", np.empty(0)))))) if np.asarray(arrays.get("command_acceleration", np.empty(0))).size else float("nan"),
        "actuator_torque_rms_nm": _rms(arrays.get("tau_actuator", np.empty(0))),
        "total_torque_rms_nm": _rms(arrays.get("tau_total", np.empty(0))),
        "residual_rms_rad": _rms(arrays.get("executed_residual", np.empty(0))),
        "feedback_rms_rad": _rms(arrays.get("feedback", np.empty(0))),
        "projection_discrepancy_rms_rad": _rms(arrays.get("projection_discrepancy", np.empty(0))),
        "projection_discrepancy_p95_rad": _percentile(np.abs(np.asarray(arrays.get("projection_discrepancy", np.empty(0)))), 95),
        "projection_discrepancy_max_rad": _max(arrays.get("projection_discrepancy", np.empty(0)), absolute=True),
        "safety_projection_offset_rms_rad": _rms(arrays.get("safety_projection_offset", np.empty(0))),
        "safety_projection_offset_p95_rad": _percentile(np.abs(np.asarray(arrays.get("safety_projection_offset", np.empty(0)))), 95),
        "safety_projection_offset_max_rad": _max(arrays.get("safety_projection_offset", np.empty(0)), absolute=True),
        "planner_execution_qref_error_rms_rad": _rms(arrays.get("planner_execution_qref_error", np.empty(0))),
        "planner_execution_qref_error_p95_rad": _percentile(np.abs(np.asarray(arrays.get("planner_execution_qref_error", np.empty(0)))), 95),
        "planner_execution_qref_error_max_rad": _max(arrays.get("planner_execution_qref_error", np.empty(0)), absolute=True),
        "projection_activation_rate": float(np.mean(np.asarray(arrays.get("projection_active", np.empty(0))) != 0)) if np.asarray(arrays.get("projection_active", np.empty(0))).size else float("nan"),
        "residual_saturation_rate": float(np.mean(np.asarray(arrays.get("residual_saturated", np.empty(0))) != 0)) if np.asarray(arrays.get("residual_saturated", np.empty(0))).size else float("nan"),
        "feedback_saturation_rate": float(np.mean(np.asarray(arrays.get("feedback_saturated", np.empty(0))) != 0)) if np.asarray(arrays.get("feedback_saturated", np.empty(0))).size else float("nan"),
        "joint_violation_count": int(np.sum(np.asarray(arrays.get("joint_limit_violation_flags", np.empty(0))) != 0)),
        "velocity_violation_count": int(np.sum(np.asarray(arrays.get("command_velocity_violation_flags", np.empty(0))) != 0)),
        "acceleration_violation_count": int(np.sum(np.asarray(arrays.get("command_acceleration_violation_flags", np.empty(0))) != 0)),
        "solve_p50_s": _percentile(solves, 50), "solve_p95_s": _percentile(solves, 95), "solve_p99_s": _percentile(solves, 99),
        "solve_max_s": _max(solves),
        "e2e_p50_s": _percentile(arrays.get("planner_end_to_end_latency_s", np.empty(0)), 50),
        "e2e_p95_s": _percentile(arrays.get("planner_end_to_end_latency_s", np.empty(0)), 95),
        "e2e_p99_s": _percentile(arrays.get("planner_end_to_end_latency_s", np.empty(0)), 99),
        "e2e_max_s": _max(arrays.get("planner_end_to_end_latency_s", np.empty(0))),
        "planner_hz": float(scalar("planner_actual_update_rate_hz", np.nan)),
        "late_packet_rate": float(late_count / solve_count) if solve_count else float("nan"),
        "active_packet_ratio": float(np.mean(packet_age >= 0)) if packet_age.size else float("nan"),
        "packet_age_p50_steps": _percentile(packet_age[packet_age >= 0], 50),
        "packet_age_p95_steps": _percentile(packet_age[packet_age >= 0], 95),
        "packet_age_max_steps": _max(packet_age[packet_age >= 0]),
        "fallback_duty_cycle": float(np.mean(fallback)) if fallback.size else float("nan"),
        "active_packet_gap_max_steps": _longest_true_run(fallback),
        "active_packet_gap_max_s": 0.01 * _longest_true_run(fallback),
        "control_compute_p50_s": _percentile(arrays.get("control_step_wall_time", np.empty(0)), 50),
        "control_compute_p95_s": _percentile(arrays.get("control_step_wall_time", np.empty(0)), 95),
        "control_compute_p99_s": _percentile(arrays.get("control_step_wall_time", np.empty(0)), 99),
        "control_compute_max_s": _max(arrays.get("control_step_wall_time", np.empty(0))),
        "control_period_p50_s": _percentile(arrays.get("actual_control_period_s", np.empty(0)), 50),
        "control_period_p95_s": _percentile(arrays.get("actual_control_period_s", np.empty(0)), 95),
        "control_period_p99_s": _percentile(arrays.get("actual_control_period_s", np.empty(0)), 99),
        "control_period_max_s": _max(arrays.get("actual_control_period_s", np.empty(0))),
        "wakeup_lateness_p50_s": _percentile(arrays.get("control_wakeup_lateness_s", np.empty(0)), 50),
        "wakeup_lateness_p95_s": _percentile(arrays.get("control_wakeup_lateness_s", np.empty(0)), 95),
        "wakeup_lateness_p99_s": _percentile(arrays.get("control_wakeup_lateness_s", np.empty(0)), 99),
        "wakeup_lateness_max_s": _max(arrays.get("control_wakeup_lateness_s", np.empty(0))),
        "start_jitter_p50_s": _percentile(arrays.get("control_start_jitter_s", np.empty(0)), 50),
        "start_jitter_p95_s": _percentile(arrays.get("control_start_jitter_s", np.empty(0)), 95),
        "start_jitter_p99_s": _percentile(arrays.get("control_start_jitter_s", np.empty(0)), 99),
        "start_jitter_max_s": _max(arrays.get("control_start_jitter_s", np.empty(0))),
        "control_deadline_miss_count": int(np.sum(np.asarray(arrays.get("control_deadline_miss", np.empty(0))) != 0)),
        "planner_failure_count": planner_failure_count,
        "packet_expiration_count": int(scalar("packet_expiration_count", 0)),
        # Plan-level quantities: one record per CEM solve, never per execution
        # tick.  They expose what the final-pool reranking actually changed.
        "planner_event_count": len(events),
        "exact_final_pool_time_p50_s": _percentile(exact_pool_time, 50),
        "exact_final_pool_time_p95_s": _percentile(exact_pool_time, 95),
        "exact_final_pool_time_max_s": _max(exact_pool_time),
        "exact_pool_candidate_count_mean": float(np.mean(exact_count)) if exact_count.size else float("nan"),
        "exact_pool_valid_count_mean": float(np.mean(exact_valid)) if exact_valid.size else float("nan"),
        "exact_selection_changed_rate": float(np.mean(exact_changed != 0)) if exact_changed.size else float("nan"),
        "baseline_selection_rate": float(np.mean(baseline_selected)) if baseline_selected.size else float("nan"),
        "control_semantics_version": int(scalar("control_semantics_version", 0)),
        "projection_semantics_version": int(scalar("projection_semantics_version", 0)),
        "residual_cost_semantics": str(scalar("residual_cost_semantics", "not_applicable")),
        "packet_residual_semantics": str(scalar("packet_residual_semantics", "not_applicable")),
        "residual_feasibility_semantics": str(scalar("residual_feasibility_semantics", "not_applicable")),
        "nominal_command_semantics": str(scalar("nominal_command_semantics", "not_applicable")),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)
    metrics = (
        "tcp_rmse_m", "lap_tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad", "joint_rmse_rad",
        "failure_rate", "planner_failure_step_rate",
        "command_acceleration_rms_rad_s2", "command_acceleration_max_abs_rad_s2", "actuator_torque_rms_nm",
        "projection_discrepancy_rms_rad", "safety_projection_offset_rms_rad",
        "projection_discrepancy_p95_rad", "projection_discrepancy_max_rad",
        "safety_projection_offset_p95_rad", "safety_projection_offset_max_rad",
        "planner_execution_qref_error_rms_rad", "planner_execution_qref_error_p95_rad",
        "planner_execution_qref_error_max_rad", "projection_activation_rate",
        "residual_saturation_rate", "feedback_saturation_rate", "e2e_p95_s",
        "solve_p95_s", "solve_p99_s", "solve_max_s", "e2e_p99_s", "e2e_max_s",
        "planner_hz", "late_packet_rate", "fallback_duty_cycle",
        "active_packet_gap_max_s", "control_deadline_miss_count",
        "control_compute_p50_s", "control_compute_p95_s", "control_compute_max_s",
        "control_period_p50_s", "control_period_p95_s", "control_period_p99_s",
        "control_period_max_s", "wakeup_lateness_p50_s", "wakeup_lateness_p95_s",
        "wakeup_lateness_p99_s", "wakeup_lateness_max_s",
        "start_jitter_p50_s", "start_jitter_p95_s", "start_jitter_p99_s",
        "start_jitter_max_s", "packet_age_p50_steps", "packet_age_p95_steps",
        "packet_age_max_steps",
        "planner_failure_count", "packet_expiration_count",
        "exact_final_pool_time_p95_s", "exact_final_pool_time_max_s",
        "exact_pool_candidate_count_mean", "exact_pool_valid_count_mean",
        "exact_selection_changed_rate", "baseline_selection_rate",
    )
    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        aggregate = {field: value for field, value in zip(group_fields, key)}
        aggregate["n_cases"] = len(members)
        for metric in metrics:
            values = np.asarray([float(member.get(metric, np.nan)) for member in members], dtype=np.float64)
            finite = values[np.isfinite(values)]
            aggregate[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            aggregate[f"{metric}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0 if finite.size == 1 else float("nan")
        output.append(aggregate)
    return output


def latency_recovery(rows: list[dict[str, Any]], epsilon_m: float = 1e-6) -> dict[str, Any]:
    grouped: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        key = (str(row.get("trajectory")), int(row.get("seed", -1)))
        grouped.setdefault(key, {})[str(row["label"])] = float(row["tcp_rmse_m"])
    values: list[float] = []
    for methods in grouped.values():
        if {"NaiveDelayed", "FullVirtual", "IdealZeroDelay"}.issubset(methods):
            denominator = methods["NaiveDelayed"] - methods["IdealZeroDelay"]
            if denominator > epsilon_m:
                values.append((methods["NaiveDelayed"] - methods["FullVirtual"]) / denominator)
    return {"epsilon_m": epsilon_m, "n": len(values), "mean": float(np.mean(values)) if values else float("nan"), "values": values}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True, default=str) + "\n", encoding="utf-8")
