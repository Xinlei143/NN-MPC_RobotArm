#!/usr/bin/env python3
"""Offline cost decomposition for the SO101 real-MPC candidate set.

This script is intentionally diagnostic-only.  It never opens a hardware
backend, changes controller weights, trains a model, or publishes a command.
For a frozen active rollout it reconstructs measured state/history anchors and
re-evaluates, under one identical learned model and executable projector:

    baseline, preview_3, preview_6, CEM_best, CEM_mean, CEM_selected

The first three candidates are deterministic residual candidates.  The three
CEM roles are replayed from the same CEM distribution.  ``CEM_selected`` is
the final candidate selected by the offline CEM replay; the historical
planner's selection modes are retained as metadata when planner_events.jsonl
is present, but the old real logs do not contain the historical best/mean
sequences needed to score them exactly.

The output contains per-anchor rows, candidate mean/median/P95 summaries, and
argmin frequencies for q_tracking and total weighted cost.  It also performs
an offline ``w_dq_limit`` re-scoring sweep without rerunning CEM, and reports
predicted/reference/logged velocity magnitudes.  Cost terms keep the
repository convention: component names are the unweighted normalized terms
returned by ``joint_space_tracking_cost`` and ``total`` is the weighted sum.
``weighted_*`` columns make the additive total auditable.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "dynamics_modeling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamics_modeling.neural_dynamics.rollout import load_dynamics_bundle
from mpc.analytical_candidates import build_preview_residual_candidates
from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.cost_functions import JointSpaceCostConfig
from mpc.executable_rollout import ExecutableRolloutEngine
from mpc.history import history_tokens
from mpc.planner_rollout import LearnedDynamicsPlanner, PlannerRolloutConfig
from mpc.robot_config import load_robot_spec
from robot_runtime.config import load_hardware_config
from robot_runtime.executable_command import (
    ExecutableCommandState,
    make_executable_command_spec,
)


CANDIDATES = ("baseline", "preview_3", "preview_6", "CEM_best", "CEM_mean", "CEM_selected")
TERM_NAMES = (
    "q_tracking",
    "dq_tracking",
    "residual",
    "servo",
    "residual_velocity",
    "residual_acceleration",
    "first",
    "joint_limit",
    "dq_limit",
    "total",
)
WEIGHT_NAMES = {
    "q_tracking": "w_q",
    "dq_tracking": "w_dq",
    "residual": "w_residual",
    "servo": "w_servo",
    "residual_velocity": "w_residual_velocity",
    "residual_acceleration": "w_residual_acceleration",
    "first": "w_first",
    "joint_limit": "w_joint_limit",
    "dq_limit": "w_dq_limit",
}
REGULARIZATION_TERMS = ("residual", "servo", "residual_velocity", "residual_acceleration", "first")
DEFAULT_DQ_LIMIT_SWEEP = (5.0, 2.0, 1.0, 0.5, 0.1, 0.0)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _vector(value: str | None, default: np.ndarray, name: str) -> np.ndarray:
    if value is None:
        result = np.asarray(default, dtype=np.float32).copy()
    else:
        try:
            parsed = np.asarray([float(item.strip()) for item in str(value).split(",")], dtype=np.float32)
        except ValueError as exc:
            raise ValueError(f"{name} must be a scalar or comma-separated finite values") from exc
        if parsed.size == 1:
            result = np.full(default.shape, float(parsed[0]), dtype=np.float32)
        else:
            result = parsed
    if result.shape != default.shape or not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain {default.size} finite positive values")
    return result.astype(np.float32)


def _nonnegative_values(value: str | None, default: tuple[float, ...], name: str) -> list[float]:
    if value is None:
        parsed = list(default)
    else:
        try:
            parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
        except ValueError as exc:
            raise ValueError(f"{name} must be comma-separated finite non-negative values") from exc
    if not parsed or not np.all(np.isfinite(parsed)) or any(item < 0.0 for item in parsed):
        raise ValueError(f"{name} must contain at least one finite non-negative value")
    # Keep the user's order while removing accidental duplicates.
    return list(dict.fromkeys(float(item) for item in parsed))


def _reference_derivatives(q_des: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    dq = np.empty_like(q_des, dtype=np.float32)
    dq[:-1] = np.diff(q_des, axis=0) / float(dt)
    dq[-1] = dq[-2] if len(dq) > 1 else 0.0
    ddq = np.empty_like(q_des, dtype=np.float32)
    ddq[:-1] = np.diff(dq, axis=0) / float(dt)
    ddq[-1] = ddq[-2] if len(ddq) > 1 else 0.0
    return dq, ddq


def _reference_calibration(
    q_des: np.ndarray,
    dq_des: np.ndarray,
    ddq_des: np.ndarray,
    physical_velocity_limit: np.ndarray,
    physical_acceleration_limit: np.ndarray,
) -> dict[str, np.ndarray]:
    q_scale = np.clip(
        0.1 * (np.percentile(q_des, 95.0, axis=0) - np.percentile(q_des, 5.0, axis=0)),
        0.04,
        0.08,
    )
    dq_scale = np.maximum(np.percentile(np.abs(dq_des), 99.0, axis=0), 0.25)
    q_ref_velocity_limit = np.clip(
        3.0 * np.percentile(np.abs(dq_des), 99.0, axis=0),
        0.05 * physical_velocity_limit,
        physical_velocity_limit,
    )
    q_ref_acceleration_limit = np.clip(
        3.0 * np.percentile(np.abs(ddq_des), 99.0, axis=0),
        0.05 * physical_acceleration_limit,
        physical_acceleration_limit,
    )
    return {
        "q_tracking_scale": q_scale.astype(np.float32),
        "dq_tracking_scale": dq_scale.astype(np.float32),
        "q_ref_velocity_limit": q_ref_velocity_limit.astype(np.float32),
        "q_ref_acceleration_limit": q_ref_acceleration_limit.astype(np.float32),
    }


def _make_executable_spec(hardware: Any):
    calibration_payload = json.loads(Path(hardware.calibration_path).read_text(encoding="utf-8"))
    calibration_low = np.asarray(
        [calibration_payload[name]["range_min"] for name in hardware.joint_names], dtype=np.float64
    )
    calibration_high = np.asarray(
        [calibration_payload[name]["range_max"] for name in hardware.joint_names], dtype=np.float64
    )
    joint_low = hardware.hardware_joint_low if hardware.hardware_joint_low is not None else hardware.joint_low
    joint_high = hardware.hardware_joint_high if hardware.hardware_joint_high is not None else hardware.joint_high
    return make_executable_command_spec(
        joint_low=joint_low,
        joint_high=joint_high,
        velocity_limit=hardware.command_velocity_limit,
        acceleration_limit=hardware.command_acceleration_limit,
        relative_limit=hardware.relative_target_limit,
        raw_low=hardware.raw_low[:5],
        raw_high=hardware.raw_high[:5],
        calibration_low=calibration_low,
        calibration_high=calibration_high,
        control_dt=hardware.control_dt,
    )


def _load_rollout(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"q_des"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"{path} is missing {sorted(missing)}")
        state_key = "observed_states" if "observed_states" in archive.files else "actual_states"
        command_key = "actuator_q_ref" if "actuator_q_ref" in archive.files else "transmitted_q_ref"
        if state_key not in archive.files or command_key not in archive.files:
            raise KeyError(f"{path} needs observed/actual states and actuator/transmitted commands")
        result = {
            "states": np.asarray(archive[state_key], dtype=np.float32)[:, :10],
            "commands": np.asarray(archive[command_key], dtype=np.float32),
            "q_des": np.asarray(archive["q_des"], dtype=np.float32),
        }
        if "planner_requested_residual" in archive.files:
            result["planner_requested_residual"] = np.asarray(
                archive["planner_requested_residual"], dtype=np.float32
            )
        if "active_start_tick" in archive.files:
            result["active_start_tick"] = np.asarray(archive["active_start_tick"], dtype=np.int64)
    n = len(result["states"])
    if result["states"].shape != (n, 10) or result["commands"].shape != (n, 5) or result["q_des"].shape != (n, 5):
        raise ValueError(
            f"rollout arrays must be states [{n},10], commands/q_des [{n},5]; "
            f"got {result['states'].shape}, {result['commands'].shape}, {result['q_des'].shape}"
        )
    return result


def _load_logged_modes(rollout: Path) -> list[str]:
    events_path = rollout.parent / "planner_events.jsonl"
    if not events_path.is_file():
        return []
    modes: list[str] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        mode = event.get("selection_mode")
        if isinstance(mode, str) and mode:
            modes.append(mode)
    return modes


def _make_history(states: np.ndarray, commands: np.ndarray, anchor: int, history_len: int) -> torch.Tensor:
    # This follows the repository's [x_t, u_t] convention.  The newest action
    # is a placeholder and is overwritten by the first candidate action in the
    # learned rollout, exactly as rollout_dynamics_batch does.
    start = max(0, int(anchor) - int(history_len) + 1)
    history = history_tokens(states[start : anchor + 1], commands[start : anchor + 1], history_len)
    return torch.as_tensor(history, dtype=torch.float32)


def _previous_command_state(commands: np.ndarray, states: np.ndarray, anchor: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if anchor <= 0:
        previous = states[anchor, :5].astype(np.float32)
        return previous, np.zeros_like(previous)
    previous = commands[anchor - 1].astype(np.float32)
    if anchor <= 1:
        return previous, np.zeros_like(previous)
    velocity = ((commands[anchor - 1] - commands[anchor - 2]) / float(dt)).astype(np.float32)
    return previous, velocity


def _previous_residual(
    residual_trace: np.ndarray | None,
    anchor: int,
    dt: float,
    n_joints: int,
) -> tuple[np.ndarray, np.ndarray]:
    zeros = np.zeros(n_joints, dtype=np.float32)
    if residual_trace is None or anchor <= 0 or residual_trace.shape != (len(residual_trace), n_joints):
        return zeros, zeros
    previous = residual_trace[anchor - 1].astype(np.float32)
    if anchor <= 1:
        return previous, zeros
    velocity = ((residual_trace[anchor - 1] - residual_trace[anchor - 2]) / float(dt)).astype(np.float32)
    return previous, velocity


def _extract_terms(evaluation: dict[str, torch.Tensor], index: int = 0) -> dict[str, float]:
    terms = evaluation.get("cost_terms", {})
    result: dict[str, float] = {}
    for name in TERM_NAMES:
        value = terms.get(name)
        if isinstance(value, torch.Tensor) and value.ndim == 1 and value.shape[0] > index:
            result[name] = float(value[index].detach().cpu())
        else:
            result[name] = float("nan")
    return result


def _evaluate_action(
    planner: LearnedDynamicsPlanner,
    action: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    candidate = torch.as_tensor(action, dtype=torch.float32, device=planner.q_des.device).unsqueeze(0)
    evaluation = planner.evaluate_exact(candidate)
    terms = _extract_terms(evaluation)
    pred_states = evaluation["pred_states"][0].detach().cpu().numpy()
    n_joints = pred_states.shape[-1] // 2
    requested_residual = evaluation["requested_residual_sequences"][0].detach().cpu().numpy()
    return (
        terms,
        np.asarray(pred_states[1:, n_joints : 2 * n_joints], dtype=np.float32),
        np.asarray(requested_residual, dtype=np.float32),
    )


def _make_planner(
    *,
    bundle: Any,
    engine: ExecutableRolloutEngine,
    spec: Any,
    data: dict[str, np.ndarray],
    anchor: int,
    horizon: int,
    q_des_velocity: np.ndarray,
    cost_config: JointSpaceCostConfig,
    rollout_config: PlannerRolloutConfig,
    nominal_preview_steps: int,
    previous_residual_source: str,
) -> LearnedDynamicsPlanner:
    device = bundle.device
    states = data["states"]
    commands = data["commands"]
    reference = data["q_des"]
    previous_q_ref, previous_velocity = _previous_command_state(commands, states, anchor, bundle.control_dt)
    previous_residual = np.zeros(5, dtype=np.float32)
    previous_residual_velocity = np.zeros(5, dtype=np.float32)
    if previous_residual_source == "logged" and "planner_requested_residual" in data:
        previous_residual, previous_residual_velocity = _previous_residual(
            data["planner_requested_residual"], anchor, bundle.control_dt, 5
        )
    nominal_start = anchor + int(nominal_preview_steps)
    nominal = reference[nominal_start : nominal_start + horizon]
    if nominal.shape != (horizon, 5):
        raise ValueError(f"nominal window is invalid at anchor {anchor}: {nominal.shape}")
    return LearnedDynamicsPlanner(
        model=bundle.model,
        normalizer=bundle.normalizer,
        model_type=bundle.model_type,
        state_dim=bundle.state_dim,
        target_mode=bundle.target_mode,
        control_dt=bundle.control_dt,
        initial_history=_make_history(states, commands, anchor, bundle.history_len).to(device).unsqueeze(0),
        q_des=torch.as_tensor(reference[anchor + 1 : anchor + 1 + horizon], dtype=torch.float32, device=device),
        dq_des=torch.as_tensor(q_des_velocity[anchor + 1 : anchor + 1 + horizon], dtype=torch.float32, device=device),
        nominal_q_ref=torch.as_tensor(nominal, dtype=torch.float32, device=device),
        previous_q_ref=torch.as_tensor(previous_q_ref, dtype=torch.float32, device=device),
        previous_q_ref_velocity=torch.as_tensor(previous_velocity, dtype=torch.float32, device=device),
        previous_residual=torch.as_tensor(previous_residual, dtype=torch.float32, device=device),
        previous_residual_velocity=torch.as_tensor(previous_residual_velocity, dtype=torch.float32, device=device),
        joint_low=torch.as_tensor(spec.joint_low, dtype=torch.float32, device=device),
        joint_high=torch.as_tensor(spec.joint_high, dtype=torch.float32, device=device),
        cost_config=cost_config,
        rollout_config=rollout_config,
        executable_command_spec=spec,
        executable_command_state=ExecutableCommandState(previous_q_ref, previous_velocity),
        executable_rollout_engine=engine,
    )


def _cem_config(args: argparse.Namespace, device: torch.device, horizon: int) -> CEMMPCConfig:
    return CEMMPCConfig(
        horizon=horizon,
        action_dim=5,
        decision_horizon=horizon,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        elite_ratio=args.elite_ratio,
        cem_iters=args.cem_iters,
        init_std=args.init_std,
        min_std=args.min_std,
        smoothing_alpha=args.smoothing_alpha,
        temporal_noise_alpha=args.temporal_noise_alpha,
        reset_std_each_step=args.reset_std_each_step,
        uniform_sample_ratio=args.uniform_sample_ratio,
        force_baseline_candidate=True,
        seed=args.seed,
        device=str(device),
        execute="lowest_cost",
        selection_validation="exact_final_pool",
        stage_one_task_mode="off",
    )


def _copy_controller_state(controller: CEMMPCController) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return controller.mean.detach().clone(), controller.std.detach().clone(), controller.generator.get_state()


def _restore_controller_state(
    controller: CEMMPCController,
    state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    mean, std, generator_state = state
    controller.mean = mean.clone()
    controller.std = std.clone()
    controller.generator.set_state(generator_state)


def _stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.percentile(finite, 50.0)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _velocity_stats(values: np.ndarray, limit: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "n": 0,
            "mean_abs": float("nan"),
            "p95_abs": float("nan"),
            "max_abs": float("nan"),
            "over_limit_fraction": float("nan"),
            "over_limit_count": 0,
        }
    absolute = np.abs(values)
    finite = np.isfinite(absolute)
    if not np.any(finite):
        return {
            "n": 0,
            "mean_abs": float("nan"),
            "p95_abs": float("nan"),
            "max_abs": float("nan"),
            "over_limit_fraction": float("nan"),
            "over_limit_count": 0,
        }
    finite_abs = absolute[finite]
    limit_array = np.asarray(limit, dtype=np.float64)
    over_limit = (absolute > limit_array) & finite
    return {
        "n": int(finite_abs.size),
        "mean_abs": float(np.mean(finite_abs)),
        "p95_abs": float(np.percentile(finite_abs, 95.0)),
        "max_abs": float(np.max(finite_abs)),
        "over_limit_fraction": float(np.count_nonzero(over_limit) / finite_abs.size),
        "over_limit_count": int(np.count_nonzero(over_limit)),
    }


def _velocity_summary(
    values: dict[str, dict[str, list[np.ndarray]]],
    state_velocity_limit: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for candidate in CANDIDATES:
        result[candidate] = {}
        for source in ("predicted_dq", "dq_des", "logged_dq"):
            chunks = values.get(candidate, {}).get(source, [])
            concatenated = np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(state_velocity_limit)))
            result[candidate][source] = _velocity_stats(concatenated, state_velocity_limit)
    return result


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _direction_agreement(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    nonzero = (np.abs(left) > 1e-9) & (np.abs(right) > 1e-9)
    if not np.any(nonzero):
        return float("nan")
    return float(np.mean(np.sign(left[nonzero]) == np.sign(right[nonzero])))


def _residual_direction_summary(
    residual_values: dict[str, list[np.ndarray]],
    dq_des_values: list[np.ndarray],
    joint_names: list[str],
) -> dict[str, Any]:
    dq_des = np.concatenate(dq_des_values, axis=0) if dq_des_values else np.empty((0, len(joint_names)))
    result: dict[str, Any] = {}
    for candidate in CANDIDATES:
        residual = (
            np.concatenate(residual_values.get(candidate, []), axis=0)
            if residual_values.get(candidate)
            else np.empty_like(dq_des)
        )
        by_joint: dict[str, Any] = {}
        for joint_index, joint_name in enumerate(joint_names):
            left = residual[:, joint_index]
            right = dq_des[:, joint_index]
            by_joint[joint_name] = {
                "correlation": _pearson_correlation(left, right),
                "direction_agreement": _direction_agreement(left, right),
                "mean_residual": float(np.nanmean(left)) if left.size else float("nan"),
                "mean_abs_residual": float(np.nanmean(np.abs(left))) if left.size else float("nan"),
                "mean_dq_des": float(np.nanmean(right)) if right.size else float("nan"),
                "n": int(np.count_nonzero(np.isfinite(left) & np.isfinite(right))),
            }
        result[candidate] = by_joint
    return result


def _weight_label(value: float) -> str:
    return f"{float(value):g}"


def _dq_limit_sweep(
    records: list[dict[str, Any]],
    anchors: list[int],
    weights: list[float],
    current_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(int(row["anchor"]), str(row["candidate"])): row for row in records}
    sweep_records: list[dict[str, Any]] = []
    sweep_summary: dict[str, Any] = {}
    sweep_selection: dict[str, Any] = {}
    for weight in weights:
        total_matrix = {candidate: [] for candidate in CANDIDATES}
        for anchor in anchors:
            for candidate in CANDIDATES:
                row = by_key.get((anchor, candidate))
                if row is None:
                    continue
                current_total = float(row["total"])
                raw_dq_limit = float(row["dq_limit"])
                if np.isfinite(current_total) and np.isfinite(raw_dq_limit):
                    non_dq_limit_cost = current_total - float(current_weight) * raw_dq_limit
                    total = non_dq_limit_cost + float(weight) * raw_dq_limit
                else:
                    non_dq_limit_cost = float("inf")
                    total = float("inf")
                total_matrix[candidate].append(total)
                sweep_records.append(
                    {
                        "anchor": anchor,
                        "candidate": candidate,
                        "w_dq_limit": float(weight),
                        "q_tracking": float(row["q_tracking"]),
                        "dq_limit": raw_dq_limit,
                        "dq_limit_contribution": float(weight) * raw_dq_limit,
                        "non_dq_limit_cost": non_dq_limit_cost,
                        "total": total,
                    }
                )
        matrix = {
            candidate: np.asarray(total_matrix[candidate], dtype=np.float64)
            for candidate in CANDIDATES
        }
        sweep_summary[_weight_label(weight)] = {
            candidate: {
                "total": _stats(matrix[candidate]),
                "dq_limit_contribution": _stats(
                    np.asarray(
                        [
                            row["dq_limit_contribution"]
                            for row in sweep_records
                            if row["w_dq_limit"] == float(weight) and row["candidate"] == candidate
                        ],
                        dtype=np.float64,
                    )
                ),
            }
            for candidate in CANDIDATES
        }
        sweep_selection[_weight_label(weight)] = _argmin_frequency(matrix, CANDIDATES)
    return sweep_records, {
        "weights": weights,
        "current_weight": float(current_weight),
        "selection": sweep_selection,
        "summary": sweep_summary,
        "note": (
            "Candidates and learned rollouts are fixed from the w_dq_limit=%.6g audit; "
            "only the total-cost re-scoring changes. CEM is not rerun for each weight."
        ) % float(current_weight),
    }


def _argmin_frequency(matrix: dict[str, np.ndarray], labels: tuple[str, ...]) -> dict[str, Any]:
    counts = {label: 0 for label in labels}
    valid_rows = 0
    for row in zip(*(matrix[label] for label in labels)):
        values = np.asarray(row, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        safe = np.where(finite, values, np.inf)
        winner = labels[int(np.argmin(safe))]
        counts[winner] += 1
        valid_rows += 1
    return {
        "counts": counts,
        "frequency": {label: float(counts[label] / valid_rows) if valid_rows else float("nan") for label in labels},
        "valid_rows": valid_rows,
    }


def _diagnosis(summary: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    means = {
        label: summary.get(label, {}).get("q_tracking", {}).get("mean", float("nan"))
        for label in CANDIDATES
    }
    totals = {
        label: summary.get(label, {}).get("total", {}).get("mean", float("nan"))
        for label in CANDIDATES
    }
    preview_q = [means[label] for label in ("preview_3", "preview_6") if np.isfinite(means[label])]
    preview_total = [totals[label] for label in ("preview_3", "preview_6") if np.isfinite(totals[label])]
    cem_q = means.get("CEM_best", float("nan"))
    if preview_q and np.isfinite(cem_q) and min(preview_q) < cem_q and preview_total:
        cem_total = totals.get("CEM_selected", float("nan"))
        if np.isfinite(cem_total) and min(preview_total) > cem_total:
            label = "cost_penalty_dominated"
            explanation = (
                "preview_3/preview_6 has lower learned q_tracking but higher total; "
                "residual/first/smoothness and dq_limit penalties outweigh the tracking gain."
            )
        else:
            label = "preview_tracking_advantage_without_clear_total_flip"
            explanation = "preview tracking is better, but the aggregate total comparison is not a clean regularization flip."
    elif np.isfinite(cem_q) and preview_q and cem_q < min(preview_q):
        label = "possible_model_or_cem_exploitation"
        explanation = (
            "CEM_best has lower learned q_tracking than both preview candidates; "
            "compare this prediction against real candidate performance before changing cost weights."
        )
    else:
        label = "inconclusive_or_mixed"
        explanation = "The aggregate q_tracking and total means do not match either simple diagnostic branch."
    return {
        "label": label,
        "explanation": explanation,
        "q_tracking_mean": means,
        "total_mean": totals,
        "argmin_q_tracking": selection.get("q_tracking", {}),
        "argmin_total": selection.get("total", {}),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SO101 offline candidate cost audit",
        "",
        f"rollout: `{report['rollout']}`",
        f"anchors: {report['protocol']['anchor_count']} ({report['protocol']['anchor_first']}–{report['protocol']['anchor_last']})",
        f"horizon: {report['protocol']['horizon']}; device: `{report['protocol']['device']}`",
        "",
        "Component columns are unweighted terms; `total` is the weighted cost. "
        "Weighted contributions are available in the JSON/CSV as `weighted_*`.",
        "",
        "## Candidate cost summary",
        "",
        "| candidate | q_tracking | dq_tracking | residual | servo | residual_velocity | residual_acceleration | first | joint_limit | dq_limit | total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in CANDIDATES:
        row = report["summary"][label]
        lines.append(
            "| " + label + " | " + " | ".join(f"{row[name]['mean']:.6g}" for name in TERM_NAMES) + " |"
        )
    lines += [
        "",
        "The JSON summary contains mean/median/P95 for every column above; the Markdown table shows means.",
    ]
    lines += [
        "",
        "## Velocity diagnostics",
        "",
        f"The velocity limit used by the barrier is `{report['protocol']['state_velocity_limit']}` rad/s per joint. "
        "`dq_des` is derived from the frozen q_des because this rollout does not contain a dq_des array; "
        "`logged_dq` is the observed state's velocity half.",
        "",
        "| candidate | source | mean | P95 | max | fraction > limit |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        for source in ("predicted_dq", "dq_des", "logged_dq"):
            stats = report["velocity_summary"][candidate][source]
            lines.append(
                f"| {candidate} | {source} | {stats['mean_abs']:.6g} | {stats['p95_abs']:.6g} | "
                f"{stats['max_abs']:.6g} | {stats['over_limit_fraction']:.6g} |"
            )
    lines += [
        "",
        "## Tracking-only comparison",
        "",
        f"tracking-only mode: `{report['protocol']['tracking_only']}`; hard projector and feasibility gates remain active.",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| baseline q_tracking mean | {report['tracking_only_comparison']['baseline_q_tracking_mean']:.6g} |",
        f"| CEM_selected q_tracking mean | {report['tracking_only_comparison']['CEM_selected_q_tracking_mean']:.6g} |",
        f"| mean improvement (baseline - selected) | {report['tracking_only_comparison']['absolute_mean_improvement']:.6g} |",
        f"| relative improvement | {report['tracking_only_comparison']['relative_mean_improvement']:.6g} |",
        f"| selected beats baseline | {report['tracking_only_comparison']['CEM_selected_beats_baseline_count']} / {report['tracking_only_comparison']['valid_anchors']} |",
        "",
        "Residual versus dq_des direction (requested residual, flattened over anchors and horizon):",
        "",
        "| candidate | joint | correlation | direction agreement |",
        "|---|---|---:|---:|",
    ]
    for candidate in CANDIDATES:
        for joint_name in report["protocol"]["joint_names"]:
            stats = report["residual_direction_summary"][candidate][joint_name]
            lines.append(
                f"| {candidate} | {joint_name} | {stats['correlation']:.6g} | {stats['direction_agreement']:.6g} |"
            )
    lines += [
        "",
        "## w_dq_limit sweep",
        "",
        report["dq_limit_sweep"]["note"],
        "",
        "| w_dq_limit | " + " | ".join(CANDIDATES) + " |",
        "|---:|" + "---:|" * len(CANDIDATES),
    ]
    for weight in report["dq_limit_sweep"]["weights"]:
        item = report["dq_limit_sweep"]["selection"][_weight_label(weight)]
        cells = []
        for candidate in CANDIDATES:
            count = item["counts"][candidate]
            frequency = item["frequency"][candidate]
            cells.append(f"{count} ({frequency:.3f})")
        lines.append(f"| {weight:g} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "For each sweep weight, the JSON contains mean/median/P95 total and dq_limit contribution. "
        "The per-anchor values are in `candidate_dq_limit_sweep.csv`.",
    ]
    lines += [
        "",
        "## Argmin frequency",
        "",
        "| metric | candidate counts | valid anchors |",
        "|---|---|---:|",
    ]
    for metric in ("q_tracking", "total"):
        item = report["selection"][metric]
        counts = ", ".join(f"{key}={value}" for key, value in item["counts"].items())
        lines.append(f"| {metric} | {counts} | {item['valid_rows']} |")
    lines += [
        "",
        "## Diagnostic branch",
        "",
        f"**{report['diagnosis']['label']}** — {report['diagnosis']['explanation']}",
        "",
        "This is an offline model/cost diagnosis. It does not establish real-robot superiority for a candidate that was not executed on the robot.",
        "",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--normalizer", required=True, type=Path)
    parser.add_argument("--hardware-config", default="configs/hardware/so101_follower.local.yaml", type=Path)
    parser.add_argument("--robot-config", default="configs/robots/so101.yaml", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto; no hardware connection is made")
    parser.add_argument("--history-len", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--max-anchors", type=int, default=100)
    parser.add_argument("--anchors", default=None, help="Comma-separated measured rollout ticks; overrides --max-anchors")
    parser.add_argument("--anchor-start", type=int, default=None)
    parser.add_argument("--anchor-end", type=int, default=None)
    parser.add_argument("--nominal-preview-steps", type=int, default=0)
    parser.add_argument("--previous-residual-source", choices=["logged", "zero"], default="logged")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--num-elites", type=int, default=None)
    parser.add_argument("--elite-ratio", type=float, default=0.08)
    parser.add_argument("--cem-iters", type=int, default=2)
    parser.add_argument("--init-std", type=float, default=0.5)
    parser.add_argument("--min-std", type=float, default=0.25)
    parser.add_argument("--smoothing-alpha", type=float, default=0.2)
    parser.add_argument("--temporal-noise-alpha", type=float, default=0.8)
    parser.add_argument("--uniform-sample-ratio", type=float, default=0.15)
    parser.add_argument("--reset-std-each-step", action="store_true")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--joint-limit-margin", type=float, default=0.02)
    parser.add_argument("--residual-max", default=None)
    parser.add_argument("--servo-scale", default=None)
    parser.add_argument("--state-velocity-limit", default=None)
    parser.add_argument("--w-q", type=float, default=1.0)
    parser.add_argument("--w-dq", type=float, default=0.10)
    parser.add_argument("--w-residual", type=float, default=0.20)
    parser.add_argument("--w-servo", type=float, default=0.05)
    parser.add_argument("--w-residual-velocity", type=float, default=0.05)
    parser.add_argument("--w-residual-acceleration", type=float, default=0.02)
    parser.add_argument("--w-first", type=float, default=0.20)
    parser.add_argument("--w-terminal", type=float, default=0.0)
    parser.add_argument("--w-joint-limit", type=float, default=10.0)
    parser.add_argument("--w-dq-limit", type=float, default=5.0)
    parser.add_argument("--temporal-discount", type=float, default=0.95)
    parser.add_argument("--barrier-max-weight", type=float, default=2.0)
    parser.add_argument("--joint-limit-safe-margin", type=float, default=0.08)
    parser.add_argument("--joint-limit-temp", type=float, default=0.02)
    parser.add_argument("--dq-limit-temp", type=float, default=0.1)
    parser.add_argument("--velocity-cost-mode", choices=["track", "damping"], default="track")
    parser.add_argument(
        "--dq-limit-sweep",
        default=",".join(str(value) for value in DEFAULT_DQ_LIMIT_SWEEP),
        help="Comma-separated w_dq_limit values for offline total-cost re-scoring; CEM candidates are not regenerated",
    )
    parser.add_argument(
        "--tracking-only",
        action="store_true",
        help="Diagnostic mode: keep only q_tracking in the soft objective; retain all physical projectors and feasibility gates",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.horizon <= 0 or args.max_anchors <= 0 or args.num_samples < 3 or args.cem_iters <= 0:
        raise SystemExit("horizon/max-anchors/cem-iters must be positive and num-samples must be at least 3")
    try:
        dq_limit_sweep = _nonnegative_values(args.dq_limit_sweep, DEFAULT_DQ_LIMIT_SWEEP, "dq_limit_sweep")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.tracking_only:
        args.w_dq = 0.0
        args.w_residual = 0.0
        args.w_servo = 0.0
        args.w_residual_velocity = 0.0
        args.w_residual_acceleration = 0.0
        args.w_first = 0.0
        args.w_terminal = 0.0
        args.w_joint_limit = 0.0
        args.w_dq_limit = 0.0
        dq_limit_sweep = [0.0]
    rollout_path = _resolve_path(args.rollout)
    checkpoint = _resolve_path(args.checkpoint)
    normalizer = _resolve_path(args.normalizer)
    hardware = load_hardware_config(_resolve_path(args.hardware_config))
    robot = load_robot_spec(_resolve_path(args.robot_config), validate_model=True)
    if robot.n_joints != 5:
        raise SystemExit("candidate_cost_audit.py currently targets the five controlled SO101 joints")
    data = _load_rollout(rollout_path)
    reference = data["q_des"]
    dq_des, ddq_des = _reference_derivatives(reference, hardware.control_dt)
    spec = _make_executable_spec(hardware)
    requested_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(requested_device)
    bundle = load_dynamics_bundle(
        checkpoint,
        normalizer,
        "gru",
        5,
        device,
        history_len=args.history_len,
        expected_robot_spec=robot,
    )
    if abs(bundle.control_dt - hardware.control_dt) > 1e-9:
        raise SystemExit(f"checkpoint dt {bundle.control_dt} does not match hardware dt {hardware.control_dt}")
    if bundle.history_len <= 0:
        raise SystemExit("checkpoint history length must be positive")
    engine = ExecutableRolloutEngine(
        model=bundle.model,
        normalizer=bundle.normalizer,
        model_type=bundle.model_type,
        state_dim=bundle.state_dim,
        target_mode=bundle.target_mode,
        control_dt=bundle.control_dt,
        spec=spec,
        backend="eager",
    )
    residual_max = _vector(args.residual_max, robot.residual_max, "residual_max")
    servo_scale = _vector(args.servo_scale, robot.servo_scale, "servo_scale")
    state_velocity_limit = _vector(args.state_velocity_limit, robot.state_velocity_limit, "state_velocity_limit")
    calibration = _reference_calibration(reference, dq_des, ddq_des, hardware.command_velocity_limit, hardware.command_acceleration_limit)
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    cost_config = JointSpaceCostConfig(
        cost_mode="residual",
        w_q=args.w_q,
        w_dq=args.w_dq,
        w_residual=args.w_residual,
        w_servo=args.w_servo,
        w_residual_velocity=args.w_residual_velocity,
        w_residual_acceleration=args.w_residual_acceleration,
        w_first=args.w_first,
        w_terminal=args.w_terminal,
        w_joint_limit=args.w_joint_limit,
        w_dq_limit=args.w_dq_limit,
        q_tracking_scale=tensor(calibration["q_tracking_scale"]),
        dq_tracking_scale=tensor(calibration["dq_tracking_scale"]),
        residual_scale=tensor(0.5 * residual_max),
        servo_scale=tensor(servo_scale),
        residual_velocity_scale=tensor(residual_max / bundle.control_dt),
        residual_acceleration_scale=tensor(residual_max / bundle.control_dt**2),
        temporal_discount=args.temporal_discount,
        barrier_max_weight=args.barrier_max_weight,
        state_velocity_limit=tensor(state_velocity_limit),
        joint_limit_safe_margin=args.joint_limit_safe_margin,
        joint_limit_temp=args.joint_limit_temp,
        dq_limit_temp=args.dq_limit_temp,
        control_dt=bundle.control_dt,
        velocity_cost_mode=args.velocity_cost_mode,
    )
    rollout_config = PlannerRolloutConfig(
        mpc_policy="residual",
        q_ref_velocity_limit=tensor(hardware.command_velocity_limit),
        q_ref_acceleration_limit=tensor(hardware.command_acceleration_limit),
        residual_max=tensor(residual_max),
        joint_limit_margin=args.joint_limit_margin,
        rollout_batch_size=args.num_samples,
        project_residual_kinematics=False,
        projection_backend="eager",
        projection_strategy="two_stage",
        residual_cost_semantics="requested",
        residual_feasibility_semantics="finite",
        residual_parameterization="full",
        residual_control_points=None,
    )
    # preview_6 must itself fit in the frozen reference window.
    max_anchor = len(reference) - args.horizon - max(0, args.nominal_preview_steps + 6)
    minimum_anchor = max(bundle.history_len - 1, int(data.get("active_start_tick", np.asarray(0)).reshape(-1)[0]))
    if args.anchor_start is not None:
        minimum_anchor = max(minimum_anchor, int(args.anchor_start))
    upper_anchor = max_anchor if args.anchor_end is None else min(max_anchor, int(args.anchor_end))
    if args.anchors:
        anchors = np.asarray([int(item.strip()) for item in args.anchors.split(",") if item.strip()], dtype=np.int64)
    else:
        if upper_anchor < minimum_anchor:
            raise SystemExit(f"no valid anchors in [{minimum_anchor}, {upper_anchor}]")
        anchors = np.linspace(minimum_anchor, upper_anchor, min(args.max_anchors, upper_anchor - minimum_anchor + 1), dtype=np.int64)
    anchors = np.unique(anchors[(anchors >= minimum_anchor) & (anchors <= upper_anchor)])
    if anchors.size == 0:
        raise SystemExit("no valid anchors after range filtering")
    if args.anchors is None and anchors.size < args.max_anchors:
        # A short rollout is valid, but make the effective count explicit.
        print(f"using {len(anchors)} valid anchors (requested at most {args.max_anchors})")

    cem_low = CEMMPCController(
        config=_cem_config(args, device, args.horizon),
        planner=None,
        joint_low=spec.joint_low,
        joint_high=spec.joint_high,
    )
    cem_mean_config = replace(_cem_config(args, device, args.horizon), execute="mean")
    cem_mean = CEMMPCController(
        config=cem_mean_config,
        planner=None,
        joint_low=spec.joint_low,
        joint_high=spec.joint_high,
    )
    previous_anchor: int | None = None
    records: list[dict[str, Any]] = []
    velocity_values: dict[str, dict[str, list[np.ndarray]]] = {
        candidate: {source: [] for source in ("predicted_dq", "dq_des", "logged_dq")}
        for candidate in CANDIDATES
    }
    residual_values: dict[str, list[np.ndarray]] = {candidate: [] for candidate in CANDIDATES}
    dq_des_values: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    offline_selection_modes: list[str] = []
    for anchor_value in anchors.tolist():
        anchor = int(anchor_value)
        try:
            planner = _make_planner(
                bundle=bundle,
                engine=engine,
                spec=spec,
                data=data,
                anchor=anchor,
                horizon=args.horizon,
                q_des_velocity=dq_des,
                cost_config=cost_config,
                rollout_config=rollout_config,
                nominal_preview_steps=args.nominal_preview_steps,
                previous_residual_source=args.previous_residual_source,
            )
            cem_low.planner = planner
            cem_mean.planner = planner
            shift = 0 if previous_anchor is None else max(0, anchor - previous_anchor)
            low_state = _copy_controller_state(cem_low)
            low_result = cem_low.plan(
                current_state=data["states"][anchor],
                previous_q_ref=data["commands"][max(0, anchor - 1)],
                warm_start_shift_steps=shift,
            )
            if low_result.failure:
                raise RuntimeError(f"CEM replay failed: {low_result.failure_reason}")
            offline_selection_modes.append(str(low_result.selection_mode))
            _restore_controller_state(cem_mean, low_state)
            mean_result = cem_mean.plan(
                current_state=data["states"][anchor],
                previous_q_ref=data["commands"][max(0, anchor - 1)],
                warm_start_shift_steps=shift,
            )
            if mean_result.failure:
                raise RuntimeError(f"CEM mean replay failed: {mean_result.failure_reason}")

            nominal = reference[
                anchor + int(args.nominal_preview_steps) : anchor + int(args.nominal_preview_steps) + args.horizon
            ]
            actions: dict[str, np.ndarray] = {
                "baseline": np.zeros((args.horizon, 5), dtype=np.float32),
            }
            preview_candidates = build_preview_residual_candidates(
                reference,
                anchor=anchor,
                horizon=args.horizon,
                nominal=nominal,
                residual_max=residual_max,
                preview_steps=(3, 6),
                nominal_preview_steps=args.nominal_preview_steps,
            )
            actions["preview_3"] = preview_candidates["preview:3"]
            actions["preview_6"] = preview_candidates["preview:6"]
            actions["CEM_best"] = np.asarray(low_result.best_sequence, dtype=np.float32)
            actions["CEM_mean"] = np.asarray(mean_result.selected_control_points, dtype=np.float32)
            actions["CEM_selected"] = np.asarray(low_result.selected_control_points, dtype=np.float32)
            dq_des_window = dq_des[anchor + 1 : anchor + 1 + args.horizon]
            logged_dq_window = data["states"][anchor + 1 : anchor + 1 + args.horizon, 5:10]
            if dq_des_window.shape != (args.horizon, 5) or logged_dq_window.shape != (args.horizon, 5):
                raise ValueError(
                    f"velocity comparison windows are invalid at anchor {anchor}: "
                    f"dq_des={dq_des_window.shape}, logged_dq={logged_dq_window.shape}"
                )
            dq_des_values.append(dq_des_window.copy())
            for label in CANDIDATES:
                terms, predicted_dq, requested_residual = _evaluate_action(planner, actions[label])
                row: dict[str, Any] = {"anchor": anchor, "candidate": label, "valid": int(np.all(np.isfinite(list(terms.values()))))}
                row.update(terms)
                for term, weight_name in WEIGHT_NAMES.items():
                    row[f"weighted_{term}"] = float(getattr(args, weight_name)) * terms[term]
                row["regularization"] = sum(row[f"weighted_{term}"] for term in REGULARIZATION_TERMS)
                row["non_tracking_penalty"] = terms["total"] - row["weighted_q_tracking"] - row["weighted_dq_tracking"]
                records.append(row)
                velocity_values[label]["predicted_dq"].append(predicted_dq)
                velocity_values[label]["dq_des"].append(dq_des_window.copy())
                velocity_values[label]["logged_dq"].append(logged_dq_window.copy())
                residual_values[label].append(requested_residual)
            previous_anchor = anchor
        except (RuntimeError, ValueError, KeyError) as exc:
            failures.append({"anchor": anchor, "error": f"{type(exc).__name__}: {exc}"})

    if not records:
        raise SystemExit("no anchor completed; first failures: " + json.dumps(failures[:3]))
    effective_anchors = sorted({int(row["anchor"]) for row in records})
    summary: dict[str, Any] = {}
    for label in CANDIDATES:
        rows = [row for row in records if row["candidate"] == label]
        summary[label] = {name: _stats(np.asarray([row[name] for row in rows], dtype=np.float64)) for name in TERM_NAMES}
        summary[label]["regularization"] = _stats(np.asarray([row["regularization"] for row in rows], dtype=np.float64))
        summary[label]["non_tracking_penalty"] = _stats(
            np.asarray([row["non_tracking_penalty"] for row in rows], dtype=np.float64)
        )
    metric_matrices = {
        name: {
            label: np.asarray(
                [next(row[name] for row in records if row["anchor"] == anchor and row["candidate"] == label) for anchor in effective_anchors],
                dtype=np.float64,
            )
            for label in CANDIDATES
        }
        for name in ("q_tracking", "total")
    }
    selection = {name: _argmin_frequency(metric_matrices[name], CANDIDATES) for name in ("q_tracking", "total")}
    velocity_summary = _velocity_summary(velocity_values, state_velocity_limit)
    joint_names = list(robot.joint_names)
    residual_direction_summary = _residual_direction_summary(
        residual_values,
        dq_des_values,
        joint_names,
    )
    baseline_q = metric_matrices["q_tracking"]["baseline"]
    selected_q = metric_matrices["q_tracking"]["CEM_selected"]
    valid_tracking_comparison = np.isfinite(baseline_q) & np.isfinite(selected_q)
    q_improvement = baseline_q[valid_tracking_comparison] - selected_q[valid_tracking_comparison]
    tracking_only_comparison = {
        "baseline_q_tracking_mean": float(np.mean(baseline_q[valid_tracking_comparison]))
        if np.any(valid_tracking_comparison)
        else float("nan"),
        "CEM_selected_q_tracking_mean": float(np.mean(selected_q[valid_tracking_comparison]))
        if np.any(valid_tracking_comparison)
        else float("nan"),
        "absolute_mean_improvement": float(np.mean(q_improvement)) if q_improvement.size else float("nan"),
        "relative_mean_improvement": float(np.mean(q_improvement) / np.mean(baseline_q[valid_tracking_comparison]))
        if q_improvement.size and np.mean(baseline_q[valid_tracking_comparison]) != 0.0
        else float("nan"),
        "CEM_selected_beats_baseline_count": int(np.count_nonzero(q_improvement > 0.0)),
        "CEM_selected_beats_baseline_frequency": float(np.mean(q_improvement > 0.0))
        if q_improvement.size
        else float("nan"),
        "valid_anchors": int(q_improvement.size),
    }
    sweep_records, dq_limit_sweep = _dq_limit_sweep(
        records,
        effective_anchors,
        dq_limit_sweep,
        current_weight=args.w_dq_limit,
    )
    weights = {name: float(getattr(args, weight_name)) for name, weight_name in WEIGHT_NAMES.items()}
    logged_modes = _load_logged_modes(rollout_path)
    report: dict[str, Any] = {
        "protocol": {
            "name": "offline_candidate_cost_audit",
            "anchor_count": len(effective_anchors),
            "anchor_first": effective_anchors[0],
            "anchor_last": effective_anchors[-1],
            "requested_anchor_count": int(len(anchors)),
            "horizon": args.horizon,
            "history_len": bundle.history_len,
            "control_dt_s": bundle.control_dt,
            "device": str(device),
            "state_history_source": "observed_states + actuator_q_ref from frozen rollout",
            "previous_residual_source": args.previous_residual_source,
            "candidate_projector": "SO101 canonical ExecutableCommandSpec via LearnedDynamicsPlanner",
            "cost_terms": "unweighted joint_space_tracking_cost terms; total is weighted",
            "state_velocity_limit": state_velocity_limit.tolist(),
            "joint_names": joint_names,
            "tracking_only": bool(args.tracking_only),
            "dq_limit_formula": "softplus((abs(dq_pred) - state_velocity_limit) / dq_limit_temp), mean+max aggregation",
            "dq_des_source": "finite difference of frozen rollout q_des; rollout has no dq_des array",
            "logged_dq_source": "states[:, 5:10] from the frozen observed_states/actual_states array",
            "cem_selected_source": "offline CEM replay with execute=lowest_cost",
            "cem_best_mean_same_population": True,
            "weights": weights,
        },
        "rollout": str(rollout_path),
        "checkpoint": str(checkpoint),
        "normalizer": str(normalizer),
        "hardware_config": str(_resolve_path(args.hardware_config)),
        "anchors": effective_anchors,
        "failures": failures,
        "historical_planner_selection_modes": {
            "available": bool(logged_modes),
            "counts": {mode: logged_modes.count(mode) for mode in sorted(set(logged_modes))},
            "note": "Historical modes are not assigned to audit anchors because old logs do not store plan anchor ticks and best/mean sequences.",
        },
        "offline_cem_selection_modes": {
            "counts": {mode: offline_selection_modes.count(mode) for mode in sorted(set(offline_selection_modes))},
            "note": "These modes belong to the offline CEM replay that produced CEM_selected.",
        },
        "summary": summary,
        "selection": selection,
        "velocity_summary": velocity_summary,
        "residual_direction_summary": residual_direction_summary,
        "tracking_only_comparison": tracking_only_comparison,
        "dq_limit_sweep": dq_limit_sweep,
    }
    report["diagnosis"] = _diagnosis(summary, selection)
    args.output_dir = _resolve_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_cost_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "candidate_cost_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "anchor", "candidate", "valid", *TERM_NAMES,
            *[f"weighted_{name}" for name in WEIGHT_NAMES],
            "regularization", "non_tracking_penalty",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with (args.output_dir / "candidate_dq_limit_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "anchor", "candidate", "w_dq_limit", "q_tracking", "dq_limit",
            "dq_limit_contribution", "non_dq_limit_cost", "total",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_records)
    with (args.output_dir / "candidate_velocity_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "candidate", "source", "n", "mean_abs", "p95_abs", "max_abs",
            "over_limit_fraction", "over_limit_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in CANDIDATES:
            for source in ("predicted_dq", "dq_des", "logged_dq"):
                writer.writerow({"candidate": candidate, "source": source, **velocity_summary[candidate][source]})
    with (args.output_dir / "candidate_residual_direction.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "candidate", "joint", "correlation", "direction_agreement",
            "mean_residual", "mean_abs_residual", "mean_dq_des", "n",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in CANDIDATES:
            for joint_name in joint_names:
                writer.writerow(
                    {
                        "candidate": candidate,
                        "joint": joint_name,
                        **residual_direction_summary[candidate][joint_name],
                    }
                )
    (args.output_dir / "candidate_cost_audit.md").write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    print(json.dumps({"output_dir": str(args.output_dir), "anchors": len(effective_anchors), "diagnosis": report["diagnosis"]}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
