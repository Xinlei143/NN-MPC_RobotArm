#!/usr/bin/env python3
"""Evaluate SO101 counterfactual sensitivity and select a checkpoint.

This replaces the old scalar ``kappa_h6`` probe.  Histories use the training
token convention ``[x_t, u_t]`` and commands are taken from the executable
transmission field when it is present.  For every anchor the script evaluates
single-joint +/- encoder-count impulse and held-step perturbations and reports
the signed 5x5 sensitivity matrix for horizons 1..12.  It also extracts the
recorded fast step/hold responses and uses held-out rollout RMSE as the second
selection gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "dynamics_modeling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neural_dynamics.rollout import load_dynamics_bundle, rollout_dynamics_batch
from mpc.robot_config import load_robot_spec
from robot_runtime.config import load_hardware_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_rollout_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        state_key = "actual_states" if "actual_states" in archive.files else "states"
        command_key = (
            "actuator_q_ref" if "actuator_q_ref" in archive.files
            else "transmitted_q_ref" if "transmitted_q_ref" in archive.files
            else "actions"
        )
        states = np.asarray(archive[state_key], dtype=np.float32)
        commands = np.asarray(archive[command_key], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 10 or commands.shape != (len(states), 5):
        raise ValueError(f"rollout must contain state [N,10] and executable command [N,5], got {states.shape} and {commands.shape}")
    return states, commands


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"states", "actions", "next_states", "split_group_ids", "valid_target"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"dataset is missing {sorted(missing)}")
        result = {key: np.asarray(archive[key]) for key in required}
        result["episode_ids"] = np.asarray(archive["episode_ids"], dtype=np.int64) if "episode_ids" in archive.files else np.zeros(len(result["states"]), dtype=np.int64)
        result["session_ids"] = np.asarray(archive["session_ids"]) if "session_ids" in archive.files else np.zeros(len(result["states"]), dtype=np.int64)
        result["motion_mode_ids"] = np.asarray(archive["motion_mode_ids"], dtype=np.int64) if "motion_mode_ids" in archive.files else np.zeros(len(result["states"]), dtype=np.int64)
        result["transmitted_q_ref"] = np.asarray(archive["transmitted_q_ref"], dtype=np.float32) if "transmitted_q_ref" in archive.files else result["actions"].astype(np.float32)
    n = len(result["states"])
    if any(len(value) != n for value in result.values()):
        raise ValueError("dataset arrays have inconsistent lengths")
    return result


def _raw_from_q(q: np.ndarray, calibration_low: np.ndarray, calibration_high: np.ndarray,
                raw_low: np.ndarray, raw_high: np.ndarray) -> np.ndarray:
    midpoint = (calibration_low + calibration_high) / 2.0
    # Match the canonical backend/planner encoder conversion exactly.
    raw = np.rint(np.rad2deg(np.asarray(q, dtype=np.float64)) * 4095.0 / 360.0 + midpoint).astype(np.int64)
    return np.clip(raw, raw_low.astype(np.int64), raw_high.astype(np.int64))


def _q_from_raw(raw: np.ndarray, calibration_low: np.ndarray, calibration_high: np.ndarray) -> np.ndarray:
    midpoint = (calibration_low + calibration_high) / 2.0
    return np.deg2rad((np.asarray(raw, dtype=np.float64) - midpoint) * 360.0 / 4095.0).astype(np.float32)


def _plant_identity_matches(expected: dict[str, Any], actual: Any) -> bool:
    """Use the same runtime-declared subset gate as real artifact loading."""
    return isinstance(actual, dict) and all(actual.get(key) == value for key, value in expected.items())


def _choose_indices(indices: np.ndarray, limit: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices.astype(np.int64)
    positions = np.linspace(0, len(indices) - 1, limit, dtype=np.int64)
    return indices[positions].astype(np.int64)


def _history_windows(states: np.ndarray, actions: np.ndarray, anchors: np.ndarray, history_len: int) -> np.ndarray:
    return np.stack([
        np.concatenate((states[index - history_len + 1:index + 1], actions[index - history_len + 1:index + 1]), axis=1)
        for index in anchors
    ]).astype(np.float32)


def _run_model_sensitivity(
    bundle: Any,
    histories: np.ndarray,
    commands: np.ndarray,
    ticks: np.ndarray,
    max_horizon: int,
    calibration_low: np.ndarray,
    calibration_high: np.ndarray,
    raw_low: np.ndarray,
    raw_high: np.ndarray,
    batch_size: int,
    perturbation_counts: int = 6,
) -> dict[str, Any]:
    if int(perturbation_counts) <= 0:
        raise ValueError("perturbation_counts must be positive")
    device = bundle.device
    matrices: dict[str, list[np.ndarray]] = {"impulse": [], "held": []}
    for tick_index, history in zip(ticks, histories, strict=True):
        baseline = commands[tick_index:tick_index + max_horizon].copy()
        if len(baseline) != max_horizon:
            continue
        raw0 = _raw_from_q(baseline[0], calibration_low, calibration_high, raw_low, raw_high)
        denominator = np.zeros(5, dtype=np.float32)
        impulse_sequences = [baseline]
        held_sequences = [baseline]
        for joint in range(5):
            plus_raw, minus_raw = raw0.copy(), raw0.copy()
            plus_raw[joint] += int(perturbation_counts)
            minus_raw[joint] -= int(perturbation_counts)
            plus_raw = np.clip(plus_raw, raw_low.astype(np.int64), raw_high.astype(np.int64))
            minus_raw = np.clip(minus_raw, raw_low.astype(np.int64), raw_high.astype(np.int64))
            plus = _q_from_raw(plus_raw, calibration_low, calibration_high)
            minus = _q_from_raw(minus_raw, calibration_low, calibration_high)
            denominator[joint] = plus[joint] - minus[joint]
            impulse_plus, impulse_minus = baseline.copy(), baseline.copy()
            impulse_plus[0, joint], impulse_minus[0, joint] = plus[joint], minus[joint]
            held_plus, held_minus = baseline.copy(), baseline.copy()
            held_plus[:, joint], held_minus[:, joint] = plus[joint], minus[joint]
            impulse_sequences.extend((impulse_plus, impulse_minus))
            held_sequences.extend((held_plus, held_minus))
        batch_histories = np.repeat(history[None, ...], len(impulse_sequences), axis=0)
        def predict(sequences: list[np.ndarray]) -> np.ndarray:
            return rollout_dynamics_batch(
                bundle.model, bundle.normalizer, bundle.model_type,
                torch.as_tensor(batch_histories, dtype=torch.float32, device=device),
                torch.as_tensor(np.stack(sequences), dtype=torch.float32, device=device),
                bundle.state_dim, bundle.target_mode, bundle.control_dt,
                rollout_batch_size=batch_size,
            )[:, 1:, :5].detach().cpu().numpy()
        impulse_prediction = predict(impulse_sequences)
        held_prediction = predict(held_sequences)
        for name, prediction in (("impulse", impulse_prediction), ("held", held_prediction)):
            matrix = np.zeros((max_horizon, 5, 5), dtype=np.float32)
            for joint in range(5):
                plus = prediction[1 + 2 * joint]
                minus = prediction[2 + 2 * joint]
                matrix[:, :, joint] = (plus - minus) / max(float(denominator[joint]), 1e-8)
            matrices[name].append(matrix)
    result: dict[str, Any] = {}
    for name, values in matrices.items():
        if not values:
            result[name] = {"count": 0, "median": [], "per_anchor": []}
            continue
        stack = np.stack(values)
        result[name] = {
            "count": int(len(stack)),
            "median": np.median(stack, axis=0).tolist(),
            "per_anchor": stack.tolist(),
            "diagonal_mean_by_horizon": np.mean(np.diagonal(np.median(stack, axis=0), axis1=1, axis2=2), axis=1).tolist(),
            "frobenius_by_horizon": np.linalg.norm(np.median(stack, axis=0), axis=(1, 2)).tolist(),
        }
    return result


def _extract_step_events(data: dict[str, np.ndarray], *, hold_steps: int, min_step_deg: float, max_other_deg: float) -> list[tuple[int, int]]:
    actions = data["transmitted_q_ref"]
    modes, sessions = data["motion_mode_ids"], data["session_ids"]
    events: list[tuple[int, int]] = []
    min_step = np.deg2rad(min_step_deg)
    max_other = np.deg2rad(max_other_deg)
    for base in range(8, len(actions) - hold_steps - 2):
        if modes[base] != 7 or sessions[base - 1] != sessions[base] or sessions[base + hold_steps] != sessions[base]:
            continue
        if np.max(np.abs(np.diff(actions[base - 7:base], axis=0))) > np.deg2rad(0.05):
            continue
        baseline = np.median(actions[base - 7:base], axis=0)
        held_delta = actions[base + hold_steps - 1] - baseline
        joint = int(np.argmax(np.abs(held_delta)))
        if abs(float(held_delta[joint])) < min_step:
            continue
        if np.max(np.delete(np.abs(held_delta), joint)) > max_other:
            continue
        crossing = np.flatnonzero(np.abs(actions[base:base + hold_steps, joint] - baseline[joint]) >= min_step)
        if crossing.size == 0:
            continue
        index = base + int(crossing[0])
        if events and events[-1][0] + hold_steps > index:
            continue
        events.append((index, joint))
    return events


def _empirical_step_response(data: dict[str, np.ndarray], events: list[tuple[int, int]], horizon: int, hold_steps: int) -> dict[str, Any]:
    states, actions = data["states"], data["transmitted_q_ref"]
    responses: list[list[list[np.ndarray]]] = [[[] for _ in range(5)] for _ in range(horizon)]
    for index, joint in events:
        delta = float(actions[index + hold_steps - 1, joint] - actions[index - 1, joint])
        if abs(delta) < 1e-6 or index + horizon >= len(states):
            continue
        pre_velocity = np.median(np.diff(states[index - 7:index, :5], axis=0), axis=0)
        for h in range(1, horizon + 1):
            response = (states[index + h, :5] - states[index - 1, :5] - h * pre_velocity) / delta
            responses[h - 1][joint].append(response.astype(np.float32))
    matrices = np.full((horizon, 5, 5), np.nan, dtype=np.float32)
    counts = np.zeros((horizon, 5), dtype=np.int64)
    for h in range(horizon):
        for joint in range(5):
            if responses[h][joint]:
                matrices[h, :, joint] = np.median(np.stack(responses[h][joint]), axis=0)
                counts[h, joint] = len(responses[h][joint])
    return {"event_count": len(events), "events": [[int(index), int(joint)] for index, joint in events], "median": matrices.tolist(), "counts": counts.tolist()}


def _heldout_rollout_rmse(bundle: Any, data: dict[str, np.ndarray], test_groups: np.ndarray, max_horizon: int, limit: int, batch_size: int) -> dict[str, Any]:
    states, actions, truth = data["states"], data["actions"].astype(np.float32), data["next_states"]
    groups, valid = data["split_group_ids"], data["valid_target"].astype(bool)
    history_len = bundle.history_len
    indices = np.flatnonzero(np.isin(groups, test_groups))
    indices = indices[(indices >= history_len - 1) & (indices + max_horizon < len(states))]
    valid_indices = []
    for index in indices:
        window = slice(index - history_len + 1, index + max_horizon)
        if valid[index:index + max_horizon].all() and np.all(groups[window] == groups[index]):
            valid_indices.append(int(index))
    selected = _choose_indices(np.asarray(valid_indices, dtype=np.int64), limit)
    if len(selected) == 0:
        return {"count": 0, "q_rmse_by_horizon": []}
    histories = _history_windows(states, actions, selected, history_len)
    future = np.stack([actions[index:index + max_horizon] for index in selected])
    target = np.stack([truth[index:index + max_horizon] for index in selected])[:, :, :5]
    prediction = rollout_dynamics_batch(
        bundle.model, bundle.normalizer, bundle.model_type,
        torch.as_tensor(histories, dtype=torch.float32, device=bundle.device),
        torch.as_tensor(future, dtype=torch.float32, device=bundle.device),
        bundle.state_dim, bundle.target_mode, bundle.control_dt,
        rollout_batch_size=batch_size,
    )[:, 1:, :5].detach().cpu().numpy()
    return {"count": int(len(selected)), "q_rmse_by_horizon": [float(np.sqrt(np.mean(np.square(prediction[:, h] - target[:, h])))) for h in range(max_horizon)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--model", nargs=3, action="append", required=True, metavar=("LABEL", "CHECKPOINT", "NORMALIZER"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hardware-config", default="configs/hardware/so101_follower.local.yaml")
    parser.add_argument("--sensitivity-ticks", default="300,400,600")
    parser.add_argument("--test-group-ids", default="43,44,45,46,47")
    parser.add_argument("--horizons", default="1,3,6,9,12")
    parser.add_argument("--max-sensitivity-anchors", type=int, default=12)
    parser.add_argument("--max-rollout-anchors", type=int, default=2048)
    parser.add_argument("--rollout-batch-size", type=int, default=1024)
    parser.add_argument("--min-step-deg", type=float, default=0.3)
    parser.add_argument("--max-other-deg", type=float, default=0.2)
    args = parser.parse_args()
    if not args.model:
        raise SystemExit("at least one --model LABEL CHECKPOINT NORMALIZER is required")
    horizons = sorted(set(_csv_ints(args.horizons)))
    max_horizon = max(horizons)
    sensitivity_ticks = np.asarray(_csv_ints(args.sensitivity_ticks), dtype=np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot = load_robot_spec("configs/robots/so101.yaml")
    hardware = load_hardware_config(args.hardware_config)
    cal_path = hardware.calibration_path
    calibration = json.loads(cal_path.read_text(encoding="utf-8"))
    cal_low = np.asarray([calibration[name]["range_min"] for name in hardware.joint_names], dtype=np.float64)
    cal_high = np.asarray([calibration[name]["range_max"] for name in hardware.joint_names], dtype=np.float64)
    raw_low, raw_high = hardware.raw_low[:5], hardware.raw_high[:5]
    rollout_states, rollout_commands = _load_rollout_arrays(Path(args.rollout))
    max_tick = len(rollout_states) - max_horizon - 1
    if np.any(sensitivity_ticks < 16) or np.any(sensitivity_ticks > max_tick):
        raise ValueError(f"sensitivity ticks must lie in [16,{max_tick}]")
    data = _load_dataset(Path(args.dataset))
    events = _extract_step_events(data, hold_steps=max_horizon, min_step_deg=args.min_step_deg, max_other_deg=args.max_other_deg)
    real_response = _empirical_step_response(data, events, max_horizon, max_horizon)
    models: dict[str, Any] = {}
    for label, checkpoint_name, normalizer_name in args.model:
        checkpoint, normalizer = Path(checkpoint_name), Path(normalizer_name)
        checkpoint_metadata = torch.load(checkpoint, map_location="cpu", weights_only=False).get("metadata", {})
        bundle = load_dynamics_bundle(checkpoint, normalizer, "gru", 5, device, expected_robot_spec=robot)
        if bundle.history_len > len(rollout_states) or bundle.history_len < 2:
            raise ValueError(f"{label}: unsupported history_len={bundle.history_len}")
        ticks = sensitivity_ticks[sensitivity_ticks >= bundle.history_len - 1]
        histories = _history_windows(rollout_states, rollout_commands, ticks, bundle.history_len)
        sensitivity = _run_model_sensitivity(
            bundle, histories, rollout_commands, ticks, max_horizon,
            cal_low, cal_high, raw_low, raw_high, args.rollout_batch_size,
        )
        heldout = _heldout_rollout_rmse(bundle, data, np.asarray(_csv_ints(args.test_group_ids), dtype=np.int64), max_horizon, args.max_rollout_anchors, args.rollout_batch_size)
        models[label] = {
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
            "normalizer": str(normalizer.resolve()), "normalizer_sha256": sha256(normalizer),
            "epoch": int(checkpoint_metadata.get("epoch", -1)), "action_input_mode": bundle.action_input_mode,
            "history_len": bundle.history_len, "target_mode": bundle.target_mode,
            "plant_identity_match": _plant_identity_matches(
                hardware.plant_identity(), bundle.config.get("plant_identity")
            ),
            "checkpoint_plant_identity": bundle.config.get("plant_identity"),
            "sensitivity": sensitivity, "heldout_rollout": heldout,
        }
    real = np.asarray(real_response["median"], dtype=np.float64)
    real_h12 = real[max_horizon - 1]
    finite_real = np.isfinite(np.diag(real_h12))
    eligible: list[tuple[str, float, int, float]] = []
    evaluated: list[dict[str, Any]] = []
    for label, report in models.items():
        held = np.asarray(report["sensitivity"]["held"]["median"], dtype=np.float64)
        model_h12 = held[max_horizon - 1]
        finite = finite_real & np.isfinite(np.diag(model_h12))
        signs = int(np.sum(np.sign(np.diag(model_h12)[finite]) == np.sign(np.diag(real_h12)[finite]))) if np.any(finite) else 0
        nonzero = finite & (np.abs(real_h12.diagonal()) > 1e-8)
        ratios = np.abs(np.diag(model_h12)[nonzero] / real_h12.diagonal()[nonzero]) if np.any(nonzero) else np.empty(0)
        sensitivity_error = float(np.linalg.norm((model_h12 - real_h12)[np.isfinite(real_h12)]) / max(np.linalg.norm(real_h12[np.isfinite(real_h12)]), 1e-8)) if np.any(np.isfinite(real_h12)) else float("inf")
        report["selection_metrics"] = {
            "horizon": max_horizon, "finite_diagonal_count": int(np.sum(finite)),
            "diagonal_sign_matches": signs, "diagonal_ratios": ratios.tolist(),
            "sensitivity_relative_error": sensitivity_error,
            "plant_identity_match": bool(report["plant_identity_match"]),
        }
        rmse_values = report["heldout_rollout"]["q_rmse_by_horizon"]
        rmse = float(rmse_values[max_horizon - 1]) if len(rmse_values) >= max_horizon else float("inf")
        sign_gate = np.sum(finite) > 0 and signs >= max(1, int(np.ceil(0.6 * np.sum(finite))))
        evaluated.append({
            "label": label,
            "plant_identity_match": bool(report["plant_identity_match"]),
            "sign_gate": bool(sign_gate),
            "sign_matches": signs,
            "h12_q_rmse": rmse,
            "sensitivity_relative_error": sensitivity_error,
        })
        if (
            bool(report["plant_identity_match"])
            and sign_gate
        ):
            eligible.append((label, rmse, signs, sensitivity_error))
    eligible.sort(key=lambda item: (item[1], -item[2], item[3]))
    selection = {
        "gate": "plant_identity_then_sensitivity_sign_then_heldout_rollout_rmse",
        "evaluated": evaluated,
        "eligible": [{"label": label, "h12_q_rmse": rmse, "sign_matches": signs, "sensitivity_relative_error": error} for label, rmse, signs, error in eligible],
        "selected_label": eligible[0][0] if eligible else None,
        "active_authorized": bool(eligible),
    }
    report = {
        "protocol": {
            "dataset": str(Path(args.dataset).resolve()), "dataset_sha256": sha256(Path(args.dataset)),
            "rollout": str(Path(args.rollout).resolve()), "rollout_sha256": sha256(Path(args.rollout)),
            "horizons": horizons, "max_horizon": max_horizon, "sensitivity_ticks": sensitivity_ticks.tolist(),
            "history_semantics": "training_equivalent_[x_t,u_t]_with_current_action_overwritten",
            "perturbation": "plus/minus six encoder counts on one joint; impulse and held variants",
            "real_response": "source dataset mode=7 fast step/hold, transmitted_q_ref, drift-corrected",
            "real_step_event_count": len(events),
        },
        "real_step_response": real_response,
        "models": models,
        "selection": selection,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sensitivity.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (output_dir / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))
    print(f"wrote {output_dir / 'sensitivity.json'}")


if __name__ == "__main__":
    main()
