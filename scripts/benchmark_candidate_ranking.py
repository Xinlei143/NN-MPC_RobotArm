#!/usr/bin/env python3
"""Benchmark counterfactual candidate ranking for SO101 dynamics checkpoints.

The benchmark deliberately measures the control question that ordinary
rollout RMSE and scalar kappa do not answer: when several commands start from
the same measured state, does a checkpoint rank the candidates in the same
order as the robot?  Model predictions use the training-equivalent history
``[x_t, u_t]`` and the canonical executable command projector.  Real run
directories are optional; without them the script produces prediction tables,
and with them it reports Spearman rank correlation, pairwise accuracy, and
top-1 selection accuracy.

Candidate syntax:

* ``direct``
* ``preview:1`` (one-tick reference preview)
* ``lead:0.05`` (seconds, clipped to ``--max-correction-deg``)
* ``offset:shoulder_pan:0.5`` (joint name or index, degrees, held over H)
* ``active`` (the executable q_ref sequence from the base active log)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "dynamics_modeling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamics_modeling.neural_dynamics.rollout import load_dynamics_bundle, rollout_dynamics_batch
from mpc.robot_config import load_robot_spec
from robot_runtime.config import load_hardware_config
from robot_runtime.executable_command import (
    ExecutableCommandSpec,
    ExecutableCommandState,
    make_executable_command_spec,
    step_executable_command_np,
)


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _parse_key_value(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(path)


def _load_reference_file(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return np.asarray(loaded, dtype=np.float32)
    with loaded:
        for key in ("q_des", "q_des_ctrl", "joint_reference", "reference"):
            if key in loaded.files:
                return np.asarray(loaded[key], dtype=np.float32)
        if len(loaded.files) == 1:
            return np.asarray(loaded[loaded.files[0]], dtype=np.float32)
    raise KeyError(f"reference archive {path} has no q_des/q_des_ctrl array")


def _load_run(path: Path, reference_file: Path | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        state_key = "actual_states" if "actual_states" in archive.files else "states"
        command_key = "actuator_q_ref" if "actuator_q_ref" in archive.files else (
            "transmitted_q_ref" if "transmitted_q_ref" in archive.files else "actions"
        )
        states = np.asarray(archive[state_key], dtype=np.float32)
        commands = np.asarray(archive[command_key], dtype=np.float32)
        q_des = np.asarray(archive["q_des"], dtype=np.float32) if "q_des" in archive.files else None
        residual = np.asarray(archive["planner_requested_residual"], dtype=np.float32) if "planner_requested_residual" in archive.files else None
    if q_des is None:
        if reference_file is None:
            raise KeyError(f"{path}: no q_des; pass --reference-file")
        q_des = _load_reference_file(reference_file)
    if q_des.shape[0] >= len(states) and q_des.shape[1:] == (5,):
        q_des = q_des[:len(states)]
    if states.ndim != 2 or states.shape[1] < 10 or commands.shape != (len(states), 5) or q_des.shape != (len(states), 5):
        raise ValueError(f"{path}: expected states [N,>=10], commands/q_des [N,5], got {states.shape}, {commands.shape}, {q_des.shape}")
    result = {"states": states[:, :10], "commands": commands, "q_des": q_des}
    if residual is not None and residual.shape == commands.shape:
        result["residual"] = residual
    return result


def _make_spec(hardware: Any) -> ExecutableCommandSpec:
    payload = json.loads(Path(hardware.calibration_path).read_text(encoding="utf-8"))
    calibration_low = np.asarray([payload[name]["range_min"] for name in hardware.joint_names], dtype=np.float64)
    calibration_high = np.asarray([payload[name]["range_max"] for name in hardware.joint_names], dtype=np.float64)
    joint_low = hardware.hardware_joint_low if hardware.hardware_joint_low is not None else hardware.joint_low
    joint_high = hardware.hardware_joint_high if hardware.hardware_joint_high is not None else hardware.joint_high
    return make_executable_command_spec(
        joint_low=joint_low, joint_high=joint_high,
        velocity_limit=hardware.command_velocity_limit,
        acceleration_limit=hardware.command_acceleration_limit,
        relative_limit=hardware.relative_target_limit,
        raw_low=hardware.raw_low[:5], raw_high=hardware.raw_high[:5],
        calibration_low=calibration_low, calibration_high=calibration_high,
        control_dt=hardware.control_dt,
    )


def _reference_velocity(q_des: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.empty_like(q_des, dtype=np.float32)
    velocity[:-1] = np.diff(q_des, axis=0) / float(dt)
    velocity[-1] = velocity[-2] if len(velocity) > 1 else 0.0
    return velocity


def _candidate_base(name: str, q_des: np.ndarray, base_commands: np.ndarray, dq_des: np.ndarray, anchor: int, horizon: int, max_correction_rad: float) -> np.ndarray:
    target = q_des[anchor:anchor + horizon].copy()
    if name == "direct":
        return target
    if name == "active":
        return base_commands[anchor:anchor + horizon].copy()
    if name.startswith("preview:"):
        preview_steps = int(name.split(":", 1)[1])
        if preview_steps < 0:
            raise ValueError(f"preview steps must be non-negative in {name!r}")
        preview_target = q_des[anchor + preview_steps:anchor + preview_steps + horizon]
        if preview_target.shape != target.shape:
            raise ValueError(f"{name!r} runs past the reference tail at anchor {anchor}")
        return preview_target.copy()
    if name.startswith("lead:"):
        tau = float(name.split(":", 1)[1])
        return target + np.clip(tau * dq_des[anchor:anchor + horizon], -max_correction_rad, max_correction_rad)
    if name.startswith("offset:"):
        parts = name.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid offset candidate {name!r}")
        joint = parts[1]
        joint_index = int(joint) if joint.isdigit() else JOINT_NAMES.index(joint)
        delta = np.deg2rad(float(parts[2]))
        if not 0 <= joint_index < 5:
            raise ValueError(f"offset joint out of range in {name!r}")
        target[:, joint_index] += np.clip(delta, -max_correction_rad, max_correction_rad)
        return target
    raise ValueError(f"unknown candidate {name!r}")


def _canonical_sequence(requested: np.ndarray, states: np.ndarray, commands: np.ndarray, anchor: int, spec: ExecutableCommandSpec) -> np.ndarray:
    previous = commands[anchor - 1].astype(np.float64) if anchor > 0 else states[anchor, :5].astype(np.float64)
    previous_velocity = ((commands[anchor - 1] - commands[anchor - 2]) / spec.control_dt).astype(np.float64) if anchor > 1 else np.zeros(5, dtype=np.float64)
    state = ExecutableCommandState(previous, previous_velocity)
    output: list[np.ndarray] = []
    for offset, row in enumerate(requested):
        measured = states[min(anchor + offset, len(states) - 1), :5]
        result = step_executable_command_np(row, measured, state, spec)
        output.append(result.transmitted_q_ref)
        state = result.next_state
    return np.stack(output).astype(np.float32)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(predicted: np.ndarray, actual: np.ndarray) -> float:
    if len(predicted) < 2:
        return float("nan")
    x, y = _rank(predicted), _rank(actual)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ranking_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    n = len(predicted)
    pairs = 0
    correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_pred = float(predicted[i] - predicted[j])
            diff_actual = float(actual[i] - actual[j])
            if abs(diff_pred) <= 1e-12 or abs(diff_actual) <= 1e-12:
                continue
            pairs += 1
            correct += int(np.sign(diff_pred) == np.sign(diff_actual))
    return {
        "spearman": _spearman(predicted, actual),
        "pairwise_accuracy": float(correct / pairs) if pairs else float("nan"),
        "pairwise_count": int(pairs),
        "top1_correct": int(np.argmin(predicted) == np.argmin(actual)),
    }


def _load_model(label: str, checkpoint: Path, normalizer: Path, robot: Any, device: torch.device) -> Any:
    return load_dynamics_bundle(checkpoint, normalizer, "gru", 5, device, expected_robot_spec=robot)


def _plant_identity_match(expected: dict[str, Any], actual: Any) -> bool:
    return isinstance(actual, dict) and all(actual.get(key) == value for key, value in expected.items())


def _predict_costs(bundle: Any, base: dict[str, np.ndarray], candidates: dict[str, list[np.ndarray]], anchors: np.ndarray, horizon: int, batch_size: int) -> dict[str, np.ndarray]:
    history_len = int(bundle.history_len)
    histories: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    labels: list[str] = []
    targets: list[np.ndarray] = []
    for anchor_index, anchor in enumerate(anchors):
        for label, sequence_list in candidates.items():
            sequence = sequence_list[anchor_index]
            history = np.concatenate(
                (base["states"][anchor - history_len + 1:anchor + 1], base["commands"][anchor - history_len + 1:anchor + 1]),
                axis=1,
            ).astype(np.float32)
            # Training token at t is [x_t, u_t], so overwrite the current
            # action with the candidate's first executable command.
            history[-1, 10:] = sequence[0]
            histories.append(history)
            futures.append(sequence)
            labels.append(label)
            targets.append(base["q_des"][anchor + 1:anchor + horizon + 1])
    predicted = rollout_dynamics_batch(
        bundle.model,
        bundle.normalizer,
        bundle.model_type,
        torch.as_tensor(np.stack(histories), dtype=torch.float32, device=bundle.device),
        torch.as_tensor(np.stack(futures), dtype=torch.float32, device=bundle.device),
        bundle.state_dim,
        bundle.target_mode,
        bundle.control_dt,
        rollout_batch_size=batch_size,
    )[:, 1:, :5].detach().cpu().numpy()
    target = np.stack(targets).astype(np.float32)
    costs = np.mean(np.square(predicted - target), axis=(1, 2))
    output = {label: np.full(len(anchors), np.nan, dtype=np.float64) for label in candidates}
    cursor = 0
    for index, anchor in enumerate(anchors):
        for label in candidates:
            output[label][index] = float(costs[cursor])
            cursor += 1
    return output


def _actual_costs(runs: dict[str, dict[str, np.ndarray]], anchors: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for label, run in runs.items():
        values = np.full(len(anchors), np.nan, dtype=np.float64)
        for index, anchor in enumerate(anchors):
            future = run["states"][anchor + 1:anchor + horizon + 1, :5]
            target = run["q_des"][anchor + 1:anchor + horizon + 1]
            if future.shape != target.shape or len(future) != horizon:
                continue
            values[index] = float(np.mean(np.square(future - target)))
        result[label] = values
    return result


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# SO101 counterfactual candidate ranking", "", f"base rollout: `{report['base_rollout']}`", ""]
    selection = report.get("selection", {})
    lines += [f"primary selection metric: `{selection.get('primary_metric', 'not_available')}`; selected: `{selection.get('selected_label')}`", ""]
    lines += ["## Model-vs-real ranking", "", "| model | anchors | Spearman | pairwise accuracy | top-1 accuracy |", "|---|---:|---:|---:|---:|"]
    for label, row in report["models"].items():
        ranking = row.get("ranking", {})
        lines.append(f"| {label} | {ranking.get('anchors', 0)} | {ranking.get('spearman_mean', float('nan')):.4f} | {ranking.get('pairwise_accuracy_mean', float('nan')):.4f} | {ranking.get('top1_accuracy', float('nan')):.4f} |")
    lines += ["", "## Candidate predicted cost (rad²)", "", "| candidate | mean predicted cost |", "|---|---:|"]
    for label, value in report["predicted_cost_mean"].items():
        lines.append(f"| {label} | {value:.8g} |")
    lines += ["", "The primary checkpoint-selection signal is candidate ranking against repeated real runs; κ/sensitivity is retained only as a secondary safety/diagnostic gate.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-rollout", required=True, type=Path)
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--hardware-config", default="configs/hardware/so101_follower.local.yaml", type=Path)
    parser.add_argument("--robot-config", default="configs/robots/so101.yaml", type=Path)
    parser.add_argument("--model", action="append", nargs=3, metavar=("LABEL", "CHECKPOINT", "NORMALIZER"))
    parser.add_argument("--run", action="append", type=_parse_key_value, metavar="LABEL=PATH",
                        help="Repeated real run for an identically named candidate.")
    parser.add_argument("--candidate", action="append", dest="candidates", default=None,
                        help="Candidate label; repeat to override the default direct/lead/active set.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--anchors", default=None, help="Comma-separated anchor ticks; default evenly samples the valid range.")
    parser.add_argument("--max-anchors", type=int, default=64)
    parser.add_argument("--max-correction-deg", type=float, default=2.0)
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.horizon <= 0 or args.max_anchors <= 0 or args.max_correction_deg <= 0:
        raise SystemExit("horizon, max anchors, and max correction must be positive")
    if not args.model and not args.run:
        raise SystemExit("provide at least one --model and/or --run")
    base = _load_run(args.base_rollout, args.reference_file)
    hardware = load_hardware_config(args.hardware_config)
    spec = _make_spec(hardware)
    robot = load_robot_spec(args.robot_config, validate_model=True)
    horizon = int(args.horizon)
    labels = list(dict.fromkeys(args.candidates or ["direct", "lead:0.05", "lead:0.10", "lead:0.15", "lead:0.20", "active"]))
    preview_steps = [
        int(label.split(":", 1)[1]) for label in labels if label.startswith("preview:")
    ]
    max_anchor = len(base["states"]) - horizon - 1 - (max(preview_steps) if preview_steps else 0)
    if max_anchor < 16:
        raise SystemExit("base rollout is too short for the requested horizon")
    if args.anchors:
        anchors = np.asarray([int(value) for value in args.anchors.split(",") if value.strip()], dtype=np.int64)
    else:
        anchors = np.linspace(16, max_anchor, min(args.max_anchors, max_anchor - 15), dtype=np.int64)
    anchors = np.unique(anchors[(anchors >= 16) & (anchors <= max_anchor)])
    if not len(anchors):
        raise SystemExit("no valid anchors")
    dq_des = _reference_velocity(base["q_des"], hardware.control_dt)
    candidate_sequences: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    for anchor in anchors:
        for label in labels:
            requested = _candidate_base(label, base["q_des"], base["commands"], dq_des, int(anchor), horizon, np.deg2rad(args.max_correction_deg))
            candidate_sequences[label].append(_canonical_sequence(requested, base["states"], base["commands"], int(anchor), spec))
    # _predict_costs indexes the sequence list by the absolute anchor.  Keep a
    # dense temporary array so this remains explicit and avoids accidental
    # off-by-one alignment when a sparse anchor list is used.
    dense_sequences: dict[str, list[np.ndarray]] = {label: [candidate_sequences[label][i] for i in range(len(anchors))] for label in labels}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions: dict[str, dict[str, np.ndarray]] = {}
    model_metadata: dict[str, dict[str, object]] = {}
    for model_spec in args.model or []:
        model_label, checkpoint, normalizer = model_spec[0], Path(model_spec[1]), Path(model_spec[2])
        bundle = _load_model(model_label, checkpoint, normalizer, robot, device)
        if bundle.history_len < 2 or any(int(anchor) < bundle.history_len - 1 for anchor in anchors):
            raise SystemExit(f"{model_label}: anchors must be >= history_len-1 ({bundle.history_len - 1})")
        predictions[model_label] = _predict_costs(bundle, base, dense_sequences, anchors, horizon, args.rollout_batch_size)
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_metadata[model_label] = {
            "checkpoint": str(checkpoint.resolve()),
            "normalizer": str(normalizer.resolve()),
            "epoch": int(checkpoint_payload.get("metadata", {}).get("epoch", -1)),
            "action_input_mode": bundle.action_input_mode,
            "target_mode": bundle.target_mode,
            "plant_identity_match": _plant_identity_match(hardware.plant_identity(), bundle.config.get("plant_identity")),
        }
    runs: dict[str, dict[str, np.ndarray]] = {}
    for label, path in args.run or []:
        runs[label] = _load_run(path, args.reference_file)
    actual = _actual_costs(runs, anchors, horizon) if runs else {}
    report: dict[str, Any] = {
        "protocol": {
            "horizon": horizon,
            "control_dt_s": float(hardware.control_dt),
            "history_semantics": "training_equivalent_[x_t,u_t]_current_action_overwritten",
            "candidate_labels": labels,
            "candidate_projector": "canonical_executable_command_numpy",
            "target_cost": "mean_squared_q_error_over_t+1:t+H",
        },
        "base_rollout": str(args.base_rollout.resolve()),
        "anchors": anchors.tolist(),
        "predicted_cost_mean": {},
        "predictions": {},
        "actual_cost_mean": {label: float(np.nanmean(values)) for label, values in actual.items()},
        "models": {label: {"metadata": metadata, "ranking": {}} for label, metadata in model_metadata.items()},
    }
    for model_label, costs in predictions.items():
        report["predictions"][model_label] = {label: values.tolist() for label, values in costs.items()}
        report["models"][model_label]["ranking"] = {}
        if actual:
            common = [label for label in labels if label in actual and label in costs]
            rows: list[dict[str, float | int]] = []
            for index in range(len(anchors)):
                pred = np.asarray([costs[label][index] for label in common], dtype=np.float64)
                truth = np.asarray([actual[label][index] for label in common], dtype=np.float64)
                valid = np.isfinite(pred) & np.isfinite(truth)
                if int(np.sum(valid)) >= 2:
                    rows.append(_ranking_metrics(pred[valid], truth[valid]))
            report["models"][model_label]["ranking"] = {
                "anchors": len(rows),
                "spearman_mean": float(np.nanmean([row["spearman"] for row in rows])) if rows else float("nan"),
                "pairwise_accuracy_mean": float(np.nanmean([row["pairwise_accuracy"] for row in rows])) if rows else float("nan"),
                "top1_accuracy": float(np.mean([row["top1_correct"] for row in rows])) if rows else float("nan"),
                "per_anchor": rows,
            }
    if predictions:
        # Report the first model's candidate mean as a compact top-level table;
        # all per-model values remain in predictions.
        first_model = next(iter(predictions))
        report["predicted_cost_mean"] = {label: float(np.nanmean(values)) for label, values in predictions[first_model].items()}
    if actual and report["models"]:
        eligible = [
            (label, row["ranking"])
            for label, row in report["models"].items()
            if int(row["ranking"].get("anchors", 0)) > 0 and bool(row["metadata"].get("plant_identity_match", False))
        ]
        eligible.sort(
            key=lambda item: (
                -float(item[1].get("pairwise_accuracy_mean", -np.inf)),
                -float(item[1].get("spearman_mean", -np.inf)),
                -float(item[1].get("top1_accuracy", -np.inf)),
            )
        )
        report["selection"] = {
            "primary_metric": "candidate_pairwise_accuracy_then_spearman_then_top1",
            "kappa_role": "secondary_diagnostic_only",
            "selected_label": eligible[0][0] if eligible else None,
            "eligible": [label for label, _ in eligible],
        }
    else:
        report["selection"] = {
            "primary_metric": "candidate_pairwise_accuracy_then_spearman_then_top1",
            "kappa_role": "secondary_diagnostic_only",
            "selected_label": None,
            "eligible": [],
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_ranking.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (args.output_dir / "candidate_ranking.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "models": report["models"], "actual_cost_mean": report["actual_cost_mean"]}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
