#!/usr/bin/env python3
"""Paired SO101 comparison for rollout-gradient and action-input ablations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neural_dynamics.rollout import load_dynamics_bundle, rollout_dynamics_batch
from neural_dynamics.train_utils import load_checkpoint
from mpc.robot_config import load_robot_spec
from dynamics_modeling.scripts.evaluate_so101_sensitivity import (
    _history_windows,
    _load_rollout_arrays,
    _run_model_sensitivity,
)
from robot_runtime.config import load_hardware_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(error: np.ndarray, n_joints: int) -> dict[str, object]:
    q_error, dq_error = error[..., :n_joints], error[..., n_joints:]
    return {
        "q_rmse": float(np.sqrt(np.mean(np.square(q_error)))),
        "dq_rmse": float(np.sqrt(np.mean(np.square(dq_error)))),
        "q_rmse_per_joint": np.sqrt(np.mean(np.square(q_error), axis=0)).tolist(),
        "dq_rmse_per_joint": np.sqrt(np.mean(np.square(dq_error), axis=0)).tolist(),
    }


def _anchors(
    states: np.ndarray, groups: np.ndarray, episodes: np.ndarray, valid: np.ndarray,
    test_groups: np.ndarray, history_len: int, horizon: int,
) -> np.ndarray:
    test_mask = np.isin(groups, test_groups)
    selected = []
    for index in range(history_len - 1, len(states) - horizon):
        window = slice(index - history_len + 1, index + horizon)
        if (test_mask[index] and np.all(valid[index:index + horizon]) and
                np.all(groups[window] == groups[index]) and np.all(episodes[window] == episodes[index])):
            selected.append(index)
    if not selected:
        raise RuntimeError("no valid paired test windows")
    return np.asarray(selected, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rollout", required=True, help="Frozen active rollout containing actual states and executable commands")
    parser.add_argument(
        "--model", nargs=3, action="append", required=True, metavar=("LABEL", "CHECKPOINT", "NORMALIZER"),
        help="Repeat exactly three times: e75, absolute_u_grad, u_minus_q_grad",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-group-ids", default="43,44,45,46,47")
    parser.add_argument("--horizons", default="1,3,5,6,10,20")
    parser.add_argument("--kappa-ticks", default="300,400,600")
    parser.add_argument("--kappa-perturbation-deg", type=float, default=0.5)
    parser.add_argument("--hardware-config", default="configs/hardware/so101_follower.local.yaml")
    parser.add_argument("--expected-epoch", type=int, default=75)
    parser.add_argument("--rollout-batch-size", type=int, default=1024)
    args = parser.parse_args()
    if len(args.model) != 3:
        raise SystemExit("--model must be supplied exactly three times")

    horizons = sorted({int(value) for value in args.horizons.split(",") if value.strip()})
    ticks = [int(value) for value in args.kappa_ticks.split(",") if value.strip()]
    test_groups = np.asarray([int(value) for value in args.test_group_ids.split(",")], dtype=np.int64)
    max_horizon = max(horizons)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot = load_robot_spec("configs/robots/so101.yaml")

    with np.load(args.dataset, allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        next_states = np.asarray(data["next_states"], dtype=np.float32)
        groups = np.asarray(data["split_group_ids"], dtype=np.int64)
        episodes = np.asarray(data["episode_ids"], dtype=np.int64)
        valid = np.asarray(data["valid_target"], dtype=bool)
    rollout_states, rollout_commands = _load_rollout_arrays(Path(args.rollout))
    hardware = load_hardware_config(args.hardware_config)
    calibration = json.loads(hardware.calibration_path.read_text(encoding="utf-8"))
    cal_low = np.asarray([calibration[name]["range_min"] for name in hardware.joint_names], dtype=np.float64)
    cal_high = np.asarray([calibration[name]["range_max"] for name in hardware.joint_names], dtype=np.float64)
    sensitivity_horizon = 12

    loaded = []
    for label, checkpoint_name, normalizer_name in args.model:
        checkpoint_path, normalizer_path = Path(checkpoint_name), Path(normalizer_name)
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        metadata, config = checkpoint.get("metadata", {}), checkpoint.get("config", {})
        epoch = int(metadata.get("epoch", -1))
        if epoch != args.expected_epoch:
            raise ValueError(f"{label} checkpoint epoch={epoch}, expected {args.expected_epoch}")
        bundle = load_dynamics_bundle(
            checkpoint_path, normalizer_path, "gru", 5, device, expected_robot_spec=robot
        )
        loaded.append((label, checkpoint_path, normalizer_path, bundle, metadata, config))
    history_lengths = {entry[3].history_len for entry in loaded}
    if len(history_lengths) != 1:
        raise ValueError(f"paired models have different history lengths: {history_lengths}")
    history_len = next(iter(history_lengths))
    anchors = _anchors(states, groups, episodes, valid, test_groups, history_len, max_horizon)
    histories = np.stack([
        np.concatenate((states[i - history_len + 1:i + 1], actions[i - history_len + 1:i + 1]), axis=1)
        for i in anchors
    ])
    future_actions = np.stack([actions[i:i + max_horizon] for i in anchors])
    truth = np.stack([next_states[i:i + max_horizon] for i in anchors])

    report: dict[str, object] = {
        "protocol": {
            "dataset": str(Path(args.dataset).resolve()), "dataset_sha256": _sha256(Path(args.dataset)),
            "rollout": str(Path(args.rollout).resolve()), "rollout_sha256": _sha256(Path(args.rollout)),
            "test_group_ids": test_groups.tolist(), "window_count": int(len(anchors)),
            "horizons": horizons, "kappa_ticks": ticks,
            "kappa_perturbation_deg": args.kappa_perturbation_deg,
            "sensitivity_horizon": sensitivity_horizon,
            "sensitivity_history_semantics": "training_equivalent_[x_t,u_t]",
            "sensitivity_perturbation": "single-joint plus/minus encoder-count impulse and held commands",
        },
        "models": {},
    }
    csv_rows = []
    for label, checkpoint_path, normalizer_path, bundle, metadata, config in loaded:
        predicted = rollout_dynamics_batch(
            bundle.model, bundle.normalizer, bundle.model_type,
            torch.as_tensor(histories, device=device), torch.as_tensor(future_actions, device=device),
            bundle.state_dim, bundle.target_mode, bundle.control_dt,
            rollout_batch_size=args.rollout_batch_size,
        )[:, 1:].cpu().numpy()
        by_horizon = {}
        for horizon in horizons:
            values = _metrics(predicted[:, horizon - 1] - truth[:, horizon - 1], 5)
            by_horizon[str(horizon)] = values
            csv_rows.append({
                "label": label, "horizon": horizon, "q_rmse": values["q_rmse"],
                "dq_rmse": values["dq_rmse"], "kappa_mean": "",
            })

        valid_ticks = [tick for tick in ticks if history_len - 1 <= tick < len(rollout_states) - sensitivity_horizon]
        if not valid_ticks:
            raise ValueError(f"none of the κ ticks {ticks} has a complete H={sensitivity_horizon} rollout window")
        sensitivity = _run_model_sensitivity(
            bundle,
            _history_windows(rollout_states, rollout_commands, np.asarray(valid_ticks, dtype=np.int64), history_len),
            rollout_commands,
            np.asarray(valid_ticks, dtype=np.int64),
            sensitivity_horizon,
            cal_low,
            cal_high,
            hardware.raw_low[:5],
            hardware.raw_high[:5],
            args.rollout_batch_size,
            perturbation_counts=max(1, int(round(args.kappa_perturbation_deg * 4095.0 / 360.0))),
        )
        held_median = np.asarray(sensitivity["held"]["median"], dtype=np.float64)
        diag_h6 = np.diag(held_median[5])
        finite_h6 = np.abs(diag_h6)[np.isfinite(diag_h6)]
        scalar_h6 = float(np.mean(finite_h6)) if finite_h6.size else float("nan")
        kappa_rows = [{
            "tick": int(tick),
            "kappa_h6": float(np.mean(np.abs(np.diag(np.asarray(sensitivity["held"]["per_anchor"][row][5]))))),
            "per_joint_signed_gain_h6": np.diag(np.asarray(sensitivity["held"]["per_anchor"][row][5])).tolist(),
        } for row, tick in enumerate(valid_ticks)]
        kappa_summary = {
            "rows": kappa_rows, "mean": scalar_h6,
            "min": float(np.min(finite_h6)) if finite_h6.size else float("nan"),
            "max": float(np.max(finite_h6)) if finite_h6.size else float("nan"),
            "protocol": "training_equivalent_[x_t,u_t]_signed_single_joint_perturbations",
        }
        csv_rows.append({
            "label": label, "horizon": "kappa_h6", "q_rmse": "", "dq_rmse": "",
            "kappa_mean": kappa_summary["mean"],
        })
        report["models"][label] = {
            "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": _sha256(checkpoint_path),
            "normalizer": str(normalizer_path.resolve()), "normalizer_sha256": _sha256(normalizer_path),
            "epoch": int(metadata["epoch"]), "action_input_mode": bundle.action_input_mode,
            "target_mode": bundle.target_mode, "history_len": bundle.history_len,
            "training_identity": {key: config.get(key) for key in (
                "dataset_sha256", "seed", "batch_size", "micro_batch_size", "lr", "source_weights",
                "rollout_loss_steps", "rollout_loss_weight", "train_sample_stride",
                "validation_group_ids", "test_group_ids",
            )},
            "by_horizon": by_horizon, "kappa": kappa_summary, "sensitivity": sensitivity,
        }

    labels = [entry[0] for entry in loaded]
    changes = {}
    for before, after in zip(labels, labels[1:]):
        changes[f"{before}_to_{after}"] = {
            f"h{h}_q_rmse_ratio": (
                report["models"][after]["by_horizon"][str(h)]["q_rmse"] /
                report["models"][before]["by_horizon"][str(h)]["q_rmse"]
            ) for h in horizons
        } | {
            "kappa_ratio": report["models"][after]["kappa"]["mean"] /
                           report["models"][before]["kappa"]["mean"]
        }
    report["paired_changes"] = changes

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["label", "horizon", "q_rmse", "dq_rmse", "kappa_mean"])
        writer.writeheader(); writer.writerows(csv_rows)
    print(json.dumps(report["paired_changes"], indent=2))
    print(f"wrote {output_dir / 'comparison.json'} and {output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
