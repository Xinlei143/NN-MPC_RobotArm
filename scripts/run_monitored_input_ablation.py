#!/usr/bin/env python3
"""Train absolute-u and u-q models sequentially while recording per-epoch kappa."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dynamics_modeling/scripts/train_dynamics.py"
DATASET = ROOT / "outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.npz"
MANIFEST = ROOT / "outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.manifest.json"
ROLLOUT = ROOT / "outputs/hardware/so101_pre_mpc/20260808_formal/active_circle_p0/rollout.npz"


def _load_checkpoint_epoch(path: Path) -> int | None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, RuntimeError, OSError, ValueError):
        return None
    metadata = checkpoint.get("metadata", {})
    epoch = metadata.get("epoch") if isinstance(metadata, dict) else None
    return int(epoch) if epoch is not None else None


def _kappa_for_checkpoint(checkpoint_path: Path, normalizer_path: Path, ticks: list[int]) -> dict[str, object]:
    """Measure signed H=1..12 sensitivity with training-equivalent histories.

    ``executed_tokens`` in old logs contain ``[x_t, u_{t-1}]`` and therefore
    cannot be used as a training token window without an off-by-one shift.  The
    current rollout stores ``actual_states`` and executable ``actuator_q_ref``
    separately; this probe builds ``[x_t, u_t]`` windows from those fields and
    perturbs one encoder joint at a time.
    """
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "dynamics_modeling"))
    from dynamics_modeling.scripts.evaluate_so101_sensitivity import (
        _load_rollout_arrays, _run_model_sensitivity, _history_windows,
    )
    from neural_dynamics.rollout import load_dynamics_bundle
    from mpc.robot_config import load_robot_spec
    from robot_runtime.config import load_hardware_config

    rollout_states, rollout_commands = _load_rollout_arrays(ROLLOUT)
    hardware = load_hardware_config("configs/hardware/so101_follower.local.yaml")
    calibration = json.loads(hardware.calibration_path.read_text(encoding="utf-8"))
    cal_low = np.asarray([calibration[name]["range_min"] for name in hardware.joint_names], dtype=np.float64)
    cal_high = np.asarray([calibration[name]["range_max"] for name in hardware.joint_names], dtype=np.float64)
    bundle = load_dynamics_bundle(
        checkpoint_path,
        normalizer_path,
        "gru",
        5,
        torch.device("cpu"),
        expected_robot_spec=load_robot_spec("configs/robots/so101.yaml"),
    )
    valid_ticks = [tick for tick in ticks if bundle.history_len - 1 <= tick < len(rollout_states) - 12]
    if not valid_ticks:
        raise ValueError(
            f"none of the requested κ ticks {ticks} has a complete history/H=12 window "
            f"for rollout length {len(rollout_states)}"
        )
    histories = _history_windows(rollout_states, rollout_commands, np.asarray(valid_ticks, dtype=np.int64), bundle.history_len)
    sensitivity = _run_model_sensitivity(
        bundle, histories, rollout_commands, np.asarray(valid_ticks, dtype=np.int64), 12,
        cal_low, cal_high, hardware.raw_low[:5], hardware.raw_high[:5], 256,
    )
    held = np.asarray(sensitivity["held"]["median"], dtype=np.float64)
    matrix_h6 = held[5] if len(held) >= 6 else np.full((5, 5), np.nan)
    diag = np.diag(matrix_h6)
    finite_abs = np.abs(diag)[np.isfinite(diag)]
    scalar_h6 = float(np.mean(finite_abs)) if finite_abs.size else float("nan")
    kappa_min = float(np.min(finite_abs)) if finite_abs.size else float("nan")
    kappa_max = float(np.max(finite_abs)) if finite_abs.size else float("nan")
    rows = [{"tick": int(tick), "per_joint_signed_gain_h6": matrix_h6.diagonal().tolist()} for tick in valid_ticks]
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "epoch": _load_checkpoint_epoch(checkpoint_path),
        "action_input_mode": bundle.action_input_mode,
        "protocol": "training_equivalent_[x_t,u_t]_signed_single_joint_encoder_perturbations",
        "sensitivity": sensitivity,
        # Compatibility scalar for the existing monitoring CSV.  It is the
        # mean absolute diagonal gain at H=6, not the old all-joints L2 norm.
        "kappa_h6": {"rows": rows, "mean": scalar_h6, "min": kappa_min, "max": kappa_max},
    }


def _train_command(input_mode: str, epochs: int, save_dir: Path) -> list[str]:
    return [
        sys.executable, str(TRAIN),
        "--data_path", str(DATASET),
        "--dataset_manifest", str(MANIFEST),
        "--robot_config", "configs/robots/so101.yaml",
        "--model_type", "gru", "--history_len", "16",
        "--target_mode", "delta_state", "--action_input_mode", input_mode,
        "--batch_size", "8192", "--micro_batch_size", "1024",
        "--epochs", str(epochs), "--lr", "1e-4", "--seed", "10",
        "--num_workers", "2", "--pin_memory", "--amp",
        "--q_weight", "1.0", "--dq_weight", "1.0",
        "--loss_type", "huber", "--huber_delta", "1.0",
        "--rollout_loss_steps", "20", "--rollout_loss_weight", "0.025",
        "--rollout_loss_discount", "1.0", "--control_dt", "0.03333333333333333",
        "--test_group_ids", "43,44,45,46,47",
        "--validation_group_ids", "38,39,40,41,42",
        "--train_sample_stride", "2", "--val_sample_stride", "1",
        "--source_weights", "0:1,1:4", "--save_dir", str(save_dir),
    ]


def _run_one(label: str, input_mode: str, epochs: int, output_root: Path, results_root: Path, ticks: list[int], poll_seconds: float) -> list[dict[str, object]]:
    save_dir = output_root / label
    save_dir.mkdir(parents=True, exist_ok=True)
    before = {path for path in save_dir.glob("gru_*") if path.is_dir()}
    command = _train_command(input_mode, epochs, save_dir)
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT)
    run_dir: Path | None = None
    seen_epoch = 0
    results: list[dict[str, object]] = []
    while process.poll() is None or run_dir is None or seen_epoch < epochs:
        candidates = {path for path in save_dir.glob("gru_*") if path.is_dir()} - before
        if candidates:
            run_dir = sorted(candidates)[-1]
        if run_dir is not None:
            latest = run_dir / "latest_model.pt"
            epoch = _load_checkpoint_epoch(latest) if latest.exists() else None
            if epoch is not None and epoch > seen_epoch:
                # Wait for the normalizer and checkpoint write to be visible.
                normalizer = run_dir / "normalizer.pt"
                if normalizer.exists():
                    result = _kappa_for_checkpoint(latest, normalizer, ticks)
                    result["label"] = label
                    result["checkpoint_type"] = "latest_model"
                    results.append(result)
                    seen_epoch = epoch
                    epoch_dir = results_root / label / f"epoch_{epoch:03d}"
                    epoch_dir.mkdir(parents=True, exist_ok=True)
                    (epoch_dir / "kappa.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                    print(f"[κ] {label} epoch={epoch} mean={result['kappa_h6']['mean']:.6f} "
                          f"min={result['kappa_h6']['min']:.6f} max={result['kappa_h6']['max']:.6f}", flush=True)
        if process.poll() is not None:
            # A completed process should have emitted the final latest checkpoint.
            # If it did not, fail rather than polling forever after an early exit.
            break
        time.sleep(poll_seconds)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if seen_epoch != epochs:
        raise RuntimeError(f"{label} exited at epoch {seen_epoch}, expected {epochs}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-root", default="dynamics_modeling/outputs/checkpoints_real/monitored_input_ablation_20260809")
    parser.add_argument("--results-root", default="docs/hardware/so101-input-ablation-20260809")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--kappa-ticks", default="300,400,600")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    ticks = [int(value) for value in args.kappa_ticks.split(",") if value.strip()]
    output_root = (ROOT / args.output_root).resolve()
    results_root = (ROOT / args.results_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, object]] = []
    for label, input_mode in (("absolute_u_grad", "absolute_q_ref"), ("u_minus_q_grad", "q_ref_minus_q")):
        all_results.extend(_run_one(label, input_mode, args.epochs, output_root, results_root, ticks, args.poll_seconds))
    summary = results_root / "kappa_by_epoch.csv"
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "epoch", "kappa_mean", "kappa_min", "kappa_max", "kappa_tick_300", "kappa_tick_400", "kappa_tick_600"])
        for result in all_results:
            rows = {int(row["tick"]): row for row in result["kappa_h6"]["rows"]}
            writer.writerow([
                result["label"], result["epoch"], result["kappa_h6"]["mean"], result["kappa_h6"]["min"], result["kappa_h6"]["max"],
                rows.get(300, {}).get("kappa_h6", ""), rows.get(400, {}).get("kappa_h6", ""), rows.get(600, {}).get("kappa_h6", ""),
            ])
    manifest = {
        "epochs": args.epochs,
        "input_modes": {"absolute_u_grad": "absolute_q_ref", "u_minus_q_grad": "q_ref_minus_q"},
        "dataset": str(DATASET.resolve()), "rollout": str(ROLLOUT.resolve()), "kappa_ticks": ticks,
        "checkpoint_root": str(output_root), "results_root": str(results_root),
        "results": all_results,
    }
    (results_root / "monitor_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary} and {results_root / 'monitor_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
