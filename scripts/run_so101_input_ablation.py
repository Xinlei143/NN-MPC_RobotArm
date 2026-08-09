#!/usr/bin/env python3
"""Run the paired SO101 rollout-gradient/action-input training experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dynamics_modeling" / "scripts" / "train_dynamics.py"
COMPARE = ROOT / "dynamics_modeling" / "scripts" / "compare_real_model_inputs.py"
DATASET = ROOT / "outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.npz"
MANIFEST = DATASET.with_suffix(".manifest.json")
ROLLOUT = ROOT / "outputs/hardware/so101_pre_mpc/20260808_formal/active_circle_p0/rollout.npz"
BASELINE_DIR = ROOT / "dynamics_modeling/outputs/checkpoints_real/gru_20260808_171118"


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _new_run(parent: Path, before: set[Path]) -> Path:
    candidates = {path for path in parent.glob("gru_*") if path.is_dir()} - before
    if len(candidates) != 1:
        raise RuntimeError(f"expected one new training directory under {parent}, got {sorted(candidates)}")
    return candidates.pop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="dynamics_modeling/outputs/checkpoints_real/paired_input_ablation_20260808",
    )
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--micro-batch-size", type=int, default=1024)
    args = parser.parse_args()
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    runs: dict[str, Path] = {}

    for label, input_mode in (
        ("absolute_u_grad", "absolute_q_ref"),
        ("u_minus_q_grad", "q_ref_minus_q"),
    ):
        parent = output_root / label
        parent.mkdir(parents=True, exist_ok=True)
        before = {path for path in parent.glob("gru_*") if path.is_dir()}
        command = [
            python, str(TRAIN),
            "--data_path", str(DATASET), "--dataset_manifest", str(MANIFEST),
            "--robot_config", "configs/robots/so101.yaml",
            "--model_type", "gru", "--history_len", "16",
            "--action_input_mode", input_mode,
            "--batch_size", "8192", "--micro_batch_size", str(args.micro_batch_size),
            "--epochs", str(args.epochs), "--lr", "0.0001",
            "--save_dir", str(parent), "--seed", "10", "--num_workers", "2",
            "--pin_memory", "--amp", "--q_weight", "1.0", "--dq_weight", "1.0",
            "--loss_type", "huber", "--huber_delta", "1.0",
            "--rollout_loss_steps", "20", "--rollout_loss_weight", "0.025",
            "--rollout_loss_discount", "1.0", "--target_mode", "delta_state",
            "--control_dt", "0.03333333333333333",
            "--test_group_ids", "43,44,45,46,47",
            "--validation_group_ids", "38,39,40,41,42",
            "--train_sample_stride", "2", "--val_sample_stride", "1",
            "--source_weights", "0:1,1:4",
        ]
        _run(command)
        runs[label] = _new_run(parent, before)

    comparison_dir = output_root / "comparison"
    _run([
        python, str(COMPARE), "--dataset", str(DATASET), "--rollout", str(ROLLOUT),
        "--model", "e75", str(BASELINE_DIR / "best_rollout_model.pt"), str(BASELINE_DIR / "normalizer.pt"),
        "--model", "absolute_u_grad", str(runs["absolute_u_grad"] / "latest_model.pt"),
        str(runs["absolute_u_grad"] / "normalizer.pt"),
        "--model", "u_minus_q_grad", str(runs["u_minus_q_grad"] / "latest_model.pt"),
        str(runs["u_minus_q_grad"] / "normalizer.pt"),
        "--output-dir", str(comparison_dir), "--expected-epoch", str(args.epochs),
    ])
    manifest = {
        "epochs": args.epochs, "effective_batch_size": 8192,
        "micro_batch_size": args.micro_batch_size,
        "runs": {label: str(path) for label, path in runs.items()},
        "comparison": str(comparison_dir / "comparison.json"),
    }
    (output_root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
