#!/usr/bin/env python3
"""Resume both monitored input-ablation runs from epoch 30 to epoch 45."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dynamics_modeling/scripts/train_dynamics.py"
OUTPUT_ROOT = ROOT / "dynamics_modeling/outputs/checkpoints_real/monitored_input_ablation_20260809"
RESULTS_ROOT = ROOT / "docs/hardware/so101-input-ablation-20260809"

_MONITOR_SPEC = importlib.util.spec_from_file_location(
    "local_monitored_input_ablation", ROOT / "scripts/run_monitored_input_ablation.py"
)
if _MONITOR_SPEC is None or _MONITOR_SPEC.loader is None:
    raise RuntimeError("cannot load monitored input-ablation helper")
_MONITOR_MODULE = importlib.util.module_from_spec(_MONITOR_SPEC)
_MONITOR_SPEC.loader.exec_module(_MONITOR_MODULE)
_kappa_for_checkpoint = _MONITOR_MODULE._kappa_for_checkpoint
_load_checkpoint_epoch = _MONITOR_MODULE._load_checkpoint_epoch


def _run_dir(label: str) -> Path:
    candidates = sorted(path for path in (OUTPUT_ROOT / label).glob("gru_*") if path.is_dir())
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one run directory for {label}, got {candidates}")
    return candidates[0]


def _wait_until_epoch(run_dir: Path, target: int, poll_seconds: float) -> None:
    while True:
        latest = run_dir / "latest_model.pt"
        epoch = _load_checkpoint_epoch(latest) if latest.exists() else None
        if epoch is not None and epoch >= target:
            print(f"[{run_dir.parent.name}] reached epoch {epoch}; continuation is ready", flush=True)
            return
        time.sleep(poll_seconds)


def _command(label: str, input_mode: str, run_dir: Path) -> list[str]:
    return [
        sys.executable, str(TRAIN),
        "--data_path", "outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.npz",
        "--dataset_manifest", "outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.manifest.json",
        "--robot_config", "configs/robots/so101.yaml",
        "--model_type", "gru", "--history_len", "16",
        "--target_mode", "delta_state", "--action_input_mode", input_mode,
        "--batch_size", "8192", "--micro_batch_size", "1024",
        "--epochs", "45", "--lr", "1e-4", "--seed", "10",
        "--num_workers", "2", "--pin_memory", "--amp",
        "--q_weight", "1.0", "--dq_weight", "1.0",
        "--loss_type", "huber", "--huber_delta", "1.0",
        "--rollout_loss_steps", "20", "--rollout_loss_weight", "0.025",
        "--rollout_loss_discount", "1.0", "--control_dt", "0.03333333333333333",
        "--test_group_ids", "43,44,45,46,47",
        "--validation_group_ids", "38,39,40,41,42",
        "--train_sample_stride", "2", "--val_sample_stride", "1",
        "--source_weights", "0:1,1:4",
        "--resume_checkpoint", str(run_dir / "latest_model.pt"),
        "--save_dir", str(run_dir),
    ]


def _resume_one(label: str, input_mode: str, run_dir: Path, poll_seconds: float) -> list[dict[str, object]]:
    latest = run_dir / "latest_model.pt"
    current_epoch = _load_checkpoint_epoch(latest) if latest.exists() else None
    if current_epoch is not None and current_epoch >= 45:
        # A prior monitor may have finished training but been interrupted just
        # before writing the final diagnostic.  Backfill that diagnostic only.
        epoch_dir = RESULTS_ROOT / label / "epoch_045"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        output = epoch_dir / "kappa.json"
        if not output.exists():
            result = _kappa_for_checkpoint(latest, run_dir / "normalizer.pt", [300, 400, 600])
            result["label"] = label
            result["phase"] = "continuation_epoch31_45"
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            summary = result["kappa_h6"]
            print(f"[κ] {label} epoch=45 mean={summary['mean']:.6f} "
                  f"min={summary['min']:.6f} max={summary['max']:.6f}", flush=True)
            return [result]
        print(f"[{label}] continuation already at epoch 45; diagnostic exists", flush=True)
        return []
    command = _command(label, input_mode, run_dir)
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT)
    seen_epoch = 30
    results: list[dict[str, object]] = []
    while process.poll() is None:
        latest = run_dir / "latest_model.pt"
        epoch = _load_checkpoint_epoch(latest) if latest.exists() else None
        if epoch is not None and epoch > seen_epoch:
            result = _kappa_for_checkpoint(latest, run_dir / "normalizer.pt", [300, 400, 600])
            result["label"] = label
            result["phase"] = "continuation_epoch31_45"
            results.append(result)
            seen_epoch = epoch
            epoch_dir = RESULTS_ROOT / label / f"epoch_{epoch:03d}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            (epoch_dir / "kappa.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            summary = result["kappa_h6"]
            print(f"[κ] {label} epoch={epoch} mean={summary['mean']:.6f} "
                  f"min={summary['min']:.6f} max={summary['max']:.6f}", flush=True)
        time.sleep(poll_seconds)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if seen_epoch != 45:
        raise RuntimeError(f"{label} continuation exited at epoch {seen_epoch}, expected 45")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    args = parser.parse_args()
    labels = (
        ("absolute_u_grad", "absolute_q_ref"),
        ("u_minus_q_grad", "q_ref_minus_q"),
    )
    run_dirs = {label: _run_dir(label) for label, _ in labels}
    print("waiting for both initial runs to reach epoch 30", flush=True)
    for run_dir in run_dirs.values():
        _wait_until_epoch(run_dir, 30, args.poll_seconds)

    all_results: list[dict[str, object]] = []
    for label, input_mode in labels:
        all_results.extend(_resume_one(label, input_mode, run_dirs[label], args.poll_seconds))
    manifest = {
        "initial_epochs": [1, 30], "continuation_epochs": [31, 45],
        "run_dirs": {label: str(path.resolve()) for label, path in run_dirs.items()},
        "results": all_results,
    }
    (RESULTS_ROOT / "continuation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {RESULTS_ROOT / 'continuation_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
