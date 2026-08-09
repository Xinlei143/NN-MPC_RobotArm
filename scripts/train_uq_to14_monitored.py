#!/usr/bin/env python3
"""Train a fresh u-q model to epoch 14 and record κ at every checkpoint."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dynamics_modeling/scripts/train_dynamics.py"
OUTPUT_ROOT = ROOT / "dynamics_modeling/outputs/checkpoints_real/u_minus_q_epoch14_20260809"
RESULTS_ROOT = ROOT / "docs/hardware/so101-input-ablation-20260809/u_minus_q_epoch14"

spec = importlib.util.spec_from_file_location("monitor_helpers", ROOT / "scripts/run_monitored_input_ablation.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load monitoring helpers")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    before = set(OUTPUT_ROOT.glob("gru_*"))
    command = [
        sys.executable, str(TRAIN),
        "--data_path", "outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.npz",
        "--dataset_manifest", "outputs/hardware/so101_pre_mpc/20260808_extension/model_a_workspace_58x15min_v2.manifest.json",
        "--robot_config", "configs/robots/so101.yaml",
        "--model_type", "gru", "--history_len", "16",
        "--target_mode", "delta_state", "--action_input_mode", "q_ref_minus_q",
        "--batch_size", "8192", "--micro_batch_size", "1024",
        "--epochs", "14", "--lr", "1e-4", "--seed", "10",
        "--num_workers", "2", "--pin_memory", "--amp",
        "--q_weight", "1.0", "--dq_weight", "1.0",
        "--loss_type", "huber", "--huber_delta", "1.0",
        "--rollout_loss_steps", "20", "--rollout_loss_weight", "0.025",
        "--rollout_loss_discount", "1.0", "--control_dt", "0.03333333333333333",
        "--test_group_ids", "43,44,45,46,47",
        "--validation_group_ids", "38,39,40,41,42",
        "--train_sample_stride", "2", "--val_sample_stride", "1",
        "--source_weights", "0:1,1:4", "--save_dir", str(OUTPUT_ROOT),
    ]
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT)
    run_dir = None
    seen_epoch = 0
    results = []
    while process.poll() is None:
        candidates = set(OUTPUT_ROOT.glob("gru_*")) - before
        if candidates:
            run_dir = sorted(candidates)[-1]
        if run_dir is not None:
            latest = run_dir / "latest_model.pt"
            epoch = helpers._load_checkpoint_epoch(latest) if latest.exists() else None
            if epoch is not None and epoch > seen_epoch and (run_dir / "normalizer.pt").exists():
                result = helpers._kappa_for_checkpoint(latest, run_dir / "normalizer.pt", [300, 400, 600])
                result.update(label="u_minus_q_epoch14", phase="fresh_epoch1_14")
                results.append(result)
                seen_epoch = epoch
                epoch_dir = RESULTS_ROOT / f"epoch_{epoch:03d}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                (epoch_dir / "kappa.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                k = result["kappa_h6"]
                print(f"[κ] epoch={epoch} mean={k['mean']:.6f} min={k['min']:.6f} max={k['max']:.6f}", flush=True)
        time.sleep(2.0)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    if seen_epoch != 14:
        raise RuntimeError(f"training exited at epoch {seen_epoch}, expected 14")
    (RESULTS_ROOT / "manifest.json").write_text(json.dumps({
        "epochs": 14, "action_input_mode": "q_ref_minus_q", "run_dir": str(run_dir.resolve()), "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS_ROOT / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
