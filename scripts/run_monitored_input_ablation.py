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
    """Measure the established six-step command sensitivity on CPU."""
    # Imported here so the launcher can show train output immediately and so
    # the tiny kappa probe never allocates training-sized CUDA tensors.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "dynamics_modeling"))
    from neural_dynamics.integration import reconstruct_next_state
    from neural_dynamics.rollout import load_dynamics_bundle
    from mpc.robot_config import load_robot_spec

    with np.load(ROLLOUT, allow_pickle=False) as recorded:
        executed_tokens = np.asarray(recorded["executed_tokens"], dtype=np.float32)
        q_des = np.asarray(recorded["q_des"], dtype=np.float32)[:, :5]
    bundle = load_dynamics_bundle(
        checkpoint_path,
        normalizer_path,
        "gru",
        5,
        torch.device("cpu"),
        expected_robot_spec=load_robot_spec("configs/robots/so101.yaml"),
    )
    perturbation = float(np.deg2rad(0.5))
    rows: list[dict[str, object]] = []
    for tick in ticks:
        history_window = executed_tokens[tick - bundle.history_len + 1:tick + 1]
        history_states = history_window[:, :10]
        history_actions = history_window[:-1, 10:15]
        commands = q_des[tick + 1:tick + 7]

        def rollout(command_sequence: np.ndarray) -> np.ndarray:
            predicted_state = torch.as_tensor(history_states[-1:], device=bundle.device)
            fixed_history = torch.cat([
                torch.as_tensor(history_states[-(bundle.history_len - 1):], device=bundle.device),
                torch.as_tensor(history_actions[-(bundle.history_len - 1):], device=bundle.device),
            ], dim=-1)
            predictions = []
            for command in command_sequence:
                current = torch.cat([
                    predicted_state,
                    torch.as_tensor(command, device=bundle.device).view(1, -1),
                ], dim=-1)
                model_input = bundle.normalizer.normalize_sequence_input(
                    torch.cat([fixed_history, current], dim=0).unsqueeze(0), bundle.state_dim
                )
                with torch.no_grad():
                    predicted_target = bundle.normalizer.denormalize_delta(bundle.model(model_input))
                predicted_state = reconstruct_next_state(
                    predicted_state, predicted_target, bundle.target_mode, bundle.control_dt, 5
                )
                predictions.append(predicted_state[0])
            return torch.stack(predictions).cpu().numpy()

        baseline = rollout(commands)[:, :5]
        perturbed = rollout(commands + perturbation)[:, :5]
        response = perturbed - baseline
        rows.append({
            "tick": tick,
            "kappa_h6": float(np.linalg.norm(response[-1]) / np.linalg.norm(np.full(5, perturbation))),
            "per_joint_terminal_gain": (np.abs(response[-1]) / perturbation).tolist(),
            "per_joint_accumulated_gain": (np.linalg.norm(response, axis=0) / perturbation).tolist(),
        })
    values = [float(row["kappa_h6"]) for row in rows]
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "epoch": _load_checkpoint_epoch(checkpoint_path),
        "action_input_mode": bundle.action_input_mode,
        "kappa_h6": {"rows": rows, "mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))},
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
