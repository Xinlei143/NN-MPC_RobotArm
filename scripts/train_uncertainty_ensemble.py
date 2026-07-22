"""Train the four replica GRUs used by the uncertainty-aware MPC ensemble.

The primary checkpoint remains untouched.  Each replica uses exactly the same
Model-A/C2 data and GRU hyperparameters, differing only in its random seed.
The baseline normalizer is frozen for every member so disagreement is measured
in a common state coordinate system.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise argparse.ArgumentTypeError("--replica_seeds must contain exactly four distinct integers")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train four fixed-seed GRU replicas for uncertainty-aware MPC.")
    parser.add_argument("--data_path", default="outputs/model_c/C2_train.npz")
    parser.add_argument("--baseline_checkpoint", default="outputs/checkpoints/gru_20260720_202923/best_model.pt")
    parser.add_argument("--baseline_normalizer", default="outputs/checkpoints/gru_20260720_202923/normalizer.pt")
    parser.add_argument("--output_root", default="outputs/uncertainty_ensemble")
    parser.add_argument("--replica_seeds", default="101,211,307,401", type=_parse_seeds)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None, help="Reserved for launch documentation; train_dynamics selects its device.")
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="Reuse the newest complete replica directory for a seed instead of retraining it.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_path = (ROOT / args.data_path).resolve()
    baseline_checkpoint = (ROOT / args.baseline_checkpoint).resolve()
    baseline_normalizer = (ROOT / args.baseline_normalizer).resolve()
    output_root = (ROOT / args.output_root).resolve()
    for path, label in ((data_path, "data_path"), (baseline_checkpoint, "baseline_checkpoint"), (baseline_normalizer, "baseline_normalizer")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "ensemble_manifest.json"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_baseline": {
            "checkpoint": str(baseline_checkpoint),
            "normalizer": str(baseline_normalizer),
            "checkpoint_sha256": _sha256(baseline_checkpoint),
            "normalizer_sha256": _sha256(baseline_normalizer),
        },
        "training_data": {"path": str(data_path), "sha256": _sha256(data_path)},
        "architecture": {"model_type": "gru", "history_len": 16, "target_mode": "delta_dq", "control_dt": 0.01},
        "replica_seeds": args.replica_seeds,
        "replicas": [],
    }
    for seed in args.replica_seeds:
        save_dir = output_root / f"seed_{seed}"
        if args.reuse_existing:
            completed = sorted(
                (
                    path.resolve()
                    for path in save_dir.glob("gru_*")
                    if path.is_dir()
                    and (path / "best_model.pt").is_file()
                    and (path / "normalizer.pt").is_file()
                ),
                key=lambda path: path.stat().st_mtime,
            )
            if completed:
                checkpoint_dir = completed[-1]
                checkpoint = checkpoint_dir / "best_model.pt"
                normalizer = checkpoint_dir / "normalizer.pt"
                print(f"Reusing completed seed {seed}: {checkpoint_dir}")
                manifest["replicas"].append(
                    {
                        "seed": seed,
                        "checkpoint": str(checkpoint),
                        "normalizer": str(normalizer),
                        "checkpoint_sha256": _sha256(checkpoint),
                        "normalizer_sha256": _sha256(normalizer),
                    }
                )
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                continue
        command = [
            sys.executable,
            str(ROOT / "dynamics_modeling" / "scripts" / "train_dynamics.py"),
            "--data_path", str(data_path),
            "--model_type", "gru",
            "--history_len", "16",
            "--target_mode", "delta_dq",
            "--control_dt", "0.01",
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--rollout_loss_steps", "1",
            "--rollout_loss_weight", "0.0",
            "--normalizer_path", str(baseline_normalizer),
            "--freeze_normalizer",
            "--seed", str(seed),
            "--save_dir", str(save_dir),
        ]
        print(" ".join(command))
        if args.dry_run:
            continue
        before = {path.resolve() for path in save_dir.glob("gru_*") if path.is_dir()}
        subprocess.run(command, check=True, cwd=ROOT)
        created = sorted({path.resolve() for path in save_dir.glob("gru_*") if path.is_dir()} - before)
        if len(created) != 1:
            raise RuntimeError(f"seed {seed}: expected one new checkpoint directory, found {created}")
        checkpoint_dir = created[0]
        checkpoint = checkpoint_dir / "best_model.pt"
        normalizer = checkpoint_dir / "normalizer.pt"
        if not checkpoint.is_file() or not normalizer.is_file():
            raise RuntimeError(f"seed {seed}: checkpoint output is incomplete: {checkpoint_dir}")
        manifest["replicas"].append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "normalizer": str(normalizer),
                "checkpoint_sha256": _sha256(checkpoint),
                "normalizer_sha256": _sha256(normalizer),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run:
        manifest["dry_run"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
