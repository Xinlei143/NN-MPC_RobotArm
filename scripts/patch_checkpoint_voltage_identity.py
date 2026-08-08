#!/usr/bin/env python3
"""Metadata-only plant-identity patch for SO101 real MPC artifacts.

Fixes the runtime identity gate after retraining on the v2 corpus: the v2
canonical manifest inherits the v1 plant identity (hardware_config_sha256
984e846, voltage thresholds null), but the connected robot now runs with
4460c456 and the frozen mid-range voltage set.  verify_real_artifact_identity
requires every runtime-declared plant field to be present and identical, so
these three keys must be brought in line.

We patch ONLY the plant_identity keys.  Model weights, optimizer/scaler state
and every other provenance field are left byte-identical.  A sidecar JSON
records before/after hashes and the old/new values for audit, mirroring the
identity_voltage_20260808.json convention from the first patch.

Usage:
  python scripts/patch_checkpoint_voltage_identity.py \
    --checkpoint <dir-or-file>/best_rollout_model.pt \
    --normalizer  <dir>/normalizer.pt \
    --hardware-config configs/hardware/so101_follower.local.yaml \
    [--sidecar <out.json>] [--dry-run]

--checkpoint may be a directory (patches best_model.pt, best_rollout_model.pt,
latest_model.pt) or a single .pt file.  The default sidecar is written next to
the checkpoint artifact as identity_voltage_<YYYYMMDD>.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_runtime.artifacts import verify_real_artifact_identity
from robot_runtime.config import load_hardware_config

PATCHED_KEYS = ("hardware_config_sha256", "voltage_warning", "voltage_hard")
CHECKPOINT_FILES = ("best_model.pt", "best_rollout_model.pt", "latest_model.pt")


def file_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plant_identity_of(artifact: dict, label: str) -> dict:
    if label == "checkpoint":
        config = artifact.get("config", {})
        pi = config.get("plant_identity")
    else:
        meta = artifact.get("metadata", {}) if isinstance(artifact, dict) else getattr(artifact, "metadata", {})
        pi = meta.get("plant_identity")
    if not isinstance(pi, dict):
        raise ValueError(f"{label} has no plant_identity mapping; refusing to patch")
    return pi


def _write_plant_identity(artifact: dict, label: str, pi: dict) -> None:
    if label == "checkpoint":
        artifact["config"]["plant_identity"] = pi
    elif isinstance(artifact, dict):
        artifact["metadata"]["plant_identity"] = pi
    else:
        artifact.metadata["plant_identity"] = pi


def _patch_file(path: Path, new_values: dict, dry_run: bool, label: str) -> dict:
    before = file_sha256(path)
    if label == "checkpoint" and path.name not in CHECKPOINT_FILES:
        raise ValueError(f"unexpected checkpoint filename {path.name}; expected one of {CHECKPOINT_FILES}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    pi = _plant_identity_of(artifact, label)
    old_values = {key: copy.deepcopy(pi.get(key)) for key in PATCHED_KEYS}
    for key in PATCHED_KEYS:
        pi[key] = copy.deepcopy(new_values[key])
    if dry_run:
        return {"file": str(path), "dry_run": True, "sha256_before": before,
                "old": old_values, "new": {k: new_values[k] for k in PATCHED_KEYS}}
    temp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(artifact, temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    after = file_sha256(path)
    return {"file": str(path), "sha256_before": before, "sha256_after": after,
            "old": old_values, "new": {k: new_values[k] for k in PATCHED_KEYS}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pt file or directory")
    parser.add_argument("--normalizer", required=True, help="normalizer .pt file")
    parser.add_argument("--hardware-config", required=True, help="runtime SO101 hardware config")
    parser.add_argument("--sidecar", default=None, help="output sidecar json (default: identity_voltage_<date>.json next to checkpoint)")
    parser.add_argument("--dry-run", action="store_true", help="print what would change without writing")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    normalizer_path = Path(args.normalizer).expanduser().resolve()
    if not normalizer_path.is_file():
        raise SystemExit(f"normalizer file does not exist: {normalizer_path}")
    hardware = load_hardware_config(Path(args.hardware_config).expanduser().resolve())
    new_pi = hardware.plant_identity()
    new_values = {key: copy.deepcopy(new_pi[key]) for key in PATCHED_KEYS}

    targets = []
    if checkpoint_path.is_dir():
        for name in CHECKPOINT_FILES:
            candidate = checkpoint_path / name
            if candidate.is_file():
                targets.append(candidate)
        if not targets:
            raise SystemExit(f"no checkpoint files ({CHECKPOINT_FILES}) in {checkpoint_path}")
    elif checkpoint_path.is_file():
        if checkpoint_path.name not in CHECKPOINT_FILES:
            raise SystemExit(f"checkpoint filename must be one of {CHECKPOINT_FILES}; got {checkpoint_path.name}")
        targets.append(checkpoint_path)
    else:
        raise SystemExit(f"checkpoint path does not exist: {checkpoint_path}")

    if not args.dry_run:
        # Sidecar written only after all patches succeed; write sidecar last.
        if args.sidecar:
            sidecar_path = Path(args.sidecar).expanduser().resolve()
        elif checkpoint_path.is_dir():
            sidecar_path = checkpoint_path / f"identity_voltage_{datetime.now().strftime('%Y%m%d')}.json"
        else:
            sidecar_path = checkpoint_path.parent / f"identity_voltage_{datetime.now().strftime('%Y%m%d')}.json"
        if sidecar_path.exists():
            raise SystemExit(f"refusing to overwrite existing sidecar: {sidecar_path}")

    records = []
    for target in targets:
        records.append(_patch_file(target, new_values, args.dry_run, "checkpoint"))
    records.append(_patch_file(normalizer_path, new_values, args.dry_run, "normalizer"))

    if args.dry_run:
        for record in records:
            print(json.dumps(record, indent=2, sort_keys=True))
        return

    # Verify the patched artifacts pass the runtime identity gate.
    if checkpoint_path.is_dir():
        verify_path = checkpoint_path / "best_rollout_model.pt"
    else:
        verify_path = checkpoint_path
    verify_real_artifact_identity(verify_path, normalizer_path, hardware.plant_identity())
    print(f"✓ verify_real_artifact_identity passed for {verify_path.name} and {normalizer_path.name}")

    payload = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "decision": ("operator-approved mid-range voltage set; metadata-only plant_identity update "
                     "(weights untouched) after v2 retraining"),
        "changed_keys": list(PATCHED_KEYS),
        "old": {key: copy.deepcopy(new_pi.get(key)) for key in PATCHED_KEYS},  # placeholder, replaced below
        "new": new_values,
        "artifacts": records,
    }
    payload["old"] = records[0]["old"]
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"✓ wrote sidecar: {sidecar_path}")


if __name__ == "__main__":
    main()
