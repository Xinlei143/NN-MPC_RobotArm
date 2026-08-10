#!/usr/bin/env python3
"""Run exactly one frozen SO101 paper trial.

The protocol YAML owns every controller parameter.  This entry point accepts
only a trial id and operator confirmation, which prevents per-trajectory
parameter changes and accidental output overwrites.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_protocol(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"protocol must be a mapping: {path}")
    return payload


def _trial_specs(protocol: dict) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    orders = protocol["controller_order_by_repeat"]
    for repeat_index, label in enumerate(protocol["circle_labels"]):
        for controller in orders[repeat_index]:
            trial_id = f"circle_{label.split('_')[-1]}_{controller}"
            specs[trial_id] = {
                "trial_id": trial_id,
                "family": "circle",
                "shape": "circle",
                "speed": "development",
                "phase_index": repeat_index,
                "reference_label": label,
                "controller": controller,
                "repeat_index": repeat_index,
            }
    for shape in protocol["heldout_shapes"]:
        for speed in protocol["speed_labels"]:
            for repeat_index in range(int(protocol["repeat_count"])):
                label = f"{shape}_{speed}_p{repeat_index}"
                for controller in orders[repeat_index]:
                    trial_id = f"{label}_{controller}"
                    specs[trial_id] = {
                        "trial_id": trial_id,
                        "family": "heldout",
                        "shape": shape,
                        "speed": speed,
                        "phase_index": repeat_index,
                        "reference_label": label,
                        "controller": controller,
                        "repeat_index": repeat_index,
                    }
    return specs


def _reference_for(protocol: dict, spec: dict) -> tuple[Path, Path]:
    if spec["family"] == "circle":
        root = _path(protocol["circle_reference_root"])
        manifest = _path(protocol["circle_reference_manifest"])
    else:
        root = _path(protocol["heldout_reference_root"])
        manifest = _path(protocol["heldout_reference_manifest"])
    reference = root / spec["reference_label"] / "joint_reference_mpc.npz"
    if not manifest.exists():
        raise FileNotFoundError(f"reference manifest not found: {manifest}")
    if not reference.exists():
        raise FileNotFoundError(f"reference file not found: {reference}")
    return reference, manifest


def _common_args(protocol: dict, spec: dict, reference: Path, manifest: Path, output: Path) -> list[str]:
    controller = spec["controller"]
    common = [
        "--hardware-config", str(_path(protocol["hardware_config"])),
        "--reference-mode", "joint_file",
        "--reference-file", str(reference),
        "--reference-manifest", str(manifest),
        "--home-tolerance-deg", str(protocol["home_tolerance_deg"]),
        "--enable-motion", "--operator-supported-shutdown",
    ]
    if controller == "direct":
        return [sys.executable, str(ROOT / "scripts/run_real_direct_control.py"), *common,
                "--preview-steps", "0", "--output", str(output / "rollout.npz")]
    if controller == "preview6":
        return [sys.executable, str(ROOT / "scripts/run_real_direct_control.py"), *common,
                "--preview-steps", "6", "--output", str(output / "rollout.npz")]
    if controller != "nn_mpc":
        raise ValueError(f"unknown controller {controller!r}")

    mpc = protocol["final_mpc"]
    weights = mpc["weights"]
    seed = mpc["seed_by_repeat"][int(spec["repeat_index"])]
    args = [
        sys.executable, str(ROOT / "scripts/run_real_cem_mpc.py"),
        "--hardware-config", str(_path(protocol["hardware_config"])),
        "--real-mode", str(mpc["real_mode"]),
        "--checkpoint", str(_path(protocol["checkpoint"])),
        "--normalizer", str(_path(protocol["normalizer"])),
        "--model_type", str(mpc["model_type"]),
        "--history_len", str(mpc["history_len"]),
        "--reference_mode", "joint_file",
        "--reference_file", str(reference),
        "--reference-manifest", str(manifest),
        "--delay-calibration", str(_path(protocol["delay_calibration"])),
        "--ood-envelope", str(_path(protocol["ood_envelope"])),
        "--home-tolerance-deg", str(protocol["home_tolerance_deg"]),
        "--horizon", str(mpc["horizon"]),
        "--num_samples", str(mpc["num_samples"]),
        "--cem_iters", str(mpc["cem_iters"]),
        "--rollout_batch_size", str(mpc["rollout_batch_size"]),
        "--planner_projection", str(mpc["planner_projection"]),
        "--planner_projection_backend", str(mpc["planner_projection_backend"]),
        "--planner_projection_strategy", str(mpc["planner_projection_strategy"]),
        "--executable_rollout_backend", str(mpc["executable_rollout_backend"]),
        "--mpc_warmup_plans", str(mpc["mpc_warmup_plans"]),
        "--mpc_preview_nominal_steps", str(mpc["mpc_preview_nominal_steps"]),
        "--nominal_command_semantics", str(mpc["nominal_command_semantics"]),
        "--asap_history_mode", str(mpc["asap_history_mode"]),
        "--asap_snapshot_mode", str(mpc["asap_snapshot_mode"]),
        "--mpc_policy", str(mpc["mpc_policy"]),
        "--cem_execute", str(mpc["cem_execute"]),
        "--analytical_preview_steps", str(mpc["analytical_preview_steps"]),
        "--cost_profile", str(mpc["cost_profile"]),
        "--residual_max", str(mpc["residual_max_rad"]),
        "--feedback_kq", str(mpc["feedback_kq"]),
        "--feedback_kdq", str(mpc["feedback_kdq"]),
        "--feedback_max", str(mpc["feedback_max_rad"]),
        "--device", "cuda",
        "--seed", str(seed),
        "--save_dir", str(output),
        "--enable-motion", "--operator-supported-shutdown",
    ]
    for name, value in weights.items():
        args.extend([f"--{name}", str(value)])
    return args


def _artifact_hashes(protocol: dict, reference: Path, manifest: Path) -> dict[str, str]:
    paths = {
        "hardware_config": _path(protocol["hardware_config"]),
        "checkpoint": _path(protocol["checkpoint"]),
        "normalizer": _path(protocol["normalizer"]),
        "delay_calibration": _path(protocol["delay_calibration"]),
        "ood_envelope": _path(protocol["ood_envelope"]),
        "reference": reference,
        "reference_manifest": manifest,
    }
    return {name: _sha256(path) for name, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/experiments/so101_final_paper_20260810.yaml")
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--confirm", help="Must equal --trial-id before motion is enabled.")
    parser.add_argument("--operator", default="unspecified")
    parser.add_argument("--notes", default="")
    parser.add_argument("--retry-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    protocol_path = _path(args.protocol)
    protocol = _load_protocol(protocol_path)
    specs = _trial_specs(protocol)
    if args.trial_id not in specs:
        choices = ", ".join(sorted(specs)[:8])
        raise SystemExit(f"unknown trial id {args.trial_id!r}; examples: {choices}")
    if args.retry_index < 0:
        raise SystemExit("--retry-index must be non-negative")
    if not args.dry_run and (args.confirm != args.trial_id):
        raise SystemExit("motion requires --confirm exactly equal to --trial-id")
    spec = specs[args.trial_id]
    reference, manifest = _reference_for(protocol, spec)
    root = _path(protocol["output_root"])
    output = root / args.trial_id
    if args.retry_index:
        output = root / f"{args.trial_id}_retry{args.retry_index}"
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}; use --retry-index")
    command = _common_args(protocol, spec, reference, manifest, output)
    if args.dry_run:
        print(json.dumps({"trial": spec, "output": str(output), "command": command}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "protocol": protocol["protocol_id"],
        "protocol_file": str(protocol_path),
        "trial": spec,
        "output_dir": str(output),
        "operator": args.operator,
        "notes": args.notes,
        "created_utc": _utc(),
        "command": command,
        "artifacts": _artifact_hashes(protocol, reference, manifest),
        "dry_run": bool(args.dry_run),
        "status": "prepared",
    }
    manifest_path = output / "trial_manifest.json"
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"trial": spec, "output": str(output), "command": command}, indent=2))

    log_path = output / "trial.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    record.update({
        "finished_utc": _utc(),
        "elapsed_s": time.monotonic() - started,
        "return_code": int(return_code),
        "status": "completed" if return_code == 0 else "failed",
    })
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return int(return_code)


if __name__ == "__main__":
    raise SystemExit(main())
