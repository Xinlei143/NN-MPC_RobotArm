#!/usr/bin/env python3
"""Run the isolated SO101 ThreadedAsync injected-delay study.

The study deliberately has one controller and one changed runtime factor:
``threaded_asap`` with the frozen SO101 NN-MPC parameters and a deterministic
wall-clock delay inserted immediately before packet publication.  It never
changes the formal 63-trial output root, residual authority, checkpoint,
reference, CEM settings, or executable projector.

There are two stages:

``shadow``
    Run the same planner on the three frozen pre-MPC calibration references
    with the injected delay.  The resulting latency/OOD artifacts are used to
    derive the delay-specific anticipation-step calibration.
``active``
    Run one held-out ellipse/fast NN-MPC trial under either ``baseline``
    (0 ms, the original calibration) or ``delay33`` (the injected calibration).

Use ``--dry-run`` first.  Physical execution still requires the existing
operator flags, startup gates, E-stop, and independent mechanical support.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_so101_paper_trial import (
    _common_args,
    _load_protocol,
    _path,
    _reference_for,
    _sha256,
    _trial_specs,
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_stress(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"delay-stress protocol must be a mapping: {path}")
    return payload


def _replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError:
        command.extend([option, value])
        return
    if index + 1 >= len(command):
        raise ValueError(f"malformed command: {option} has no value")
    command[index + 1] = value


def _remove_option(command: list[str], option: str) -> None:
    while option in command:
        index = command.index(option)
        del command[index:index + 2]


def _artifact_hashes(base: dict, reference: Path, manifest: Path, delay: Path | None,
                     ood: Path | None) -> dict[str, str]:
    paths = {
        "hardware_config": _path(base["hardware_config"]),
        "checkpoint": _path(base["checkpoint"]),
        "normalizer": _path(base["normalizer"]),
        "reference": reference,
        "reference_manifest": manifest,
    }
    if delay is not None:
        paths["delay_calibration"] = delay
    if ood is not None:
        paths["ood_envelope"] = ood
    return {name: _sha256(path) for name, path in paths.items()}


def _active_condition(stress: dict, base: dict, condition: str) -> dict:
    if condition == "baseline":
        return {
            "label": "baseline",
            "injected_delay_ms": 0.0,
            "delay_calibration": _path(base["delay_calibration"]),
            "ood_envelope": _path(base["ood_envelope"]),
        }
    if condition == "delay33":
        return {
            "label": "delay33",
            "injected_delay_ms": float(stress["injected_planner_delay_ms"]),
            "delay_calibration": _path(stress["delay_calibration"]),
            "ood_envelope": _path(stress["ood_envelope"]),
        }
    raise ValueError(f"unknown active condition: {condition!r}")


def _build_command(base: dict, stress: dict, *, stage: str, output: Path,
                   reference: Path, manifest: Path, condition: dict,
                   shadow_provisional_delay_steps: int | None = None,
                   check_condition_files: bool = True) -> list[str]:
    spec = {
        "controller": "nn_mpc",
        "repeat_index": 0,
        "trial_id": output.name,
    }
    command = _common_args(base, spec, reference, manifest, output)
    _replace_option(command, "--delay_protocol", "full")
    _replace_option(command, "--injected-planner-delay-ms", f"{condition['injected_delay_ms']:.6f}")
    # The injected-delay study must prove planner readiness before it enables
    # torque.  This is a safety/preflight ordering guard, not a controller
    # parameter and does not alter the planner, CEM, or executable projector.
    if "--planner-preflight-before-motion" not in command:
        command.append("--planner-preflight-before-motion")

    if stage == "shadow":
        _replace_option(command, "--real-mode", "shadow_mpc")
        _remove_option(command, "--delay-calibration")
        _remove_option(command, "--ood-envelope")
        if shadow_provisional_delay_steps is None:
            raise ValueError("shadow command requires provisional anticipation delay steps")
        _replace_option(command, "--anticipation_delay_steps", str(int(shadow_provisional_delay_steps)))
    else:
        if check_condition_files and not condition["delay_calibration"].exists():
            raise FileNotFoundError(
                f"active condition calibration is missing: {condition['delay_calibration']}"
            )
        if check_condition_files and not condition["ood_envelope"].exists():
            raise FileNotFoundError(
                f"active condition OOD envelope is missing: {condition['ood_envelope']}"
            )
        _replace_option(command, "--delay-calibration", str(condition["delay_calibration"]))
        _replace_option(command, "--ood-envelope", str(condition["ood_envelope"]))
    return command


def _reference_for_shadow(stress: dict, label: str) -> tuple[Path, Path]:
    root = _path(stress["shadow_reference_root"])
    manifest = _path(stress["shadow_reference_manifest"])
    reference = root / label / "joint_reference_mpc.npz"
    if not manifest.exists():
        raise FileNotFoundError(f"shadow reference manifest not found: {manifest}")
    if not reference.exists():
        raise FileNotFoundError(f"shadow reference not found: {reference}")
    return reference, manifest


def _run(command: list[str], output: Path, record: dict) -> int:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "trial_manifest.json"
    manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log_path = output / "trial.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/experiments/so101_threaded_delay_stress_20260810.yaml")
    parser.add_argument("--stage", choices=["shadow", "active"], required=True)
    parser.add_argument("--condition", choices=["baseline", "delay33"], default="delay33")
    parser.add_argument("--trial-id", help="Held-out NN-MPC id, e.g. ellipse_fast_p0_nn_mpc (active only).")
    parser.add_argument("--shadow-label", help="Frozen calibration reference label (shadow only).")
    parser.add_argument("--confirm", help="Must equal the printed condition token before motion is enabled.")
    parser.add_argument("--operator", default="unspecified")
    parser.add_argument("--notes", default="")
    parser.add_argument("--retry-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Initialize the torque-disabled planner preflight and exit without enabling robot motion.",
    )
    args = parser.parse_args()

    if args.retry_index < 0:
        raise SystemExit("--retry-index must be non-negative")
    stress_path = _path(args.protocol)
    stress = _load_stress(stress_path)
    base_path = _path(stress["base_protocol"])
    base = _load_protocol(base_path)
    if base.get("protocol_id") != stress.get("base_protocol_id"):
        raise SystemExit("base protocol identity does not match delay-stress protocol")
    if not math.isclose(float(stress["injected_planner_delay_ms"]), 33.0, abs_tol=1e-6):
        raise SystemExit("this protocol is frozen to a 33 ms injected delay")
    if float(base["final_mpc"]["residual_max_rad"]) != 0.017453292:
        raise SystemExit("delay-stress requires the frozen formal 1 degree residual authority")
    condition = {
        "label": "delay33",
        "injected_delay_ms": float(stress["injected_planner_delay_ms"]),
        "delay_calibration": None,
        "ood_envelope": None,
    }

    if args.stage == "shadow":
        if args.condition != "delay33":
            raise SystemExit("shadow calibration is only defined for the injected delay33 condition")
        labels = set(stress["shadow_reference_labels"])
        if args.shadow_label is None or args.shadow_label not in labels:
            raise SystemExit(f"--shadow-label must be one of {sorted(labels)}")
        reference, manifest = _reference_for_shadow(stress, args.shadow_label)
        token = f"shadow__{args.shadow_label}__delay{condition['injected_delay_ms']:g}ms"
        output_root = _path(stress["shadow_output_root"])
        spec_label = args.shadow_label
        command = _build_command(
            base, stress, stage="shadow", output=output_root / token,
            reference=reference, manifest=manifest, condition=condition,
            shadow_provisional_delay_steps=int(stress["shadow_provisional_delay_steps"]),
            check_condition_files=not args.dry_run,
        )
        output = output_root / token
    else:
        if args.trial_id is None:
            raise SystemExit("active stage requires --trial-id")
        specs = _trial_specs(base)
        if args.trial_id not in specs:
            raise SystemExit(f"unknown trial id {args.trial_id!r}")
        spec = specs[args.trial_id]
        if spec["family"] != "heldout" or spec["controller"] != "nn_mpc":
            raise SystemExit("delay stress is restricted to held-out NN-MPC trials")
        condition = _active_condition(stress, base, args.condition)
        reference, manifest = _reference_for(base, spec)
        token = f"{args.trial_id}__{condition['label']}"
        output_root = _path(stress["active_output_root"])
        output = output_root / token
        spec_label = args.trial_id
        command = _build_command(base, stress, stage="active", output=output,
                                 reference=reference, manifest=manifest,
                                 condition=condition,
                                 check_condition_files=not args.dry_run)

    if args.retry_index:
        output = output.parent / f"{output.name}_retry{args.retry_index}"
        command = [str(output) if value == str(output.parent / token) else value for value in command]
        # The save directory is the only command argument that points at the
        # original output; replace it explicitly after the generic rewrite.
        _replace_option(command, "--save_dir", str(output))
    if args.preflight_only:
        output = output.parent / f"{output.name}__preflight"
        _replace_option(command, "--save_dir", str(output))
        command.append("--planner-preflight-only")
    if not args.dry_run and not args.preflight_only and args.confirm != token:
        raise SystemExit(f"motion requires --confirm exactly equal to {token!r}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}; use --retry-index")

    delay_path = condition.get("delay_calibration")
    ood_path = condition.get("ood_envelope")
    record = {
        "protocol": base["protocol_id"],
        "delay_stress_protocol": stress.get("protocol_id"),
        "protocol_file": str(base_path),
        "delay_stress_protocol_file": str(stress_path),
        "stage": args.stage,
        "condition": condition["label"],
        "trial_or_reference": spec_label,
        "injected_planner_delay_ms": condition["injected_delay_ms"],
        "delay_calibration": None if delay_path is None else str(delay_path),
        "ood_envelope": None if ood_path is None else str(ood_path),
        "output_dir": str(output),
        "command": command,
        "operator": args.operator,
        "notes": args.notes,
        "created_utc": _utc(),
        "formal_output_root_untouched": str(_path(base["output_root"])),
        "status": "prepared",
        "preflight_only": bool(args.preflight_only),
    }
    if not args.dry_run:
        hashes = _artifact_hashes(base, reference, manifest, delay_path, ood_path)
        expected = base.get("artifacts", {})
        for actual_name, expected_name in (("checkpoint", "checkpoint_sha256"),
                                           ("normalizer", "normalizer_sha256")):
            expected_hash = expected.get(expected_name)
            if expected_hash is not None and hashes[actual_name] != expected_hash:
                raise SystemExit(
                    f"frozen {actual_name} identity mismatch: "
                    f"{hashes[actual_name]} != {expected_hash}"
                )
        record["artifacts"] = hashes
    print(json.dumps(record, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    return _run(command, output, record)


if __name__ == "__main__":
    raise SystemExit(main())
