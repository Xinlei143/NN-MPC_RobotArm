#!/usr/bin/env python3
"""Aggregate Direct-IK preview/lead sweep runs against the same q_des target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _run_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=ROLLOUT_NPZ")
    label, path = value.split("=", 1)
    return label, Path(path)


def _load_reference(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        return np.asarray(loaded, dtype=np.float32)
    with loaded:
        for key in ("q_des", "q_des_ctrl", "joint_reference", "reference"):
            if key in loaded.files:
                return np.asarray(loaded[key], dtype=np.float32)
        if len(loaded.files) == 1:
            return np.asarray(loaded[loaded.files[0]], dtype=np.float32)
    raise KeyError(f"{path}: no q_des array")


def _load_run(path: Path, reference: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        states_key = "actual_states" if "actual_states" in archive.files else "states"
        command_key = "actuator_q_ref" if "actuator_q_ref" in archive.files else (
            "transmitted_q_ref" if "transmitted_q_ref" in archive.files else "actions"
        )
        states = np.asarray(archive[states_key], dtype=np.float64)
        commands = np.asarray(archive[command_key], dtype=np.float64)
        q_des = np.asarray(archive["q_des"], dtype=np.float64) if "q_des" in archive.files else None
    if q_des is None:
        if reference is None:
            raise KeyError(f"{path}: no q_des; pass --reference-file")
        q_des = np.asarray(reference, dtype=np.float64)
    if q_des.shape[0] >= len(states) and q_des.shape[1:] == (5,):
        q_des = q_des[:len(states)]
    if states.ndim != 2 or states.shape[1] < 5 or commands.shape != (len(states), 5) or q_des.shape != (len(states), 5):
        raise ValueError(f"{path}: incompatible shapes states={states.shape}, commands={commands.shape}, q_des={q_des.shape}")
    metadata: dict[str, object] = {}
    manifest = path.with_suffix(".manifest.json")
    if manifest.exists():
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    return states[:, :5], commands, q_des, metadata


def _metrics(states: np.ndarray, commands: np.ndarray, q_des: np.ndarray, dt: float) -> dict[str, object]:
    error = states - q_des
    command_error = commands - q_des
    return {
        "samples": int(len(states)),
        "position_rmse_rad": float(np.sqrt(np.mean(np.square(error)))),
        "position_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(error))))),
        "max_abs_error_deg": float(np.rad2deg(np.max(np.abs(error)))),
        "per_joint_rmse_deg": [float(np.rad2deg(np.sqrt(np.mean(np.square(error[:, j]))))) for j in range(5)],
        "per_joint_bias_deg": [float(np.rad2deg(np.mean(error[:, j]))) for j in range(5)],
        "command_offset_p95_deg": [float(np.rad2deg(np.percentile(np.abs(command_error[:, j]), 95.0))) for j in range(5)],
        "command_velocity_p95_deg_s": [float(np.rad2deg(np.percentile(np.abs(np.diff(commands[:, j], prepend=commands[:1, j])) / dt, 95.0))) for j in range(5)],
    }


def _markdown(report: dict[str, object]) -> str:
    lines = ["# Direct preview / lead sweep", "", "| run | RMSE [deg] | max [deg] | pan [deg] | lift [deg] | elbow [deg] | wrist flex [deg] | wrist roll [deg] |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, row in report["runs"].items():  # type: ignore[index]
        per = row["metrics"]["per_joint_rmse_deg"]
        lines.append(f"| {label} | {row['metrics']['position_rmse_deg']:.4f} | {row['metrics']['max_abs_error_deg']:.4f} | " + " | ".join(f"{v:.4f}" for v in per) + " |")
    lines += ["", "RMSE is always evaluated against the original q_des, not against the previewed command.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_run_arg, metavar="LABEL=ROLLOUT_NPZ")
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--control-dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    reference = _load_reference(args.reference_file) if args.reference_file else None
    report: dict[str, object] = {"control_dt_s": float(args.control_dt), "runs": {}}
    for label, path in args.run:
        states, commands, q_des, metadata = _load_run(path, reference)
        report["runs"][label] = {"rollout": str(path.resolve()), "metadata": metadata, "metrics": _metrics(states, commands, q_des, args.control_dt)}  # type: ignore[index]
    rows = sorted(report["runs"].items(), key=lambda item: item[1]["metrics"]["position_rmse_rad"])  # type: ignore[index]
    report["best_by_rmse"] = rows[0][0] if rows else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preview_sweep.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (args.output_dir / "preview_sweep.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"best_by_rmse": report["best_by_rmse"], "runs": report["runs"]}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
