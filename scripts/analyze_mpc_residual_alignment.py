#!/usr/bin/env python3
"""Measure whether MPC residuals provide physical lead compensation.

The diagnostic uses the *original* q_des reference, not the command sent to
the actuator.  A useful anticipatory residual should roughly satisfy
``r[t] ~= k * (q_des[t+1] - q_des[t]) / dt`` during motion.  The report keeps
planner-requested and actually-transmitted residuals separate because the
former diagnoses CEM/model intent while the latter diagnoses hardware effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _finite_stats(value: np.ndarray) -> dict[str, float | int]:
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    value = value[np.isfinite(value)]
    if value.size == 0:
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "count": int(value.size),
        "mean": float(np.mean(value)),
        "p50": float(np.percentile(value, 50.0)),
        "p95": float(np.percentile(value, 95.0)),
        "max": float(np.max(value)),
    }


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size != x.size or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _load_arrays(path: Path, reference_file: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    with np.load(path, allow_pickle=False) as archive:
        states_key = "actual_states" if "actual_states" in archive.files else "states"
        states = np.asarray(archive[states_key], dtype=np.float64)
        if "q_des" in archive.files:
            q_des = np.asarray(archive["q_des"], dtype=np.float64)
        elif reference_file is not None:
            q_des = np.asarray(np.load(reference_file), dtype=np.float64)
        else:
            raise KeyError("rollout has no q_des; pass --reference-file")
        if "planner_requested_residual" not in archive.files:
            raise KeyError("rollout has no planner_requested_residual")
        requested = np.asarray(archive["planner_requested_residual"], dtype=np.float64)
        active = None
        if "active_start_tick" in archive.files:
            active = int(np.asarray(archive["active_start_tick"]).reshape(-1)[0])
    if states.ndim != 2 or states.shape[1] < 5 or q_des.shape != (len(states), 5) or requested.shape != (len(states), 5):
        raise ValueError(f"incompatible rollout shapes: states={states.shape}, q_des={q_des.shape}, residual={requested.shape}")
    return states[:, :5], q_des, requested, active


def _series_report(
    residual: np.ndarray,
    dq_des: np.ndarray,
    *,
    active_start_tick: int,
    velocity_threshold: float,
    residual_threshold: float,
) -> dict[str, object]:
    n = min(len(residual), len(dq_des))
    residual, dq_des = residual[:n], dq_des[:n]
    active = np.arange(n) >= int(active_start_tick)
    finite = np.isfinite(residual).all(axis=1) & np.isfinite(dq_des).all(axis=1)
    motion = np.max(np.abs(dq_des), axis=1) >= float(velocity_threshold)
    base_mask = active & finite & motion
    per_joint: list[dict[str, object]] = []
    for joint, name in enumerate(JOINT_NAMES):
        mask = base_mask & (np.abs(dq_des[:, joint]) >= float(velocity_threshold)) & (np.abs(residual[:, joint]) >= float(residual_threshold))
        r = residual[mask, joint]
        v = dq_des[mask, joint]
        slope = float(np.dot(v, r) / max(np.dot(v, v), 1e-12)) if v.size else float("nan")
        same = np.sign(r) == np.sign(v)
        per_joint.append({
            "joint": name,
            "joint_index": joint,
            "samples": int(r.size),
            "pearson_corr": _corr(r, v),
            "lead_slope_s": slope,
            "sign_agreement": float(np.mean(same)) if same.size else float("nan"),
            "opposite_sign_rate": float(np.mean(~same)) if same.size else float("nan"),
            "residual_abs_rad": _finite_stats(np.abs(r)),
            "dq_des_abs_rad_s": _finite_stats(np.abs(v)),
        })
    return {
        "active_start_tick": int(active_start_tick),
        "motion_samples": int(np.sum(base_mask)),
        "velocity_threshold_rad_s": float(velocity_threshold),
        "residual_threshold_rad": float(residual_threshold),
        "per_joint": per_joint,
    }


def _markdown(report: dict[str, object]) -> str:
    lines = [
        "# MPC residual / reference-velocity alignment",
        "",
        f"rollout: `{report['rollout']}`",
        f"dt: `{report['control_dt_s']:.9f}` s; active_start_tick: `{report['active_start_tick']}`",
        "",
        "| joint | samples | corr(r,dq_des) | lead slope [s] | same-sign | opposite-sign |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["series"]["planner_requested"]["per_joint"]:  # type: ignore[index]
        lines.append(
            f"| {row['joint']} | {row['samples']} | {row['pearson_corr']:.4f} | "
            f"{row['lead_slope_s']:.4f} | {row['sign_agreement']:.3f} | {row['opposite_sign_rate']:.3f} |"
        )
    lines += [
        "",
        "`lead_slope_s` is the least-squares coefficient in "
        "`residual ~= lead_slope_s * dq_des`. Positive correlation and same-sign "
        "rate above 0.5 are necessary (not sufficient) evidence of anticipatory control.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--control-dt", type=float, default=1.0 / 30.0)
    parser.add_argument("--active-start-tick", type=int, default=None)
    parser.add_argument("--velocity-threshold-deg-s", type=float, default=1.0)
    parser.add_argument("--residual-threshold-deg", type=float, default=0.05)
    args = parser.parse_args()
    if args.control_dt <= 0.0:
        raise SystemExit("--control-dt must be positive")
    states, q_des, requested, inferred_active = _load_arrays(args.rollout, args.reference_file)
    active_start = inferred_active if args.active_start_tick is None and inferred_active is not None else int(args.active_start_tick or 0)
    # Forward difference matches the command at tick t: q_ref[t] is sent now,
    # and this is the desired motion it must anticipate before tick t+1.
    dq_des = np.empty_like(q_des)
    dq_des[:-1] = np.diff(q_des, axis=0) / float(args.control_dt)
    dq_des[-1] = dq_des[-2] if len(dq_des) > 1 else 0.0
    executed = np.full_like(requested, np.nan)
    with np.load(args.rollout, allow_pickle=False) as archive:
        if "actuator_q_ref" in archive.files:
            executed = np.asarray(archive["actuator_q_ref"], dtype=np.float64) - q_des
        elif "transmitted_q_ref" in archive.files:
            executed = np.asarray(archive["transmitted_q_ref"], dtype=np.float64) - q_des
    report: dict[str, object] = {
        "rollout": str(args.rollout.resolve()),
        "control_dt_s": float(args.control_dt),
        "active_start_tick": int(active_start),
        "reference_velocity_definition": "forward_difference_(q_des[t+1]-q_des[t])/dt",
        "series": {
            "planner_requested": _series_report(
                requested, dq_des, active_start_tick=active_start,
                velocity_threshold=np.deg2rad(args.velocity_threshold_deg_s),
                residual_threshold=np.deg2rad(args.residual_threshold_deg),
            ),
            "executed_qref_minus_qdes": _series_report(
                executed, dq_des, active_start_tick=active_start,
                velocity_threshold=np.deg2rad(args.velocity_threshold_deg_s),
                residual_threshold=np.deg2rad(args.residual_threshold_deg),
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "alignment.json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    (args.output_dir / "alignment.md").write_text(_markdown(report), encoding="utf-8")
    shoulder = report["series"]["planner_requested"]["per_joint"][0]  # type: ignore[index]
    print(json.dumps({"shoulder_pan": shoulder, "output_dir": str(args.output_dir.resolve())}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
