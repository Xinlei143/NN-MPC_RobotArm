#!/usr/bin/env python3
"""Offline audit of the fixed real SO101 q/dq state definition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary(values: np.ndarray) -> dict[str, list[float]]:
    return {name: np.percentile(values, percentile, axis=0).tolist()
            for name, percentile in (("p01", 1), ("p50", 50), ("p99", 99), ("min", 0), ("max", 100))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit recorded causal SO101 q/dq estimates; Present_Velocity remains diagnostic only.")
    parser.add_argument("dataset")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = np.load(args.dataset)
    required = {"measured_q", "estimated_dq", "raw_dq", "raw_present_velocity", "actual_dt", "state_estimator_version"}
    missing = required - set(data.files)
    if missing:
        raise SystemExit(f"missing state-estimator fields: {sorted(missing)}")
    q, dq, raw_dq = data["measured_q"], data["estimated_dq"], data["raw_dq"]
    finite = bool(np.all(np.isfinite(q)) and np.all(np.isfinite(dq)) and np.all(np.isfinite(raw_dq)))
    reset_rows = np.flatnonzero(np.r_[True, np.diff(data["estimator_generation"]) != 0])
    report = {
        "samples": int(len(q)), "finite_q_dq": finite,
        "state_estimator_versions": sorted(set(np.asarray(data["state_estimator_version"]).astype(str))),
        "estimator_reset_rows": reset_rows.tolist(), "q_rad": _summary(q),
        "estimated_dq_rad_s": _summary(dq), "raw_dq_rad_s": _summary(raw_dq),
        "present_velocity_raw_diagnostic": _summary(np.asarray(data["raw_present_velocity"], dtype=np.float64)),
        "actual_dt_s": {"min": float(np.min(data["actual_dt"])), "p50": float(np.median(data["actual_dt"])),
                         "p99": float(np.percentile(data["actual_dt"], 99)), "max": float(np.max(data["actual_dt"]))},
        "note": "Present_Velocity raw units are diagnostic only and are not used by the deployed estimator.",
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
