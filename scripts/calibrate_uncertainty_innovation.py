"""Calibrate active-packet prediction-innovation thresholds from clean ID rollouts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", nargs="+", required=True, help="Clean threaded rollout.npz files.")
    parser.add_argument("--output", required=True, help="Output JSON calibration file.")
    parser.add_argument("--fallback-quantile", type=float, default=0.99, help="Innovation quantile for hard-fallback confirmation.")
    parser.add_argument("--recovery-quantile", type=float, default=0.95, help="Lower innovation quantile required to leave fallback.")
    args = parser.parse_args()
    if not 0.0 < args.recovery_quantile < args.fallback_quantile < 1.0:
        raise ValueError("require 0 < recovery_quantile < fallback_quantile < 1")
    return args


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def scalar(archive: np.lib.npyio.NpzFile, name: str) -> int:
    if name not in archive.files:
        raise ValueError(f"rollout is missing {name}")
    return int(np.asarray(archive[name]).reshape(-1)[0])


def main() -> None:
    args = parse_args()
    per_rollout: list[dict[str, object]] = []
    values: list[np.ndarray] = []
    for raw_path in args.rollouts:
        path = resolve(raw_path)
        with np.load(path, allow_pickle=False) as archive:
            if "packet_prediction_q_innovation" not in archive.files:
                raise ValueError(f"{path} predates packet_prediction_q_innovation logging; collect a new ID rollout")
            for name in ("payload_level", "actuator_gain_level", "force_pulse_level", "observation_noise_level"):
                if scalar(archive, name) != 0:
                    raise ValueError(f"{path} is not an ID rollout: {name} is nonzero")
            innovation = np.asarray(archive["packet_prediction_q_innovation"], dtype=np.float64)
        innovation = innovation[np.isfinite(innovation)]
        if not len(innovation):
            raise ValueError(f"{path} has no finite active-packet innovation samples")
        values.append(innovation)
        per_rollout.append(
            {
                "rollout": str(path.resolve()),
                "samples": int(len(innovation)),
                "p95": float(np.quantile(innovation, 0.95)),
                "p99": float(np.quantile(innovation, 0.99)),
                "max": float(np.max(innovation)),
            }
        )
    pooled = np.concatenate(values)
    fallback = float(np.quantile(pooled, args.fallback_quantile))
    recovery = float(np.quantile(pooled, args.recovery_quantile))
    payload = {
        "metric": "packet_prediction_q_innovation_rad_l2",
        "rollouts": per_rollout,
        "pooled_samples": int(len(pooled)),
        "fallback_quantile": args.fallback_quantile,
        "recovery_quantile": args.recovery_quantile,
        "fallback_threshold": fallback,
        "recovery_threshold": recovery,
        "distribution": {
            "p50": float(np.quantile(pooled, 0.50)),
            "p90": float(np.quantile(pooled, 0.90)),
            "p95": float(np.quantile(pooled, 0.95)),
            "p99": float(np.quantile(pooled, 0.99)),
            "max": float(np.max(pooled)),
        },
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
