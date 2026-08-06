#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from robot_runtime.backends.so101_backend import ARM_JOINTS, SO101Backend, degrees_to_raw, raw_to_degrees
from robot_runtime.config import load_hardware_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare custom rad/degree/raw conversion with pinned LeRobot.")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    config = load_hardware_config(args.hardware_config)
    backend = SO101Backend(config)
    follower = backend._make_follower(config)
    if Path(follower.calibration_fpath).resolve() != config.calibration_path.resolve():
        follower._load_calibration(config.calibration_path)
        follower.bus.calibration = follower.calibration
    rng = np.random.default_rng(101)
    maximum_raw_error = maximum_roundtrip_error = 0.0
    for name in ARM_JOINTS:
        motor = follower.bus.motors[name]
        calibration = follower.calibration[name]
        values = rng.uniform(np.rad2deg(config.joint_low[ARM_JOINTS.index(name)]),
                             np.rad2deg(config.joint_high[ARM_JOINTS.index(name)]), args.samples)
        custom = degrees_to_raw(values, np.full(args.samples, calibration.range_min),
                                np.full(args.samples, calibration.range_max))
        expected = np.asarray([follower.bus._unnormalize({motor.id: float(value)})[motor.id] for value in values])
        maximum_raw_error = max(maximum_raw_error, float(np.max(np.abs(custom - expected))))
        recovered = raw_to_degrees(custom, calibration.range_min, calibration.range_max)
        maximum_roundtrip_error = max(maximum_roundtrip_error, float(np.max(np.abs(recovered - values))))
    print({"samples_per_joint": args.samples, "max_raw_count_error": maximum_raw_error,
           "max_roundtrip_degree_error": maximum_roundtrip_error})
    if maximum_raw_error > 1 or maximum_roundtrip_error > 360 / 4095 + 1e-12:
        raise SystemExit(1)


if __name__ == "__main__": main()
