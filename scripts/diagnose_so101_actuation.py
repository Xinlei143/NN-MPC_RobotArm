#!/usr/bin/env python3
"""SO101 actuation diagnosis, with an explicit no-torque read-only mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_runtime.config import load_hardware_config
from robot_runtime.factory import make_so101_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="No-motion SO101 torque, limits, dead-zone, PID, voltage, current, load, and temperature readback.")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--read-only", action="store_true",
                        help="Read registers without motor configuration writes or torque enable.")
    parser.add_argument("--operator-supported-shutdown", action="store_true")
    args = parser.parse_args()
    if not args.read_only and not args.operator_supported_shutdown:
        raise SystemExit("refusing torque enable without --operator-supported-shutdown")
    config = load_hardware_config(args.hardware_config)
    backend = make_so101_backend(args.hardware_config)
    try:
        # The post-fault path must not re-enable torque just to inspect a
        # temperature.  It has no Goal_Position or configuration write.
        if args.read_only:
            backend.connect_read_only()
        else:
            backend.connect()
        state = backend.read_state(tick_index=0)
        inside_hardware = (config.hardware_joint_low is not None and config.hardware_joint_high is not None
                           and bool(np.all(state.q_ctrl >= config.hardware_joint_low)
                                    and np.all(state.q_ctrl <= config.hardware_joint_high)))
        report = {"phase": "B2_actuation_diagnosis", "read_only": args.read_only,
                  "state_valid": state.valid, "state_validity_flags": list(state.validity_flags),
                  "inside_hardware_envelope": inside_hardware,
                  "inside_experiment_envelope": bool(state.diagnostics.get("inside_experiment_envelope", False)),
                  "q_ctrl_rad": state.q_ctrl.tolist(),
                  "q_ctrl_deg": np.rad2deg(state.q_ctrl).tolist(),
                  "present_position_raw": state.diagnostics["present_position_raw"].tolist(),
                  "actuation_registers": backend.read_actuation_registers(),
                  "plant_identity": config.plant_identity(),
                  "expected": {"Torque_Enable": 1, "Lock": 1, "Operating_Mode": 0,
                               "Present_Voltage_raw_to_v": "raw * 0.1"}}
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
