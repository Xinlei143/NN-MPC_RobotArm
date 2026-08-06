#!/usr/bin/env python3
"""B2-only, operator-supervised tiny single-joint jog to establish safe limits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_runtime.config import load_hardware_config
from robot_runtime.factory import make_so101_backend


def main() -> None:
    parser = argparse.ArgumentParser(description="B2 only: move one SO101 joint by at most 1.0 degree under raw safety limits.")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--joint", choices=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"], required=True)
    parser.add_argument("--direction", choices=["positive", "negative"], required=True)
    parser.add_argument("--step-deg", type=float, default=.5)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--enable-manual-jog", action="store_true")
    parser.add_argument("--operator-supported-shutdown", action="store_true")
    args = parser.parse_args()
    if not args.enable_manual_jog or not args.operator_supported_shutdown:
        raise SystemExit("refusing B2 jog without --enable-manual-jog and --operator-supported-shutdown")
    if not 0 < args.step_deg <= 1.0 or args.hold_seconds <= 0:
        raise SystemExit("B2 jog requires 0 < --step-deg <= 1.0 and positive --hold-seconds")
    config = load_hardware_config(args.hardware_config)
    backend = make_so101_backend(args.hardware_config)
    try:
        backend.connect()
        before = backend.read_state(tick_index=0)
        sign = 1.0 if args.direction == "positive" else -1.0
        result = backend.manual_raw_safety_jog(config.joint_names.index(args.joint), sign * np.deg2rad(args.step_deg))
        if not result.tx_local_success:
            raise RuntimeError("B2 jog transmission failed")
        time.sleep(args.hold_seconds)
        after = backend.read_state(tick_index=1)
        evidence = {"phase": "B2_manual_raw_safety_jog", "joint": args.joint, "direction": args.direction,
                    "step_deg": args.step_deg, "hold_seconds": args.hold_seconds,
                    "before_q_ctrl_rad": before.q_ctrl.tolist(), "after_q_ctrl_rad": after.q_ctrl.tolist(),
                    "before_raw": np.asarray(before.diagnostics["present_position_raw"]).tolist(),
                    "transmitted_raw": result.tx_goal_position_raw.tolist(),
                    "after_raw": np.asarray(after.diagnostics["present_position_raw"]).tolist(),
                    "goal_position_readback_raw": np.asarray(
                        after.diagnostics.get("goal_position_readback_raw", np.full(6, -1))).tolist(),
                    "goal_readback_mismatch": bool(after.diagnostics.get("goal_readback_mismatch", False)),
                    "command_delivery_uncertain": bool(after.diagnostics.get("command_delivery_uncertain", False)),
                    "read_retry_count": int(after.diagnostics.get("read_retry_count", -1)),
                    "diagnostic_sample_age_s": float(after.diagnostics.get("diagnostic_sample_age_s", float("inf"))),
                    "after_flags": list(after.validity_flags), "plant_identity": config.plant_identity(),
                    "warning": "B2-only manual range evidence; not a direct-control or MPC result."}
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
