#!/usr/bin/env python3
"""B3/B5 startup-only verification; does not run an excitation trajectory."""
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
    parser = argparse.ArgumentParser(description="Verify SO101 hold-current and constrained startup-to-home only.")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--startup-seconds", type=float, default=3.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=15.0,
                        help="Maximum additional hold time while waiting for home convergence.")
    parser.add_argument("--home-tolerance-deg", type=float, default=1.0,
                        help="Maximum absolute q_ctrl error from home required for pass; must leave room inside the +/-3 deg envelope.")
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--operator-supported-shutdown", action="store_true")
    args = parser.parse_args()
    if not args.enable_motion or not args.operator_supported_shutdown:
        raise SystemExit("refusing torque enable without --enable-motion and --operator-supported-shutdown")
    if (args.startup_seconds <= 0 or args.startup_timeout_seconds <= 0 or args.home_tolerance_deg <= 0
            or args.home_tolerance_deg >= 3 or args.settle_seconds < 0):
        raise SystemExit("startup duration/timeout and home tolerance must be positive; tolerance must be below 3 deg")
    config = load_hardware_config(args.hardware_config)
    backend = make_so101_backend(args.hardware_config)
    before = None
    try:
        backend.connect()
        before = backend.read_state(tick_index=0)
        startup_state = backend.startup_to_home(args.startup_seconds,
                                                convergence_timeout_s=args.startup_timeout_seconds,
                                                home_tolerance_rad=float(np.deg2rad(args.home_tolerance_deg)))
        time.sleep(args.settle_seconds)
        after = backend.read_state(tick_index=1)
        evidence = {"phase": "B3_B5_startup_only", "startup_seconds": args.startup_seconds,
                    "startup_timeout_seconds": args.startup_timeout_seconds,
                    "home_tolerance_deg": args.home_tolerance_deg,
                    "settle_seconds": args.settle_seconds, "initial_q_ctrl_rad": before.q_ctrl.tolist(),
                    "q_ctrl_at_envelope_entry_rad": startup_state.q_ctrl.tolist(),
                    "final_q_ctrl_rad": after.q_ctrl.tolist(), "home_q_ctrl_rad": config.home_q_ctrl.tolist(),
                    "steady_state_error_rad": (after.q_ctrl - config.home_q_ctrl).tolist(),
                    "initial_raw": before.diagnostics["present_position_raw"].tolist(),
                    "final_raw": after.diagnostics["present_position_raw"].tolist(),
                    "final_flags": list(after.validity_flags), "plant_identity": config.plant_identity()}
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(evidence, indent=2))
    except Exception as exc:
        state = getattr(backend, "_last_state", None)
        failure = {"phase": "B3_B5_startup_only_failed", "startup_seconds": args.startup_seconds,
                   "startup_timeout_seconds": args.startup_timeout_seconds, "home_tolerance_deg": args.home_tolerance_deg,
                   "error": str(exc), "plant_identity": config.plant_identity()}
        if before is not None:
            failure["initial_q_ctrl_rad"] = before.q_ctrl.tolist()
            failure["initial_raw"] = before.diagnostics["present_position_raw"].tolist()
        if state is not None:
            failure["final_q_ctrl_rad"] = state.q_ctrl.tolist()
            failure["final_raw"] = np.asarray(state.diagnostics.get("present_position_raw", [])).tolist()
            failure["goal_position_readback_raw"] = np.asarray(
                state.diagnostics.get("goal_position_readback_raw", [])).tolist()
            failure["final_flags"] = list(state.validity_flags)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        raise
    finally:
        backend.close()


if __name__ == "__main__":
    main()
