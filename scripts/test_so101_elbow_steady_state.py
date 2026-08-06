#!/usr/bin/env python3
"""Isolated elbow steady-state tracking test with the other joints held near home."""
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


ELBOW_INDEX = 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure SO101 elbow steady-state error while the other arm joints are moved near home.")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--elbow-offset-deg", type=float, default=0.0,
                        help="Elbow absolute q_ctrl target relative to home; start with 0 for the home-hold test.")
    parser.add_argument("--setup-seconds", type=float, default=6.0,
                        help="Constrained move duration for the four non-elbow joints.")
    parser.add_argument("--hold-seconds", type=float, default=15.0)
    parser.add_argument("--steady-window-seconds", type=float, default=5.0)
    parser.add_argument("--output", required=True, help="NPZ output path; a JSON summary is written alongside it.")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--operator-supported-shutdown", action="store_true")
    args = parser.parse_args()
    if not args.enable_motion or not args.operator_supported_shutdown:
        raise SystemExit("refusing torque enable without --enable-motion and --operator-supported-shutdown")
    if args.setup_seconds <= 0 or args.hold_seconds <= 0 or not 0 < args.steady_window_seconds <= args.hold_seconds:
        raise SystemExit("setup/hold durations must be positive and steady window must lie within hold duration")
    config = load_hardware_config(args.hardware_config)
    elbow_target = float(config.home_q_ctrl[ELBOW_INDEX] + np.deg2rad(args.elbow_offset_deg))
    if not config.joint_low[ELBOW_INDEX] <= elbow_target <= config.joint_high[ELBOW_INDEX]:
        raise SystemExit("elbow target must lie inside the narrow experiment envelope")
    backend = make_so101_backend(args.hardware_config)
    records: list[dict[str, np.ndarray | int | float | bool]] = []
    try:
        backend.connect()
        initial = backend.prepare_hardware_motion()
        target = config.home_q_ctrl.astype(np.float32).copy()
        target[ELBOW_INDEX] = elbow_target
        non_elbow = np.ones(5, dtype=bool); non_elbow[ELBOW_INDEX] = False
        # Preserve the measured elbow during non-elbow setup.
        setup_target = target.copy(); setup_target[ELBOW_INDEX] = initial.q_ctrl[ELBOW_INDEX]
        backend.move_selected_joints_to_configuration(setup_target, non_elbow, args.setup_seconds)
        steps = int(np.ceil(args.hold_seconds / config.control_dt))
        for tick in range(steps):
            state = backend.read_state(tick_index=tick)
            command = backend.send_hardware_joint_targets(target, tick_index=tick)
            if not state.valid or not command.tx_local_success:
                raise RuntimeError(f"elbow steady-state test aborted at tick {tick}")
            records.append({"timestamp_ns": state.timestamp_ns, "q_ctrl": state.q_ctrl.copy(),
                            "raw": np.asarray(state.diagnostics["present_position_raw"]).copy(),
                            "goal_raw": command.tx_goal_position_raw.copy(),
                            "goal_readback_raw": np.asarray(state.diagnostics.get("goal_position_readback_raw", np.full(6, -1))).copy(),
                            "readback_mismatch": bool(state.diagnostics.get("goal_readback_mismatch", False))})
            time.sleep(config.control_dt)
        q = np.asarray([row["q_ctrl"] for row in records], dtype=np.float64)
        raw = np.asarray([row["raw"] for row in records], dtype=np.int64)
        window = max(1, int(np.ceil(args.steady_window_seconds / config.control_dt)))
        elbow_error = q[-window:, ELBOW_INDEX] - elbow_target
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, q_ctrl=q, present_raw=raw,
                            goal_raw=np.asarray([row["goal_raw"] for row in records]),
                            goal_readback_raw=np.asarray([row["goal_readback_raw"] for row in records]),
                            timestamp_ns=np.asarray([row["timestamp_ns"] for row in records]),
                            elbow_target_rad=np.asarray(elbow_target), elbow_error_rad=elbow_error)
        summary = {"phase": "B3_elbow_steady_state", "sample_count": len(records),
                   "elbow_target_rad": elbow_target, "elbow_target_deg": float(np.rad2deg(elbow_target)),
                   "steady_window_seconds": args.steady_window_seconds,
                   "elbow_error_deg": {"mean": float(np.rad2deg(np.mean(elbow_error))),
                                        "std": float(np.rad2deg(np.std(elbow_error))),
                                        "max_abs": float(np.rad2deg(np.max(np.abs(elbow_error)))),
                                        "final": float(np.rad2deg(elbow_error[-1]))},
                   "other_joint_final_error_deg": np.rad2deg(q[-1] - config.home_q_ctrl).tolist(),
                   "goal_readback_mismatch_count": int(sum(row["readback_mismatch"] for row in records)),
                   "plant_identity": config.plant_identity(), "dataset": output.name}
        output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
