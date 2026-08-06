#!/usr/bin/env python3
"""Read-only SO101 state probe: print current joint angles and raw positions.

Never enables torque and never sends any command.  Used to record the real
arm's q_ctrl at a manually-positioned pose so the fine model can be rendered
at that same q_ctrl (scripts/view_so101_pose.py) to compare model-vs-real.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only SO101 state probe (no torque, no motion).")
    parser.add_argument("--hardware-config", default="configs/hardware/so101_follower.local.yaml")
    args = parser.parse_args()

    from robot_runtime.config import load_hardware_config
    from robot_runtime.factory import make_so101_backend

    hw = load_hardware_config(args.hardware_config)
    backend = make_so101_backend(args.hardware_config)
    try:
        backend.connect_read_only()
        state = backend.read_state(tick_index=0)
        if not state.valid:
            print(f"state invalid: {state.validity_flags}")
            return
        raw = np.asarray(state.diagnostics["present_position_raw"], dtype=int)
        names = list(hw.joint_names) + ["gripper"]
        print("q_ctrl(deg) =", np.round(np.rad2deg(state.q_ctrl), 2).tolist())
        print("q_ctrl(rad) =", np.round(state.q_ctrl, 5).tolist())
        for i, name in enumerate(names):
            q = np.rad2deg(state.q_ctrl[i]) if i < 5 else float("nan")
            print(f"  {name:<14} q_ctrl={q:>8.2f} deg   raw={raw[i]:>5}   "
                  f"(raw range [{hw.raw_low[i]}, {hw.raw_high[i]}])")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
