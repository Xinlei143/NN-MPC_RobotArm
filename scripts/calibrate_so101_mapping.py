#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit q_kin = sign*q_ctrl + offset from >=5 instrumented poses.")
    parser.add_argument("poses", help="NPZ with q_ctrl and q_kin arrays [N,5]")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = np.load(args.poses)
    ctrl, kin = data["q_ctrl"], data["q_kin"]
    if ctrl.shape != kin.shape or ctrl.ndim != 2 or ctrl.shape[0] < 5 or ctrl.shape[1] != 5:
        raise SystemExit("mapping requires matching q_ctrl/q_kin arrays with at least five [5]-joint poses")
    signs, offsets, residuals = [], [], []
    for joint in range(5):
        candidates = []
        for sign in (-1.0, 1.0):
            offset = float(np.median(kin[:, joint] - sign * ctrl[:, joint]))
            error = kin[:, joint] - (sign * ctrl[:, joint] + offset)
            candidates.append((np.median(np.abs(error)), sign, offset, error))
        _, sign, offset, error = min(candidates, key=lambda item: item[0])
        signs.append(sign); offsets.append(offset); residuals.append(error)
    error = np.stack(residuals, axis=1)
    payload = {"schema_version": 1, "joint_sign": signs, "joint_offset_rad": offsets,
               "pose_count": int(ctrl.shape[0]), "reprojection_p95_deg": np.rad2deg(np.percentile(np.abs(error), 95, axis=0)).tolist(),
               "source": "instrumented_reference_poses"}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()
