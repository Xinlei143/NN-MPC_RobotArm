#!/usr/bin/env python3
"""Derive ctrl-space MPC playback files from the frozen kin-space bundles.

The frozen 20260808_refs_6phase bundles are kin-space (reference.npz) with
only horizon+1 tail rows -- enough for simulation padding, but the real
threadedASAP path needs execution_steps + horizon + anticipation +
preview + 1 rows (run_real_cem_mpc.py:69; shadow default anticipation=6
needs execution_steps + 13).  This script converts each frozen bundle to a
ctrl-space joint_reference_mpc.npz with PADDING_ROWS explicit hold rows
(q holds the final pose, dq/ddq = 0, matching a stopped nominal playback),
then freezes SHA-256 entries in a new mpc_manifest.json.

The manifest mirrors the frozen manifest's artifact path
(artifacts.<label>.q_des_ctrl.npy.sha256) so find_reference_manifest_entry
(scripts/run_real_direct_control.py:98) validates MPC references unchanged.

The execution-window conversion q_ctrl = (q_kin - OFFSET) / SIGN must
reproduce the frozen q_des_ctrl.npy (np.allclose atol=1e-9, float64-level
identity; the saved npz is float32, matching _load_joint_file_reference's
cast).  Offline gates: JointFilePlayer (hardware envelope / home start /
per-sample step) + manifest hash match, for all 12 labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import numpy as np

from robot_runtime.config import load_hardware_config

# Same constants as the frozen generator (dynamics_modeling/scripts/
# so101_gen_tcp_reference.py): the runtime profile mapping q_kin = SIGN*q_ctrl + OFFSET.
CTRL_HOME = np.array([0.0, 0.0, 0.0, 0.2516342834, 0.0])
OFFSET = np.array([0.0, +0.00872665, -0.1396263, -0.2516342834, 0.0])
SIGN = np.ones(5)
HORIZON = 6  # locked active authority: H=6 @ 30 Hz == 0.2 s
# Shadow default anticipation_delay_steps = 6 -> need exec+13 rows; 20 gives
# margin for the calibrated D (expected ~2-4) plus worker tail effects.
PADDING_ROWS = 20
REFS_DIR = ROOT / "outputs/hardware/so101_pre_mpc/20260808_refs_6phase"
HARDWARE_CONFIG_DEFAULT = "configs/hardware/so101_follower.local.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs-dir", type=Path, default=REFS_DIR,
                        help="Frozen 6-phase reference bundle directory.")
    parser.add_argument("--hardware-config", default=HARDWARE_CONFIG_DEFAULT,
                        help="Hardware config for the offline JointFilePlayer gate verification.")
    parser.add_argument("--skip-gate-verification", action="store_true",
                        help="Skip the offline JointFilePlayer/hash gate check (not recommended).")
    args = parser.parse_args()

    refs_dir = args.refs_dir
    frozen_manifest_path = refs_dir / "manifest.json"
    if not frozen_manifest_path.exists():
        raise FileNotFoundError(f"frozen manifest not found: {frozen_manifest_path}")
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))

    labels = sorted(f"circle_p{i}" for i in range(6)) + sorted(f"figure8_p{i}" for i in range(6))
    mpc_manifest = {
        "format_version": 1,
        "design": {
            **frozen_manifest.get("design", {}),
            "mpc": {
                "horizon": HORIZON,
                "padding_rows": PADDING_ROWS,
                "playback_envelope": "hardware",
                # exec+13 is the worst-case requirement (shadow default D=6);
                # remaining rows are the calibrated-D margin.
                "anticipation_margin_steps": PADDING_ROWS - (HORIZON + 6 + 0 + 1),
            },
        },
        "plant_identity": frozen_manifest.get("plant_identity", {}),
        "artifacts": {},
    }
    for label in labels:
        entry_dir = refs_dir / label
        bundle_path = entry_dir / "reference.npz"
        frozen_ctrl_path = entry_dir / "q_des_ctrl.npy"
        if not bundle_path.exists() or not frozen_ctrl_path.exists():
            raise FileNotFoundError(f"frozen bundle incomplete for {label}: {entry_dir}")

        with np.load(bundle_path, allow_pickle=False) as bundle:
            exec_steps = int(bundle["execution_steps"])
            q_kin = np.asarray(bundle["q_des"], dtype=np.float64)
            dq_kin = np.asarray(bundle["dq_des"], dtype=np.float64)
            ddq_kin = np.asarray(bundle["ddq_des"], dtype=np.float64)
        frozen_ctrl = np.asarray(np.load(frozen_ctrl_path), dtype=np.float64)

        q_ctrl_exec = (q_kin[:exec_steps] - OFFSET[None, :]) / SIGN[None, :]
        if not np.allclose(q_ctrl_exec, frozen_ctrl, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"{label}: ctrl conversion does not reproduce the frozen q_des_ctrl.npy")
        if not np.allclose(q_ctrl_exec[0], CTRL_HOME, rtol=0.0, atol=1e-9):
            raise RuntimeError(f"{label}: first executed row is not CTRL_HOME")

        hold_q = q_ctrl_exec[-1]
        q_full = np.concatenate([q_ctrl_exec, np.tile(hold_q, (PADDING_ROWS, 1))]).astype(np.float32)
        zeros = np.zeros((PADDING_ROWS, q_ctrl_exec.shape[1]), dtype=np.float32)
        dq_full = np.concatenate([dq_kin[:exec_steps] / SIGN[None, :], zeros]).astype(np.float32)
        ddq_full = np.concatenate([ddq_kin[:exec_steps] / SIGN[None, :], zeros]).astype(np.float32)

        out_path = entry_dir / "joint_reference_mpc.npz"
        np.savez(out_path, q_des=q_full, dq_des=dq_full, ddq_des=ddq_full,
                 execution_steps=exec_steps, padding_rows=PADDING_ROWS,
                 source="derived from frozen kin-space reference.npz")
        mpc_manifest["artifacts"][label] = {
            "shape": label.split("_")[0],
            "phase_index": int(label.split("_p")[1]),
            "start_phase_deg": frozen_manifest.get("artifacts", {}).get(label, {}).get("start_phase_deg"),
            "lap_duration_s": frozen_manifest.get("artifacts", {}).get(label, {}).get("lap_duration_s"),
            "q_des_ctrl.npy": {"sha256": sha256_file(out_path), "bytes": out_path.stat().st_size},
            "execution_steps": exec_steps,
            "rows": int(q_full.shape[0]),
            "padding_rows": PADDING_ROWS,
            "derived_from": {"reference.npz": {"sha256": sha256_file(bundle_path)}},
        }
        print(f"[{label}] {out_path.name}: {exec_steps}+{PADDING_ROWS} rows, "
              f"exec-window match: allclose(1e-9), sha256={mpc_manifest['artifacts'][label]['q_des_ctrl.npy']['sha256'][:16]}...")

    mpc_manifest_path = refs_dir / "mpc_manifest.json"
    mpc_manifest_path.write_text(json.dumps(mpc_manifest, sort_keys=True, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"manifest: {mpc_manifest_path} (sha256={sha256_file(mpc_manifest_path)[:16]}...)")

    if not args.skip_gate_verification:
        from scripts.run_real_direct_control import JointFilePlayer, find_reference_manifest_entry
        config = load_hardware_config(args.hardware_config)
        for label in labels:
            path = refs_dir / label / "joint_reference_mpc.npz"
            with np.load(path, allow_pickle=False) as archive:
                q_des = np.asarray(archive["q_des"], dtype=np.float32)
            JointFilePlayer(q_des, config=config)  # raises on any gate failure
            shape_name, entry, sha = find_reference_manifest_entry(mpc_manifest_path, path)
            if shape_name != label or entry["execution_steps"] != int(q_des.shape[0] - PADDING_ROWS):
                raise RuntimeError(f"{label}: manifest entry inconsistent ({shape_name})")
            print(f"[{label}] gates passed: hardware envelope / home start / per-sample step, "
                  f"manifest sha256={sha[:16]}...")
    print("all 12 MPC references derived and verified")


if __name__ == "__main__":
    main()
