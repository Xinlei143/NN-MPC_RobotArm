#!/usr/bin/env python3
"""Generate and freeze the SO101 Model-A TCP reference bundles.

Locked experiment design (grill 2026-08-07, user-confirmed):
  - orientation policy: position-only (5-DOF, tool direction free, DLS min-norm)
  - plane: YZ vertical (u=(0,1,0), v=(0,0,1)), center = home TCP (offset 0)
  - circle r=0.04 / figure8 a=0.045 b=0.025, 3 laps @ 9.0/7.2 s per lap
    (quintic lap profile re-derived 2026-08-07: constant-speed assumption gave
    elbow 18.8 deg/s @5 s/lap vs P99 10.9; slowing the lap is the frozen fix),
    approach/return 2 s, holds 0.5 s, joint departure/return 2 s
  - velocity ceiling: per-joint max |dq| over laps <= collection measured P99
  - position gate: reference joint range inside the collection's measured
    P0.1..P99.9 span (the collection spanned the full hardware workspace;
    the references exceed the first-motion +/-3 deg authority by design)
  - generated on the fine model (kin space), then converted to ctrl space
    with the runtime profile's three-joint mapping.

Outputs (SHA-256 frozen) under
  outputs/hardware/so101_pre_mpc/20260808_refs/{circle,figure8}/:
    reference.npz    pipeline bundle (kin-space q_des + task_positions_des)
    q_des_ctrl.npy   (T, 5) ctrl-space q for the direct controller
  plus manifest.json with per-file hashes, validation, and envelope checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "dynamics_modeling"):
    sys.path.insert(0, str(p))

import mujoco
import numpy as np

from mpc.ik_solver import IKConfig
from mpc.reference_pipeline import (
    ReferenceConfig,
    build_reference,
    save_reference_bundle,
)
from mpc.task_space_reference import SEGMENT_SHAPE_LOOP

MODEL = ROOT / "dynamics_modeling/robots/so101_fine/scene_table_guard_25mm.xml"
CTRL_HOME = np.array([0.0, 0.0, 0.0, 0.2516342834, 0.0])
OFFSET = np.array([0.0, +0.00872665, -0.1396263, -0.2516342834, 0.0])
SIGN = np.ones(5)
KIN_HOME = SIGN * CTRL_HOME + OFFSET
DT = 1.0 / 30.0
HORIZON = 6  # locked active authority: H=6 @ 30 Hz == 0.2 s

# Measured-channel per-sample P99 of |dq| over the frozen Model-A collection
# (outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.npz,
#  states dq channel, all 48 sessions x all motion modes, per joint).
P99_MEASURED_DEG_S = np.array([10.89, 11.46, 10.89, 10.31, 10.89])
# The same frozen collection, measured_q channel, provides the position
# envelope: references must lie inside the per-joint P0.1..P99.9 span of the
# data the model was trained on (collection spanned the full workspace, so
# this is a far wider gate than the first-motion +/-3 deg authority).
COLLECTION_NPZ = ROOT / "outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.npz"
POSITION_P_LO, POSITION_P_HI = 0.1, 99.9
JOINTS = ["pan", "lift", "elbow", "wrist_flex", "wrist_roll"]
GUARD_Z = 0.025
ARM_BODIES = ("upper_arm", "lower_arm", "wrist", "gripper", "moving_jaw_so101_v1")

SHAPES = {
    "circle": {"shape_name": "circle", "circle_radius": 0.04, "lap_duration": 9.0},
    "figure8": {
        "shape_name": "figure8",
        "figure8_axis_a": 0.045,
        "figure8_axis_b": 0.025,
        # 8.2 s/lap (was 7.2 s): the Gerono parameterization is non-uniform in
        # arc length, so phase offsets shift the high ds/dtheta region into the
        # high dtheta/dt quintic middle.  At 7.2 s, phases 60/120/240/300 deg
        # peaked at 11.82 deg/s (pan) vs measured P99 10.89; slowing to 8.2 s
        # brings every phase under the ceiling (10.38, ~4.7% margin).
        "lap_duration": 8.2,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(shape_params: dict) -> ReferenceConfig:
    return ReferenceConfig(
        repeat_count=3,
        start_hold_duration=0.5,
        joint_departure_duration=2.0,
        approach_duration=2.0,
        return_duration=2.0,
        joint_return_duration=2.0,
        final_hold_duration=0.5,
        center_mode="relative",
        center_offset=(0.0, 0.0, 0.0),
        plane_axis_u=(0.0, 1.0, 0.0),
        plane_axis_v=(0.0, 0.0, 1.0),
        fixed_orientation="safe",
        ee_site_name="gripperframe",
        safe_departure_mode="always",
        safe_sigma_threshold=0.04,  # 6D sigma at kin home (5-DOF: gate unreachable at 0.10)
        safe_joint_limit_margin=0.05,
        safe_q=tuple(KIN_HOME),
        ik_config=IKConfig(orientation_mode="position_only"),
        # 5-DOF: defaults are 6-tuples (ABB). Generic pipeline gates, far above
        # the measured-P99 design ceiling used in envelope_report.
        max_joint_velocity=(1.0, 1.0, 1.0, 2.0, 2.0),
        max_joint_acceleration=(5.0, 5.0, 5.0, 10.0, 10.0),
        **shape_params,
    )


def envelope_report(bundle) -> dict:
    lap_mask = bundle.segment_ids == SEGMENT_SHAPE_LOOP
    max_dq = np.rad2deg(np.max(np.abs(bundle.dq_des[lap_mask]), axis=0))
    margins = P99_MEASURED_DEG_S - max_dq
    return {
        "max_dq_lap_deg_s": {name: float(value) for name, value in zip(JOINTS, max_dq)},
        "p99_measured_deg_s": {name: float(value) for name, value in zip(JOINTS, P99_MEASURED_DEG_S)},
        "margin_deg_s": {name: float(value) for name, value in zip(JOINTS, margins)},
        "all_joints_at_or_below_p99": bool(np.all(max_dq <= P99_MEASURED_DEG_S + 1e-9)),
    }


def position_distribution_report(bundle) -> dict:
    """Per-joint reference range vs the collection's measured P0.1..P99.9 span.

    The references are allowed to leave the first-motion +/-3 deg authority
    (the 4 cm circle needs pan +/-6.5 deg) but must stay inside the measured
    training distribution, which spans the full workspace.
    """
    if not COLLECTION_NPZ.exists():
        raise FileNotFoundError(f"collection npz not found: {COLLECTION_NPZ}")
    measured = np.load(COLLECTION_NPZ)["measured_q"]
    q_ctrl = (bundle.q_des[: bundle.execution_steps] - OFFSET[None, :]) / SIGN[None, :]
    report: dict = {}
    ok = True
    for j, name in enumerate(JOINTS):
        lo, hi = np.percentile(measured[:, j], [POSITION_P_LO, POSITION_P_HI])
        rmin, rmax = float(q_ctrl[:, j].min()), float(q_ctrl[:, j].max())
        inside = rmin >= lo and rmax <= hi
        ok = ok and inside
        report[name] = {
            "ref_min": rmin, "ref_max": rmax,
            "collection_p0.1": float(lo), "collection_p99.9": float(hi),
            "inside_collection_span": bool(inside),
        }
    report["all_joints_inside_collection_span"] = ok
    return report


def clearance_report(model, data, bundle) -> dict:
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ARM_BODIES]
    geoms = [g for bid in body_ids if bid >= 0
             for g in range(model.ngeom) if model.geom(g).bodyid == bid]
    min_z = 1e9
    for i in range(0, bundle.execution_steps, 2):
        data.qpos[:5] = bundle.q_des[i]
        mujoco.mj_forward(model, data)
        min_z = min(min_z, float(np.min([data.geom_xpos[g][2] for g in geoms])))
    return {
        "min_arm_geom_z_m": min_z,
        "guard_z_m": GUARD_Z,
        "clearance_above_guard_m": min_z - GUARD_Z,
        "passes_offline_gate": bool(min_z - GUARD_Z > 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs/hardware/so101_pre_mpc/20260808_refs",
    )
    parser.add_argument("--shape", choices=["circle", "figure8"], default=None,
                        help="Generate only this shape (default: both frozen shapes). "
                             "For extra tests (e.g. an 8 cm circle) generate into a "
                             "separate --out-dir so the formal artifacts stay untouched.")
    parser.add_argument("--lap-duration", type=float, default=None,
                        help="Override the shape's lap duration in seconds.")
    parser.add_argument("--circle-radius", type=float, default=None,
                        help="Override the circle radius in meters.")
    parser.add_argument("--figure8-axis-a", type=float, default=None,
                        help="Override the figure-8 axis a in meters.")
    parser.add_argument("--figure8-axis-b", type=float, default=None,
                        help="Override the figure-8 axis b in meters.")
    parser.add_argument("--start-phase", type=float, default=0.0,
                        help="Start phase of the closed shape traversal in degrees "
                             "(rotates the start point along the circle / figure-8; "
                             "approach and lap starts stay aligned).")
    parser.add_argument("--phase-index", type=int, default=0,
                        help="Phase trial index used in the output artifact label "
                             "(e.g. circle_p3); only affects naming, not geometry.")
    args = parser.parse_args()
    if not np.isfinite(args.start_phase):
        parser.error("--start-phase must be finite")
    if args.phase_index < 0:
        parser.error("--phase-index must be non-negative")

    shapes: dict = {name: dict(params) for name, params in SHAPES.items()}
    if args.shape is not None:
        shapes = {args.shape: shapes[args.shape]}
    for params in shapes.values():
        params["start_phase"] = np.deg2rad(args.start_phase)
    if args.lap_duration is not None:
        for params in shapes.values():
            params["lap_duration"] = float(args.lap_duration)
    if args.circle_radius is not None and "circle" in shapes:
        shapes["circle"]["circle_radius"] = float(args.circle_radius)
    if args.figure8_axis_a is not None and "figure8" in shapes:
        shapes["figure8"]["figure8_axis_a"] = float(args.figure8_axis_a)
    if args.figure8_axis_b is not None and "figure8" in shapes:
        shapes["figure8"]["figure8_axis_b"] = float(args.figure8_axis_b)

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    manifest_path = args.out_dir / "manifest.json"
    # Merge into an existing manifest so repeated calls with different
    # --start-phase / --phase-index share one frozen artifact registry.
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("artifacts", {})
    else:
        manifest = {
            "format_version": 1,
            "design": {
                "policy": "position_only",
                "plane_axis_u": [0.0, 1.0, 0.0],
                "plane_axis_v": [0.0, 0.0, 1.0],
                "center_mode": "relative",
                "center_offset": [0.0, 0.0, 0.0],
                "repeat_count": 3,
                "approach_duration_s": 2.0,
                "return_duration_s": 2.0,
                "velocity_ceiling": "measured P99",
                "horizon": HORIZON,
                "control_dt_s": DT,
            },
            "plant_identity": {
                "robot_id": "so101_real_v1",
                "model_xml": str(MODEL),
                "model_xml_sha256": sha256_file(MODEL),
                "mapping": {"sign": SIGN.tolist(), "offset": OFFSET.tolist()},
            },
            "artifacts": {},
        }
    for shape, params in shapes.items():
        cfg = build_config(params)
        bundle = build_reference(cfg, model, KIN_HOME, DT, horizon=HORIZON, lookahead_steps=0)
        env = envelope_report(bundle)
        pos = position_distribution_report(bundle)
        clear = clearance_report(model, data, bundle)
        if not env["all_joints_at_or_below_p99"]:
            raise RuntimeError(f"{shape}: lap max |dq| exceeds measured P99 envelope")
        if not pos["all_joints_inside_collection_span"]:
            raise RuntimeError(f"{shape}: reference leaves the measured collection position span")
        if not clear["passes_offline_gate"]:
            raise RuntimeError(f"{shape}: clearance above guard plane not positive")

        shape_label = f"{shape}_p{args.phase_index}"
        out_dir = args.out_dir / shape_label
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = save_reference_bundle(bundle, out_dir)

        # Execution window only: the bundle tail carries horizon/lookahead
        # padding (execution_steps < reference_length); playback must not
        # replay padding rows as commands.
        q_ctrl = (bundle.q_des[: bundle.execution_steps] - OFFSET[None, :]) / SIGN[None, :]
        ctrl_path = out_dir / "q_des_ctrl.npy"
        np.save(ctrl_path, q_ctrl)

        validation = bundle.metadata.get("validation")
        manifest["artifacts"][shape_label] = {
            "shape": shape,
            "phase_index": args.phase_index,
            "start_phase_deg": args.start_phase,
            "lap_duration_s": float(params["lap_duration"]),
            "reference.npz": {"sha256": sha256_file(bundle_path), "bytes": bundle_path.stat().st_size},
            "q_des_ctrl.npy": {"sha256": sha256_file(ctrl_path), "bytes": ctrl_path.stat().st_size},
            "execution_steps": bundle.execution_steps,
            "reference_length": bundle.reference_length,
            "envelope": env,
            "position_distribution": pos,
            "clearance": clear,
            "validation": {
                "max_fk_position_error_m": validation.get("max_fk_position_error"),
                "min_task_sigma_min": validation.get("min_task_sigma_min"),
                "final_q_error_rad": validation.get("final_q_error"),
                "max_joint_velocity_rad_s": validation.get("max_joint_velocity"),
                "max_joint_acceleration_rad_s2": validation.get("max_joint_acceleration"),
                "lap_closure": validation.get("lap_closure"),
            },
            "config": asdict(cfg) if isinstance(cfg, ReferenceConfig) else str(cfg),
        }
        print(f"[{shape}] frozen: {bundle_path}")
        print(f"  execution_steps={bundle.execution_steps}  laps at/below P99: {env['all_joints_at_or_below_p99']}  "
              f"position in collection span: {pos['all_joints_inside_collection_span']}")
        print(f"  max lap |dq| deg/s: " + " ".join(f"{v:.2f}" for v in env["max_dq_lap_deg_s"].values()))
        print(f"  clearance above guard: {clear['clearance_above_guard_m'] * 1000:.1f} mm")

    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2, default=float) + "\n")
    print(f"manifest: {manifest_path} (sha256={sha256_file(manifest_path)[:16]}…)")


if __name__ == "__main__":
    main()
