#!/usr/bin/env python3
"""Open the first mesh-unsafe point on the configured SO101 recovery path.

This is an offline MuJoCo viewer only.  It never opens the serial port or
communicates with the robot.  The displayed pose reconstructs the same
cosine-interpolated home-to-recovery path checked by the E3 collector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mpc.robot_config import load_robot_spec
from robot_runtime.backends.so101_backend import raw_to_degrees
from robot_runtime.config import load_hardware_config
from robot_runtime.table_safety import load_table_safety_profile, make_table_clearance_checker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline MuJoCo replay of the E3 recovery-path collision.")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--table-safety-config", required=True)
    parser.add_argument("--step", type=int, default=None,
                        help="Optional 0..N recovery-path sample; default is the first unsafe one.")
    parser.add_argument("--no-viewer", action="store_true", help="Print the reconstructed pose without opening a GUI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_hardware_config(args.hardware_config)
    if config.shutdown_recovery_raw is None:
        raise SystemExit("hardware config has no shutdown_recovery_raw")
    robot = load_robot_spec("configs/robots/so101.yaml", validate_model=True)
    profile = load_table_safety_profile(
        args.table_safety_config, n_joints=robot.n_joints,
        expected_model_xml_sha256=robot.model_xml_sha256, require_approved=True,
    )
    checker = make_table_clearance_checker(robot.model_xml, profile, robot.n_joints)
    if profile.mesh_collision_model_xml is None:
        raise SystemExit("this viewer requires a detailed mesh_collision table profile")

    names = (*config.joint_names, config.gripper_name)
    with config.calibration_path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    range_low = np.asarray([calibration[name]["range_min"] for name in names], dtype=np.int64)
    range_high = np.asarray([calibration[name]["range_max"] for name in names], dtype=np.int64)
    target_q = np.deg2rad(raw_to_degrees(
        config.shutdown_recovery_raw[:robot.n_joints], range_low[:robot.n_joints], range_high[:robot.n_joints]
    )).astype(np.float64)
    start_q = config.home_q_ctrl.astype(np.float64)
    steps = int(np.ceil(config.shutdown_recovery_seconds / config.control_dt))

    samples: list[tuple[int, float, np.ndarray, object]] = []
    for index in range(steps + 1):
        blend = .5 - .5 * np.cos(np.pi * index / steps)
        q = start_q + blend * (target_q - start_q)
        samples.append((index, blend, q, checker.evaluate(q)))
    if args.step is None:
        selected = next((sample for sample in samples if not sample[3].safe), None)
        if selected is None:
            raise SystemExit("the configured home-to-recovery path is mesh-safe; no collision pose to display")
    else:
        if not 0 <= args.step <= steps:
            raise SystemExit(f"--step must lie in [0, {steps}]")
        selected = samples[args.step]
    index, blend, q_ctrl, clearance = selected

    # Locate the exact mesh/plane nearest points used by mj_geomDistance so
    # the viewer can mark the sub-millimetre guard crossing visually.
    checker._forward(q_ctrl)  # Same detailed collision model as the safety gate.
    nearest_geom = -1
    nearest_fromto = np.zeros(6, dtype=np.float64)
    nearest_distance = float("inf")
    for geom_ids in checker._geom_ids_by_body.values():
        for geom_id in geom_ids:
            fromto = np.zeros(6, dtype=np.float64)
            distance = float(mujoco.mj_geomDistance(
                checker.model, checker.data, checker._floor_geom_id, geom_id, 1.0, fromto
            ))
            if distance < nearest_distance:
                nearest_distance, nearest_geom, nearest_fromto = distance, int(geom_id), fromto

    model = mujoco.MjModel.from_xml_path(str(profile.mesh_collision_model_xml))
    data = mujoco.MjData(model)
    for joint_name, value in zip(profile.mesh_collision_joint_names, q_ctrl, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)
    print({
        "offline_only": True,
        "path_step": index,
        "path_steps": steps,
        "path_time_s": index * config.control_dt,
        "blend": blend,
        "q_ctrl_deg": np.rad2deg(q_ctrl).tolist(),
        "effective_clearance_m": clearance.effective_clearance_m,
        "nearest_component": clearance.nearest_component,
        "nearest_collision_geom_id": nearest_geom,
        "nearest_points_m": {
            "guard_plane": nearest_fromto[:3].tolist(),
            "mesh_surface": nearest_fromto[3:].tolist(),
        },
        "safe": clearance.safe,
        "scene": str(profile.mesh_collision_model_xml),
    })
    if args.no_viewer:
        return
    print("Opening MuJoCo viewer. Red = mesh nearest point; blue = guard-plane nearest point.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            scene = viewer.user_scn
            scene.ngeom = 0
            for position, color in ((nearest_fromto[3:], (1.0, 0.05, 0.05, 1.0)),
                                    (nearest_fromto[:3], (0.05, 0.3, 1.0, 1.0))):
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.asarray([0.006, 0.0, 0.0]), np.asarray(position), np.eye(3).reshape(-1), color,
                )
                scene.ngeom += 1
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
