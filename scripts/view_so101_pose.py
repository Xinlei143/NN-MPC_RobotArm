#!/usr/bin/env python3
"""Read-only MuJoCo viewer for the fine SO101 model at a given q_ctrl pose.

Opens the approved fine mesh scene (including the virtual table guard plane)
and places the arm at a specified q_ctrl joint configuration, applying the
profile's q_ctrl_to_q_kin mapping.  Never connects to hardware, never moves
anything; purely for visual inspection (e.g. checking a folded pose against
the real arm).
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
    parser = argparse.ArgumentParser(description="View the fine SO101 model at a specified q_ctrl pose (read-only).")
    parser.add_argument("--hardware-config", default="configs/hardware/so101_follower.local.yaml")
    parser.add_argument("--robot-config", default="configs/robots/so101.yaml")
    parser.add_argument("--table-safety-config", default="configs/hardware/so101_table_safety.mesh_guard20_runtime10.local.yaml")
    parser.add_argument("--q-deg", nargs=5, type=float, metavar=("pan", "lift", "elbow", "wrist_flex", "wrist_roll"),
                        default=None, help="q_ctrl joint angles in degrees (5 values); default is home.")
    parser.add_argument("--gripper-deg", type=float, default=None,
                        help="Fine-model gripper joint angle in degrees (range [-10, 100]); default leaves it at 0.")
    args = parser.parse_args()

    import mujoco
    import mujoco.viewer

    from mpc.robot_config import load_robot_spec
    from robot_runtime.config import load_hardware_config
    from robot_runtime.kinematics import gripper_model_angle_rad
    from robot_runtime.table_safety import load_table_safety_profile

    hw = load_hardware_config(args.hardware_config)
    robot = load_robot_spec(args.robot_config, validate_model=True)
    profile = load_table_safety_profile(
        args.table_safety_config, n_joints=len(hw.joint_names),
        expected_model_xml_sha256=robot.model_xml_sha256, require_approved=False)

    q_ctrl = (np.deg2rad(args.q_deg) if args.q_deg is not None
              else np.asarray(hw.home_q_ctrl, dtype=np.float64))
    q_kin = profile.mapping.to_kinematics(q_ctrl)
    # Default the gripper to the collection hold position so the model matches
    # the real arm; --gripper-deg overrides it.
    gripper_deg = (args.gripper_deg if args.gripper_deg is not None
                   else float(np.rad2deg(gripper_model_angle_rad(hw.gripper_hold_raw))))

    model = mujoco.MjModel.from_xml_path(str(profile.mesh_collision_model_xml))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    for name, value in zip(profile.mesh_collision_joint_names, q_kin):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
    if gripper_id < 0:
        raise ValueError("fine model has no 'gripper' joint")
    data.qpos[model.jnt_qposadr[gripper_id]] = float(np.deg2rad(gripper_deg))
    mujoco.mj_forward(model, data)

    print("Fine SO101 model viewer (read-only, no hardware connection).")
    print("q_ctrl(deg):", np.round(np.rad2deg(q_ctrl), 2).tolist())
    print("q_kin(rad): ", np.round(q_kin, 4).tolist())
    print(f"gripper(deg): {gripper_deg:.1f} (raw 2048 hold; use --gripper-deg to override)")
    print("Close the viewer window (or Ctrl+C in the terminal) to exit.")

    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        import time
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
