#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from robot_runtime.config import load_hardware_config
from robot_runtime.excitation import SafeExcitation
from robot_runtime.factory import make_so101_backend
from robot_runtime.runner import RealControlMode, RealTimeRunner


class SingleJointExcitation:
    """One-joint sinusoid for the mandatory direction and small-motion test."""

    def __init__(self, home: np.ndarray, control_dt: float, joint_index: int, amplitude_rad: float):
        self.home = np.asarray(home, dtype=np.float32)
        self.dt, self.joint_index, self.amplitude = float(control_dt), int(joint_index), float(amplitude_rad)

    def __call__(self, tick: int, state: object) -> np.ndarray:
        target = self.home.copy()
        target[self.joint_index] += self.amplitude * np.sin(2 * np.pi * 0.10 * tick * self.dt)
        return target


class SingleJointStep:
    """One signed held offset for the B4 direction check."""

    def __init__(self, home: np.ndarray, joint_index: int, amplitude_rad: float, direction: float):
        self.target = np.asarray(home, dtype=np.float32).copy()
        self.target[int(joint_index)] += float(direction) * float(amplitude_rad)

    def __call__(self, tick: int, state: object) -> np.ndarray:
        return self.target.copy()


def save_evidence(path: str | Path, records: list, config, *, motion_mode: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "states": np.asarray([record.state.vector for record in records], dtype=np.float32),
        "actions": np.asarray([record.command.transmitted_q_ref for record in records], dtype=np.float32),
        "requested_q_ref": np.asarray([record.command.requested_q_ref for record in records], dtype=np.float32),
        "projected_q_ref": np.asarray([record.command.projected_q_ref for record in records], dtype=np.float32),
        "transmitted_q_ref": np.asarray([record.command.transmitted_q_ref for record in records], dtype=np.float32),
        "raw_goal": np.asarray([record.command.tx_goal_position_raw for record in records], dtype=np.int64),
        "state_timestamp_ns": np.asarray([record.state.timestamp_ns for record in records], dtype=np.int64),
        "wake_lateness_s": np.asarray([record.wake_lateness_s for record in records], dtype=np.float64),
        "skipped_ticks": np.asarray([record.skipped_ticks for record in records], dtype=np.int64),
        "motor_voltage_v": np.asarray([record.state.diagnostics.get("motor_voltage_v", np.full(6, np.nan)) for record in records]),
        "motor_temperature_c": np.asarray([record.state.diagnostics.get("motor_temperature", np.full(6, np.nan)) for record in records]),
        # STS3215 exposes these diagnostics as device-native raw register
        # values.  Preserve them verbatim for D3 thermal/current review;
        # unlike voltage, no physical-unit conversion is assumed here.
        "motor_current_raw": np.asarray([record.state.diagnostics.get("motor_current_raw", np.full(6, np.nan)) for record in records]),
        "motor_load_raw": np.asarray([record.state.diagnostics.get("motor_load_raw", np.full(6, np.nan)) for record in records]),
        "diagnostic_sample_age_s": np.asarray([record.state.diagnostics.get("diagnostic_sample_age_s", np.nan) for record in records]),
    }
    np.savez_compressed(target, **arrays)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "motion_mode": motion_mode,
        "sample_count": len(records), "plant_identity": config.plant_identity(),
        "dataset": target.name, "action_semantics": "software_transmitted_absolute_position_target",
        "motor_acknowledged": False,
    }
    target.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_mirror(args: argparse.Namespace, config) -> object | None:
    """Build the passive fine-model MuJoCo mirror, or None if no DISPLAY.

    The mirror renders each measured real state through the profile mapping,
    optionally stacked with a rendering-only ``--mapping-preview-offset-rad``
    so an operator can visually tune ``joint_offset`` without editing files.
    It is purely observational and never affects commands or safety gates.
    """
    from mpc.robot_config import load_robot_spec
    from robot_runtime.kinematics import JointCoordinateMapping, gripper_model_angle_rad
    from robot_runtime.mujoco_mirror import MuJoCoMirror
    from robot_runtime.table_safety import load_table_safety_profile

    robot = load_robot_spec(args.robot_config, validate_model=True)
    profile = load_table_safety_profile(
        args.table_safety_config, n_joints=len(config.joint_names),
        expected_model_xml_sha256=robot.model_xml_sha256, require_approved=False)
    mapping = profile.mapping
    if args.mapping_preview_offset_rad is not None:
        mapping = JointCoordinateMapping(
            mapping.joint_sign,
            mapping.joint_offset + np.asarray(args.mapping_preview_offset_rad, dtype=np.float32),
            source=f"preview_over_profile:{mapping.source}")
    mirror = MuJoCoMirror(profile, len(config.joint_names), mapping=mapping,
                          gripper_model_angle_rad=gripper_model_angle_rad(config.gripper_hold_raw))
    print("MuJoCo mirror mapping (joint_sign, joint_offset_rad):")
    print(f"  sign  = {mapping.joint_sign.tolist()}")
    print(f"  offset= {[round(float(v), 6) for v in mapping.joint_offset.tolist()]}")
    if args.mapping_preview_offset_rad is not None:
        print(f"  note  = rendering-only preview offset applied over profile: "
              f"{[round(v, 6) for v in args.mapping_preview_offset_rad]}")
    return mirror


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conservative joint-space direct control on SO101.")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--amplitude-deg", type=float, default=2.0)
    parser.add_argument("--joint", choices=["all", "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                        default="all", help="Use one named joint for B4; all is only permitted after B4.")
    parser.add_argument("--direction", choices=["positive", "negative", "oscillate"], default="oscillate",
                        help="For a single joint, use positive/negative for B4 or oscillate only after direction checks.")
    parser.add_argument("--output", help="Evidence NPZ path; required for a formal direct baseline.")
    parser.add_argument("--startup-seconds", type=float, default=3.0)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--operator-supported-shutdown", action="store_true")
    parser.add_argument("--robot-config", default="configs/robots/so101.yaml",
                        help="Robot spec used to validate the table-safety profile hash (with --visualize-mujoco).")
    parser.add_argument("--table-safety-config",
                        help="Table-safety profile with mesh_collision (required with --visualize-mujoco).")
    parser.add_argument("--visualize-mujoco", action="store_true",
                        help="Open a passive fine-model MuJoCo mirror driven by each measured state.")
    parser.add_argument("--mapping-preview-offset-rad", nargs=5, type=float, metavar=("j0", "j1", "j2", "j3", "j4"),
                        help="Rendering-only joint_offset (5 floats) added on top of the profile mapping for visual "
                             "preview; requires --visualize-mujoco. Does not change commands or the profile.")
    args = parser.parse_args()
    if not args.enable_motion: raise SystemExit("refusing motion without --enable-motion")
    if not args.operator_supported_shutdown: raise SystemExit("refusing torque enable without --operator-supported-shutdown")
    config = load_hardware_config(args.hardware_config)
    if args.amplitude_deg <= 0 or args.amplitude_deg > 3.0: raise SystemExit("initial direct-control amplitude must be in (0, 3] degrees")
    if args.joint != "all" and args.amplitude_deg > 2.0:
        raise SystemExit("single-joint direction testing is limited to 2 degrees")
    if args.joint == "all" and args.direction != "oscillate":
        raise SystemExit("--direction applies only to a single joint")
    if args.visualize_mujoco and not args.table_safety_config:
        raise SystemExit("--visualize-mujoco requires --table-safety-config")
    if args.mapping_preview_offset_rad is not None and not args.visualize_mujoco:
        raise SystemExit("--mapping-preview-offset-rad requires --visualize-mujoco")
    backend = make_so101_backend(args.hardware_config)
    if args.joint == "all":
        nominal = SafeExcitation(config.home_q_ctrl, config.control_dt, np.deg2rad(args.amplitude_deg))
    elif args.direction == "oscillate":
        nominal = SingleJointExcitation(config.home_q_ctrl, config.control_dt,
                                        config.joint_names.index(args.joint), np.deg2rad(args.amplitude_deg))
    else:
        nominal = SingleJointStep(config.home_q_ctrl, config.joint_names.index(args.joint), np.deg2rad(args.amplitude_deg),
                                  1.0 if args.direction == "positive" else -1.0)
    mirror: object | None = None
    if args.visualize_mujoco:
        try:
            mirror = build_mirror(args, config)
        except RuntimeError as exc:
            print(f"warning: {exc}; continuing without the MuJoCo mirror", file=sys.stderr)
            mirror = None

    def on_tick(record) -> None:
        if mirror is not None and record.state.valid:
            mirror.update(record.state.q_ctrl)

    try:
        backend.connect()
        backend.startup_to_home(args.startup_seconds)
        records = RealTimeRunner(backend, nominal, mode=RealControlMode.DIRECT).run(
            int(args.seconds / config.control_dt), on_tick=on_tick if mirror is not None else None)
        backend.move_to_configuration(config.home_q_ctrl, 3.0)
        if args.output:
            mode = "multi_joint_sine" if args.joint == "all" else f"single_joint_{args.joint}_{args.direction}"
            save_evidence(args.output, records, config, motion_mode=mode)
    finally:
        if mirror is not None:
            mirror.close()
        backend.close()


if __name__ == "__main__": main()
