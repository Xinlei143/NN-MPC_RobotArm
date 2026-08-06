#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from robot_runtime.config import load_hardware_config
from robot_runtime.dataset import RealDatasetRecorder, TransitionValidity
from robot_runtime.excitation import ModelAExcitation
from robot_runtime.factory import make_so101_backend
from robot_runtime.runner import RealControlMode, RealTimeRunner
from mpc.robot_config import load_robot_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect aligned SO101 (x_k, transmitted_u_k, x_k+1) transitions.")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--motion-mode", choices=sorted(ModelAExcitation.MOTION_MODE_IDS), default="multi_joint_sine")
    parser.add_argument("--amplitude-deg", type=float, default=2.0)
    parser.add_argument("--joint", choices=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                        default="shoulder_pan")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--startup-seconds", type=float, default=3.0)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--operator-supported-shutdown", action="store_true")
    args = parser.parse_args()
    if not args.enable_motion: raise SystemExit("refusing motion without --enable-motion")
    if not args.operator_supported_shutdown: raise SystemExit("refusing torque enable without --operator-supported-shutdown")
    if args.minutes <= 0 or args.amplitude_deg <= 0 or args.amplitude_deg > 3:
        raise SystemExit("minutes must be positive and amplitude must be in (0, 3] degrees")
    if args.motion_mode in {"single_joint_sine", "hold_perturbation"} and args.amplitude_deg > 2:
        raise SystemExit("single-joint collection is limited to 2 degrees")
    config = load_hardware_config(args.hardware_config)
    robot = load_robot_spec("configs/robots/so101.yaml", validate_model=True)
    backend = make_so101_backend(args.hardware_config)
    session = args.session_id or str(uuid.uuid4())
    split_group = int.from_bytes(hashlib.sha256(session.encode()).digest()[:4], "big")
    recorder = RealDatasetRecorder(session, control_dt=config.control_dt)
    previous = None
    episode = args.episode_id
    nominal = ModelAExcitation(config.home_q_ctrl, config.control_dt, mode=args.motion_mode,
                               amplitude_rad=float(np.deg2rad(args.amplitude_deg)),
                               joint_index=config.joint_names.index(args.joint), seed=args.seed)
    def record(tick):
        nonlocal previous
        if bool(tick.state.diagnostics.get("goal_readback_mismatch", False)):
            first_uncertain = int(tick.state.diagnostics.get("last_matching_goal_readback_tick", -1)) + 1
            recorder.mark_command_uncertain_since_tick(first_uncertain)
        if previous is not None:
            dt = (tick.state.timestamp_ns - previous.state.timestamp_ns) * 1e-9
            validity = TransitionValidity(abs(dt - config.control_dt) / config.control_dt <= .10,
                                          previous.command.command_delivery_uncertain)
            recorder.append(previous.state, previous.command, tick.state, episode_id=episode,
                            split_group_id=split_group, motion_mode_id=nominal.motion_mode_id, validity=validity)
        previous = tick
    runner = RealTimeRunner(backend, nominal, mode=RealControlMode.DIRECT)
    try:
        backend.connect()
        backend.startup_to_home(args.startup_seconds)
        runner.run(int(args.minutes * 60 / config.control_dt), on_tick=record)
        backend.move_to_configuration(config.home_q_ctrl, 3.0)
        recorder.save(args.output, {"created_utc": datetime.now(timezone.utc).isoformat(),
                                    "robot_identity": robot.artifact_identity(),
                                    "plant_identity": config.plant_identity(), "collection_mode": "model_a_excitation",
                                    "motion_mode": args.motion_mode, "motion_mode_id": nominal.motion_mode_id,
                                    "amplitude_deg": args.amplitude_deg, "joint": args.joint, "seed": args.seed,
                                    "episode_id": episode, "hardware_config_path": str(Path(args.hardware_config).resolve())})
    finally:
        backend.close()


if __name__ == "__main__": main()
