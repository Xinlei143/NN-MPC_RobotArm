#!/usr/bin/env python3
"""Offline table-clearance preflight for one deterministic E3 workspace session."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mpc.robot_config import load_robot_spec
from robot_runtime.config import load_hardware_config
from robot_runtime.kinematics import gripper_model_angle_rad
from robot_runtime.table_safety import load_table_safety_profile, make_table_clearance_checker
from robot_runtime.workspace_model_a import (
    build_workspace_reference,
    effective_workspace_bounds,
    expected_model_a_split,
    model_a_collection_plan_identity,
    session_program,
)


def cosine_path(start: np.ndarray, target: np.ndarray, duration_s: float, dt: float) -> np.ndarray:
    steps = max(1, int(np.ceil(duration_s / dt)))
    alpha = .5 - .5 * np.cos(np.pi * np.arange(steps + 1)[:, None] / steps)
    return np.asarray(start)[None, :] + alpha * (np.asarray(target) - np.asarray(start))[None, :]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and table-check one E3 session without hardware access.")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--table-safety-config", required=True)
    parser.add_argument("--session-index", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace-margin-deg", type=float, default=1.0)
    parser.add_argument("--preposition-seconds", type=float, default=8.0)
    parser.add_argument("--max-resamples", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        expected_split = expected_model_a_split(args.session_index)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.workspace_margin_deg <= 0 or args.max_resamples <= 0:
        raise SystemExit("invalid workspace margin or max resamples")
    config = load_hardware_config(args.hardware_config)
    if config.hardware_joint_low is None or config.hardware_joint_high is None:
        raise SystemExit("preflight requires hardware_joint_low/high")
    robot = load_robot_spec("configs/robots/so101.yaml", validate_model=True)
    profile = load_table_safety_profile(args.table_safety_config, n_joints=robot.n_joints,
                                        expected_model_xml_sha256=robot.model_xml_sha256)
    if profile.mesh_collision_model_xml is None:
        raise SystemExit(
            "E3 preflight requires an approved detailed mesh_collision profile; "
            "the legacy capsule table profile is not sufficient for real-hardware clearance."
        )
    # Offline MuJoCo screening uses the stricter planning threshold, while
    # the collector separately applies the runtime threshold to measured q.
    checker = make_table_clearance_checker(robot.model_xml, profile.for_offline_planning(), robot.n_joints,
                                           gripper_model_angle_rad=gripper_model_angle_rad(config.gripper_hold_raw))
    low, high = effective_workspace_bounds(config.hardware_joint_low, config.hardware_joint_high,
                                           np.deg2rad(args.workspace_margin_deg))
    checker.require_safe(config.home_q_ctrl)
    rng = np.random.default_rng(args.seed + args.session_index * 1009)
    current = config.home_q_ctrl.copy()
    report_segments = []
    overall = checker.evaluate(current)
    for segment in session_program(args.session_index):
        last_error: Exception | None = None
        for attempt in range(1, args.max_resamples + 1):
            candidate = build_workspace_reference(
                rng=rng, mode=segment.mode, start_q=current, low=low, high=high,
                steps=int(round(segment.seconds / config.control_dt)), dt=config.control_dt,
                velocity_limit=.70 * config.command_velocity_limit,
                acceleration_limit=.70 * config.command_acceleration_limit,
                single_joint_index=segment.single_joint_index,
            )
            try:
                preposition = checker.require_safe_sequence(
                    cosine_path(current, candidate[0], args.preposition_seconds, config.control_dt)
                )
                trajectory = checker.require_safe_sequence(candidate)
                segment_min = min((preposition, trajectory), key=lambda result: result.effective_clearance_m)
                overall = min((overall, segment_min), key=lambda result: result.effective_clearance_m)
                report_segments.append({
                    "name": segment.name, "mode": segment.mode, "attempt": attempt,
                    "min_predicted_clearance_m": segment_min.predicted_clearance_m,
                    "min_effective_clearance_m": segment_min.effective_clearance_m,
                    "nearest_component": segment_min.nearest_component,
                })
                current = candidate[-1]
                break
            except ValueError as exc:
                last_error = exc
        else:
            raise SystemExit(f"unable to generate a table-safe {segment.name} segment: {last_error}")
    return_path = checker.require_safe_sequence(cosine_path(current, config.home_q_ctrl,
                                                            args.preposition_seconds, config.control_dt))
    overall = min((overall, return_path), key=lambda result: result.effective_clearance_m)
    payload = {
        "status": "pass", "session_index": args.session_index, "split": expected_split, "seed": args.seed,
        "collection_plan": model_a_collection_plan_identity(),
        "table_safety": profile.identity(), "segments": report_segments,
        "overall_min_predicted_clearance_m": overall.predicted_clearance_m,
        "overall_min_effective_clearance_m": overall.effective_clearance_m,
        "overall_nearest_component": overall.nearest_component,
        "note": "Offline preflight validates home and planned session paths; live startup begins from an unknown measured pose and is rechecked at runtime.",
    }
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite preflight evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
