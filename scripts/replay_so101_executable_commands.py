#!/usr/bin/env python3
"""Replay a real SO101 log through the canonical executable command state machine."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_runtime.config import load_hardware_config
from robot_runtime.executable_command import (
    ExecutableCommandState,
    make_executable_command_spec,
    step_executable_command_np,
)


def _calibration(path: Path, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        np.asarray([payload[name]["range_min"] for name in names], dtype=np.float64),
        np.asarray([payload[name]["range_max"] for name in names], dtype=np.float64),
    )


def _p95(values: np.ndarray) -> list[float]:
    return np.percentile(np.abs(values), 95, axis=0).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--hardware-config", default="configs/hardware/so101_follower.local.yaml")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rollout_path = Path(args.rollout)
    hardware = load_hardware_config(args.hardware_config)
    calibration_low, calibration_high = _calibration(hardware.calibration_path, hardware.joint_names)
    if hardware.hardware_joint_low is None or hardware.hardware_joint_high is None:
        raise ValueError("hardware config must define hardware_joint_low/high")
    spec = make_executable_command_spec(
        joint_low=hardware.hardware_joint_low,
        joint_high=hardware.hardware_joint_high,
        velocity_limit=hardware.command_velocity_limit,
        acceleration_limit=hardware.command_acceleration_limit,
        relative_limit=hardware.hardware_joint_high - hardware.hardware_joint_low,
        raw_low=hardware.raw_low[:5], raw_high=hardware.raw_high[:5],
        calibration_low=calibration_low, calibration_high=calibration_high,
        control_dt=hardware.control_dt, braking=True,
    )
    with np.load(rollout_path, allow_pickle=False) as archive:
        required = {"actual_states", "requested_absolute_command", "actuator_q_ref"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"rollout is missing {sorted(missing)}")
        states = np.asarray(archive["actual_states"], dtype=np.float32)
        requested = np.asarray(archive["requested_absolute_command"], dtype=np.float32)
        logged_transmitted = np.asarray(archive["actuator_q_ref"], dtype=np.float32)
        logged_projected = np.asarray(archive["projected_absolute_command"], dtype=np.float32) if "projected_absolute_command" in archive.files else np.full_like(requested, np.nan)
        if "transmitted_goal_position_raw" in archive.files:
            # Real SO101 packets contain five arm joints plus the independent
            # gripper Goal_Position.  The canonical executable state machine
            # intentionally models only the five MPC joints, so compare the
            # arm prefix and leave the gripper out of the parity calculation.
            logged_raw = np.asarray(archive["transmitted_goal_position_raw"], dtype=np.int64)
            if logged_raw.ndim != 2 or logged_raw.shape[1] < 5:
                raise ValueError("transmitted_goal_position_raw must have at least five motor columns")
            logged_raw = logged_raw[:, :5]
        else:
            logged_raw = None
        planner_expected_raw = (
            np.asarray(archive["planner_expected_raw"], dtype=np.int64)
            if "planner_expected_raw" in archive.files else None
        )
    n = min(len(states), len(requested), len(logged_transmitted))
    if n == 0 or states.shape[1] != 10 or requested.shape[1] != 5:
        raise ValueError("rollout arrays have incompatible shapes")
    # The first logged command is the first post-startup transmission.  The
    # backend's previous transmitted command is the home hold, which is the
    # same quantized target in the frozen SO101 runs; anchoring to the measured
    # gravity-deflected pose would invent a velocity that did not exist in the
    # real projector state.
    state = ExecutableCommandState.anchored(logged_transmitted[0])
    replay_transmitted, replay_projected, replay_raw, replay_velocity = [], [], [], []
    flags: list[tuple[str, ...]] = []
    for index in range(n):
        result = step_executable_command_np(
            requested[index], states[index, :5], state, spec,
        )
        replay_transmitted.append(result.transmitted_q_ref)
        replay_projected.append(result.projected_q_ref)
        replay_raw.append(result.tx_goal_position_raw)
        replay_velocity.append(result.command_velocity)
        flags.append(result.projection_flags)
        state = result.next_state
    replay_transmitted = np.stack(replay_transmitted)
    replay_projected = np.stack(replay_projected)
    replay_raw = np.stack(replay_raw)
    replay_velocity = np.stack(replay_velocity)
    tx_error = replay_transmitted - logged_transmitted[:n]
    projected_error = replay_projected - logged_projected[:n] if np.all(np.isfinite(logged_projected[:n])) else np.full_like(replay_projected, np.nan)
    if logged_raw is not None:
        raw_error = replay_raw - logged_raw[:n]
        raw_source = "logged_transmitted_goal_position_raw"
        raw_exact = bool(np.array_equal(replay_raw, logged_raw[:n]))
    else:
        midpoint = (calibration_low + calibration_high) / 2.0
        inferred = np.trunc(np.rad2deg(logged_transmitted[:n]) * 4095.0 / 360.0 + midpoint).astype(np.int64)
        raw_error = replay_raw - inferred
        raw_source = "inferred_from_float_actuator_q_ref"
        raw_exact = bool(np.array_equal(replay_raw, inferred))
    planner_expected_match = None
    planner_expected_count = 0
    planner_expected_nonzero_error = 0
    planner_expected_max_abs_count_error = None
    if planner_expected_raw is not None:
        expected = planner_expected_raw[:n]
        valid_expected = np.all(expected >= 0, axis=1)
        planner_expected_count = int(np.sum(valid_expected))
        if planner_expected_count:
            expected_error = replay_raw[valid_expected] - expected[valid_expected]
            planner_expected_nonzero_error = int(np.sum(np.any(expected_error != 0, axis=1)))
            planner_expected_max_abs_count_error = int(np.max(np.abs(expected_error)))
            planner_expected_match = planner_expected_nonzero_error == 0
    report = {
        "rollout": str(rollout_path.resolve()),
        "ticks": int(n),
        "raw_comparison_source": raw_source,
        "raw_exact": raw_exact,
        "raw_max_abs_count_error": int(np.max(np.abs(raw_error))),
        "raw_nonzero_tick_count": int(np.sum(np.any(raw_error != 0, axis=1))),
        "planner_expected_raw_count": planner_expected_count,
        "planner_expected_raw_exact": planner_expected_match,
        "planner_expected_raw_nonzero_tick_count": planner_expected_nonzero_error,
        "planner_expected_raw_max_abs_count_error": planner_expected_max_abs_count_error,
        "request_to_canonical_transmitted_p95_deg": np.rad2deg(_p95(replay_transmitted - requested[:n])).tolist(),
        "canonical_to_logged_transmitted_p95_deg": np.rad2deg(_p95(tx_error)).tolist(),
        "canonical_to_logged_projected_p95_deg": None if not np.any(np.isfinite(projected_error)) else np.rad2deg(_p95(projected_error)).tolist(),
        "canonical_to_logged_transmitted_max_deg": float(np.rad2deg(np.max(np.abs(tx_error)))),
        "projection_flag_counts": {
            flag: int(sum(flag in tick_flags for tick_flags in flags))
            for flag in sorted({flag for tick_flags in flags for flag in tick_flags})
        },
        "command_velocity_p95_rad_s": _p95(replay_velocity),
        "state_machine": "hardware_bounds -> measured-relative -> velocity -> acceleration -> braking -> raw quantization -> transmitted velocity",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "replay.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output_dir / "replay_arrays.npz", replay_transmitted_q_ref=replay_transmitted,
        replay_projected_q_ref=replay_projected, replay_goal_position_raw=replay_raw,
        replay_command_velocity=replay_velocity, logged_transmitted_q_ref=logged_transmitted[:n],
        raw_error=raw_error,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
