#!/usr/bin/env python3
"""Collect one audited 15-minute real-SO101 Model-A workspace session.

This is deliberately separate from ``collect_real_data.py``: the latter is a
small-envelope smoke collector, while this program uses the manually verified
hardware envelope with a one-degree internal inset.  It is direct control only
and cannot invoke MPC.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynamics_modeling.scripts.validate_real_dataset import validate_dataset
from mpc.robot_config import load_robot_spec
from robot_runtime.config import load_hardware_config
from robot_runtime.dataset import RealDatasetRecorder, TransitionValidity
from robot_runtime.factory import make_so101_backend
from robot_runtime.kinematics import gripper_model_angle_rad
from robot_runtime.real_dataset_store import append_completed_session
from robot_runtime.runner import RealControlMode, RealTimeRunner
from robot_runtime.table_safety import load_table_safety_profile, make_table_clearance_checker
from robot_runtime.workspace_model_a import (
    build_workspace_reference,
    effective_workspace_bounds,
    expected_model_a_split,
    model_a_collection_plan_identity,
    session_program,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect one strict-append real workspace Model-A session.")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--output", required=True, help="Canonical strict-append dataset (.npz).")
    parser.add_argument("--session-index", required=True, type=int, help="0..47 for the fixed 38/5/5 split.")
    parser.add_argument("--session-id", default=None, help="Unique audit label; default derives from session index.")
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workspace-margin-deg", type=float, default=1.0)
    parser.add_argument("--table-safety-config", required=True,
                        help="Approved local table-clearance profile; full workspace collection refuses to run without it.")
    parser.add_argument("--table-max-resamples", type=int, default=100)
    parser.add_argument("--startup-seconds", type=float, default=3.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--home-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--preposition-seconds", type=float, default=8.0)
    parser.add_argument(
        "--visualize-mujoco", action="store_true",
        help=("Open a passive detailed-MuJoCo mirror driven by each valid measured real-arm state. "
              "Diagnostic only: it does not change commands or safety gates."),
    )
    parser.add_argument("--append", action="store_true", help="Required: create or strictly append the canonical dataset.")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--operator-supported-shutdown", action="store_true")
    return parser.parse_args()


def _session_shard_path(canonical: Path, session_id: str) -> Path:
    return canonical.parent / "sessions" / f"{canonical.stem}_{session_id}.npz"


class LiveMuJoCoMirror:
    """Passive detailed-model viewer driven only by measured SO101 positions.

    The viewer owns a separate MuJoCo model/data pair so rendering cannot
    mutate the model/data pair used by the table-clearance safety checker.
    It is deliberately observational: closing its window never interrupts a
    real collection and it never contributes to command generation.
    """

    def __init__(self, table_profile, n_joints: int, *, gripper_model_angle_rad: float | None = None) -> None:
        if table_profile.mesh_collision_model_xml is None:
            raise ValueError("a detailed mesh collision model is required for live visualization")
        try:
            import mujoco
            import mujoco.viewer
        except Exception as exc:
            raise RuntimeError("--visualize-mujoco requires the Python MuJoCo viewer package") from exc
        self._mujoco = mujoco
        self._profile = table_profile
        self._model = mujoco.MjModel.from_xml_path(str(table_profile.mesh_collision_model_xml))
        self._data = mujoco.MjData(self._model)
        addresses: list[int] = []
        for name in table_profile.mesh_collision_joint_names:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"viewer model has no joint {name!r}")
            addresses.append(int(self._model.jnt_qposadr[joint_id]))
        if len(addresses) != n_joints:
            raise ValueError("viewer joint count does not match controlled joint count")
        self._qpos_addresses = np.asarray(addresses, dtype=np.intp)
        self._gripper_angle: float | None = gripper_model_angle_rad
        self._gripper_qposadr: int | None = None
        if gripper_model_angle_rad is not None:
            gripper_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
            if gripper_id >= 0:
                self._gripper_qposadr = int(self._model.jnt_qposadr[gripper_id])
        try:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        except Exception as exc:
            raise RuntimeError(
                "Could not open the MuJoCo mirror. Use a local graphical DISPLAY or omit --visualize-mujoco."
            ) from exc
        self._closed_notice_emitted = False

    def update(self, q_ctrl: np.ndarray) -> None:
        """Render one valid measured position without affecting collection."""
        if not self._viewer.is_running():
            if not self._closed_notice_emitted:
                print("MuJoCo mirror window was closed; real collection continues without visualization.")
                self._closed_notice_emitted = True
            return
        values = np.asarray(q_ctrl, dtype=np.float64)
        if values.shape != (len(self._qpos_addresses),) or not np.all(np.isfinite(values)):
            return
        with self._viewer.lock():
            self._mujoco.mj_resetData(self._model, self._data)
            self._data.qpos[self._qpos_addresses] = self._profile.mapping.to_kinematics(values)
            if self._gripper_qposadr is not None and self._gripper_angle is not None:
                self._data.qpos[self._gripper_qposadr] = self._gripper_angle
            self._data.qvel[:] = 0.0
            self._mujoco.mj_forward(self._model, self._data)
        self._viewer.sync()

    def close(self) -> None:
        try:
            self._viewer.close()
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    if not args.enable_motion:
        raise SystemExit("refusing physical motion without --enable-motion")
    if not args.operator_supported_shutdown:
        raise SystemExit("refusing torque enable without --operator-supported-shutdown")
    if not args.append:
        raise SystemExit("workspace collection requires --append; canonical data is never overwritten")
    if args.workspace_margin_deg <= 0 or args.preposition_seconds <= 0:
        raise SystemExit("workspace margin and preposition duration must be positive")
    if args.table_max_resamples <= 0:
        raise SystemExit("--table-max-resamples must be positive")
    try:
        expected_split = expected_model_a_split(args.session_index)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.split != expected_split:
        raise SystemExit(f"session index {args.session_index} belongs to split {expected_split!r}, not {args.split!r}")

    config = load_hardware_config(args.hardware_config)
    if config.hardware_joint_low is None or config.hardware_joint_high is None:
        raise SystemExit("workspace collection requires verified hardware_joint_low/high in the hardware config")
    robot = load_robot_spec("configs/robots/so101.yaml", validate_model=True)
    table_profile = load_table_safety_profile(
        args.table_safety_config, n_joints=robot.n_joints,
        expected_model_xml_sha256=robot.model_xml_sha256, require_approved=True,
    )
    if table_profile.mesh_collision_model_xml is None:
        raise SystemExit(
            "E3 workspace collection requires an approved detailed mesh_collision profile; "
            "the legacy capsule table profile is not sufficient for real-hardware clearance."
        )
    runtime_table_checker = make_table_clearance_checker(
        robot.model_xml, table_profile, robot.n_joints,
        gripper_model_angle_rad=gripper_model_angle_rad(config.gripper_hold_raw))
    planning_table_checker = make_table_clearance_checker(
        robot.model_xml, table_profile.for_offline_planning(), robot.n_joints,
        gripper_model_angle_rad=gripper_model_angle_rad(config.gripper_hold_raw))
    mirror = LiveMuJoCoMirror(table_profile, robot.n_joints,
                              gripper_model_angle_rad=gripper_model_angle_rad(config.gripper_hold_raw)
                              ) if args.visualize_mujoco else None
    collection_plant_identity = {**config.plant_identity(), "table_safety": table_profile.identity()}
    low, high = effective_workspace_bounds(
        config.hardware_joint_low, config.hardware_joint_high, np.deg2rad(args.workspace_margin_deg)
    )
    if np.any(config.home_q_ctrl < low) or np.any(config.home_q_ctrl > high):
        raise SystemExit("home must remain inside the requested effective workspace")
    if not (np.allclose(robot.data_reset_low, low, atol=1e-6)
            and np.allclose(robot.data_reset_high, high, atol=1e-6)
            and np.allclose(robot.data_target_low, low, atol=1e-6)
            and np.allclose(robot.data_target_high, high, atol=1e-6)):
        raise SystemExit(
            "configs/robots/so101.yaml Model-A reset/target bounds do not match "
            "hardware_joint_low/high minus the requested workspace margin"
        )

    canonical = Path(args.output).expanduser().resolve()
    session_id = args.session_id or f"workspace_{args.session_index:02d}_{uuid.uuid4().hex[:8]}"
    shard = _session_shard_path(canonical, session_id)
    if shard.exists() or shard.with_suffix(".manifest.json").exists() or shard.with_suffix(".failure.json").exists():
        raise SystemExit(f"refusing to overwrite existing session evidence: {shard}")

    program = session_program(args.session_index)
    split_group = int(args.session_index)
    rng = np.random.default_rng(args.seed + args.session_index * 1009)
    recorder = RealDatasetRecorder(session_id, control_dt=config.control_dt)
    backend = make_so101_backend(args.hardware_config)
    completed = False
    status = "failed"
    failure: str | None = None
    total_ticks = 0
    recovery: dict[str, object] = {
        "status": "not_attempted",
        "target_raw": (None if config.shutdown_recovery_raw is None
                       else config.shutdown_recovery_raw.astype(int).tolist()),
    }

    def safe_segment(segment, start_q: np.ndarray) -> tuple[np.ndarray, object]:
        """Resample an entire candidate until every 30 Hz point clears the table."""
        last_error: Exception | None = None
        for _attempt in range(args.table_max_resamples):
            sequence = build_workspace_reference(
                rng=rng, mode=segment.mode, start_q=start_q, low=low, high=high,
                steps=int(round(segment.seconds / config.control_dt)), dt=config.control_dt,
                velocity_limit=0.70 * config.command_velocity_limit,
                acceleration_limit=0.70 * config.command_acceleration_limit,
                single_joint_index=segment.single_joint_index,
            )
            try:
                return sequence, planning_table_checker.require_safe_sequence(sequence)
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f"unable to generate a table-safe {segment.name} segment after "
                           f"{args.table_max_resamples} attempts: {last_error}")

    def guarded_shutdown_recovery() -> dict[str, object]:
        """Try the configured return pose only while every safety prerequisite holds."""
        if config.shutdown_recovery_raw is None:
            return {"status": "skipped", "reason": "shutdown_recovery_raw_not_configured"}
        if not backend.is_connected:
            return {"status": "skipped", "reason": "hardware_not_connected",
                    "target_raw": config.shutdown_recovery_raw.astype(int).tolist()}
        try:
            state = backend.read_state(tick_index=0)
        except Exception as exc:
            return {"status": "skipped", "reason": f"state_read_failed: {type(exc).__name__}: {exc}",
                    "target_raw": config.shutdown_recovery_raw.astype(int).tolist()}
        if not state.valid:
            return {"status": "skipped", "reason": f"state_invalid: {list(state.validity_flags)}",
                    "target_raw": config.shutdown_recovery_raw.astype(int).tolist(),
                    "present_raw": np.asarray(state.diagnostics.get("present_position_raw", [])).astype(int).tolist()}
        diagnostics = state.diagnostics
        failed_reads = sorted(key for key, value in diagnostics.items()
                              if key.endswith("_read_failed") and bool(value))
        age = float(diagnostics.get("diagnostic_sample_age_s", np.inf))
        temperatures = np.asarray(diagnostics.get("motor_temperature", []), dtype=np.float64)
        if failed_reads or not np.isfinite(age) or age > 3.0:
            return {"status": "skipped", "reason": "diagnostics_not_fresh",
                    "failed_reads": failed_reads, "diagnostic_sample_age_s": age,
                    "target_raw": config.shutdown_recovery_raw.astype(int).tolist()}
        if temperatures.shape != (6,) or not np.all(np.isfinite(temperatures)):
            return {"status": "skipped", "reason": "temperature_diagnostics_unavailable",
                    "motor_temperature_c": temperatures.tolist(),
                    "target_raw": config.shutdown_recovery_raw.astype(int).tolist()}
        if float(np.max(temperatures)) >= 50.0:
            return {"status": "skipped", "reason": "temperature_not_normal",
                    "motor_temperature_c": temperatures.tolist(),
                    "target_raw": config.shutdown_recovery_raw.astype(int).tolist()}
        if bool(diagnostics.get("command_delivery_uncertain", False)) or bool(diagnostics.get("goal_readback_mismatch", False)):
            return {"status": "skipped", "reason": "command_delivery_not_healthy",
                    "target_raw": config.shutdown_recovery_raw.astype(int).tolist()}
        try:
            result = backend.move_to_shutdown_recovery_raw(
                config.shutdown_recovery_raw, config.shutdown_recovery_seconds,
                path_validator=runtime_table_checker.require_safe,
            )
        except Exception as exc:
            return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                    "target_raw": config.shutdown_recovery_raw.astype(int).tolist()}
        return {"status": "completed", "motor_temperature_c": temperatures.tolist(), **result}

    try:
        backend.connect()
        backend.startup_to_home(
            args.startup_seconds,
            convergence_timeout_s=args.startup_timeout_seconds,
            home_tolerance_rad=np.deg2rad(args.home_tolerance_deg),
            path_validator=runtime_table_checker.require_safe,
        )
        if mirror is not None:
            startup_state = backend.read_state(tick_index=0)
            if startup_state.valid:
                mirror.update(startup_state.q_ctrl)
        current_q = config.home_q_ctrl.copy()
        for segment_index, segment in enumerate(program):
            steps = int(round(segment.seconds / config.control_dt))
            sequence, _segment_clearance = safe_segment(segment, current_q)
            # The preposition is intentionally outside the dataset.  Each
            # recorded episode therefore has one stable motion semantics.
            backend.prepare_hardware_motion()
            backend.move_to_configuration(sequence[0], args.preposition_seconds, envelope="hardware",
                                          path_validator=planning_table_checker.require_safe)
            previous = None
            episode_id = args.session_index * 10 + segment_index

            def record(tick) -> None:
                nonlocal previous
                if bool(tick.state.diagnostics.get("goal_readback_mismatch", False)):
                    first_uncertain = int(tick.state.diagnostics.get("last_matching_goal_readback_tick", -1)) + 1
                    recorder.mark_command_uncertain_since_tick(first_uncertain)
                if previous is not None:
                    actual_dt = (tick.state.timestamp_ns - previous.state.timestamp_ns) * 1e-9
                    table_results = [
                        runtime_table_checker.evaluate(previous.state.q_ctrl),
                        runtime_table_checker.evaluate(previous.command.transmitted_q_ref),
                        runtime_table_checker.evaluate(tick.state.q_ctrl),
                    ]
                    table_result = min(table_results, key=lambda item: item.effective_clearance_m)
                    table_violation = tick.safety_guard_failure or ("table_clearance" if not table_result.safe else "")
                    validity = TransitionValidity(
                        abs(actual_dt - config.control_dt) <= 0.10 * config.control_dt and table_result.safe,
                        previous.command.command_delivery_uncertain,
                        table_violation,
                    )
                    recorder.append(
                        previous.state, previous.command, tick.state,
                        episode_id=episode_id, split_group_id=split_group,
                        motion_mode_id=segment.mode_id, validity=validity, motion_variant=segment.name,
                        table_safety={
                            "predicted_clearance_m": table_result.predicted_clearance_m,
                            "effective_clearance_m": table_result.effective_clearance_m,
                            "nearest_component": table_result.nearest_component,
                            "identity": table_profile.sha256,
                            "violation": table_violation,
                        },
                    )
                if mirror is not None and tick.state.valid:
                    mirror.update(tick.state.q_ctrl)
                previous = tick

            runner = RealTimeRunner(
                backend, lambda tick, state: sequence[min(tick, len(sequence) - 1)],
                mode=RealControlMode.DIRECT, command_envelope="hardware",
                state_validator=runtime_table_checker.require_safe,
                command_validator=runtime_table_checker.require_safe,
            )
            records = runner.run(steps, on_tick=record)
            if len(records) != steps:
                raise RuntimeError(f"{segment.name} stopped early: {len(records)}/{steps} control ticks")
            total_ticks += len(records)
            current_q = records[-1].state.q_ctrl.copy()
        backend.prepare_hardware_motion()
        backend.move_to_configuration(config.home_q_ctrl, args.preposition_seconds, envelope="hardware",
                                      path_validator=planning_table_checker.require_safe)
        completed, status = True, "completed"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        raise
    finally:
        # Always make the recovery decision before torque-off.  It may move
        # only from a freshly verified normal state; an error/interrupt that
        # leaves the state uncertain therefore results in an explicit skip and
        # immediate torque disable in backend.close().
        recovery = guarded_shutdown_recovery()
        if completed and recovery.get("status") != "completed":
            completed = False
            status = "failed_recovery"
            failure = f"guarded shutdown recovery {recovery.get('status')}: {recovery.get('reason', '')}"
        manifest = {
            "schema_version": 4,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "robot_identity": robot.artifact_identity(),
            "plant_identity": collection_plant_identity,
            "collection_mode": "model_a_excitation",
            "collection_plan": model_a_collection_plan_identity(),
            "action_semantics": "software_transmitted_absolute_position_target",
            "session_id": session_id,
            "session_index": args.session_index,
            "split": args.split,
            "split_group_id": split_group,
            "seed": args.seed,
            "total_control_ticks": total_ticks,
            "hardware_config_path": str(Path(args.hardware_config).resolve()),
            "workspace_bounds": {
                "source": "hardware_joint_low_high_minus_margin",
                "margin_deg": args.workspace_margin_deg,
                "low_rad": low.tolist(), "high_rad": high.tolist(),
            },
            "table_safety": table_profile.identity(),
            "shutdown_recovery": recovery,
            "program": [
                {"name": item.name, "mode": item.mode, "mode_id": item.mode_id,
                 "seconds": item.seconds, "single_joint_index": item.single_joint_index}
                for item in program
            ],
        }
        if failure is not None:
            manifest["failure"] = failure
        if recorder.arrays():
            recorder.save(shard, manifest)
        else:
            # A startup/preposition failure produces no transition arrays, but
            # its recovery decision is still audit evidence.  Do not create an
            # empty NPZ, because it can never be appended.
            failure_path = shard.with_suffix(".failure.json")
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if mirror is not None:
            mirror.close()
        backend.close()
    if not completed:
        raise RuntimeError("workspace session did not complete; evidence shard was retained and was not appended")
    validate_dataset(shard)
    canonical_path, manifest_path = append_completed_session(canonical, shard)
    print({
        "status": "completed", "session_shard": str(shard), "canonical_dataset": str(canonical_path),
        "canonical_manifest": str(manifest_path), "samples": len(recorder.arrays()["states"]),
        "split_group_id": split_group,
    })


if __name__ == "__main__":
    main()
