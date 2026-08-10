#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import scripts.run_cem_mpc as simulation_cli
from mpc.asap_planner_worker import ASAPPlannerWorker
from mpc.asap_shared import LatestSnapshotStore, PlanPacketStore, PlannerResultStore
from mpc.logging import save_mpc_run
from robot_runtime.asap_adapter import ASAPStorePlannerAdapter
from robot_runtime.artifacts import verify_real_artifact_identity
from robot_runtime.config import load_hardware_config
from robot_runtime.factory import make_so101_backend
from robot_runtime.ood import RobustEnvelope
from robot_runtime.runner import RealControlMode, RealTimeRunner
from scripts.run_real_direct_control import JointFilePlayer, find_reference_manifest_entry


def parser() -> argparse.ArgumentParser:
    value = simulation_cli.build_arg_parser()
    value.description = "Run joint-space SO101 shadow or low-authority active ASAP-MPC."
    value.add_argument("--hardware-config", required=True)
    value.add_argument("--real-mode", choices=["shadow_mpc", "active_mpc"], default="shadow_mpc")
    value.add_argument("--enable-motion", action="store_true")
    value.add_argument("--operator-supported-shutdown", action="store_true")
    value.add_argument(
        "--home-tolerance-deg",
        default=1.0,
        type=float,
        help="Home convergence tolerance; must remain below the configured +/-3 deg experiment envelope.",
    )
    value.add_argument("--delay-calibration", default=None)
    value.add_argument("--ood-envelope", default=None)
    value.add_argument("--reference-manifest", required=True,
                       help="mpc_manifest.json whose artifacts.q_des_ctrl.npy SHA-256 must match the "
                            "reference file; also gates playback through JointFilePlayer (hardware "
                            "envelope / home start / per-sample step) before torque is enabled.")
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.enable_motion: raise SystemExit("refusing hardware connection without --enable-motion")
    if not args.operator_supported_shutdown: raise SystemExit("refusing torque enable without --operator-supported-shutdown")
    if args.home_tolerance_deg <= 0.0 or args.home_tolerance_deg >= 3.0:
        raise SystemExit("--home-tolerance-deg must be > 0 and < 3 degrees")
    if args.reference_mode != "joint_file": raise SystemExit("real MPC currently requires a prevalidated --reference_mode joint_file")
    if not args.checkpoint or not args.normalizer: raise SystemExit("real MPC requires --checkpoint and --normalizer")
    args.multirate_mode = "threaded_asap"
    args.controller_mode = "mpc"
    # The real path only ever plays a joint-space reference, so the sim
    # parser's task-space final-pool cost (default "on") has no targets to
    # score and would abort worker init.  The sim CLI rejects this combination
    # in its argument validation; the real CLI mirrors that here by forcing
    # the joint-only cost (paper workflow "joint_only_two_stage" variant).
    args.exact_task_space_cost = "off"
    args.n_joints = 5
    args.robot_config = "configs/robots/so101.yaml"
    robot = simulation_cli._resolve_robot_from_args(args)
    hardware = load_hardware_config(args.hardware_config)
    if not np.allclose(robot.home_q, hardware.home_q_ctrl, atol=1e-6):
        raise SystemExit("configs/robots/so101.yaml home_q must match hardware home_q_ctrl before real MPC")
    if args.real_mode == "active_mpc" and hardware.voltage_hard_low is None:
        raise SystemExit("active MPC requires frozen voltage warning/hard thresholds in hardware config")
    if args.real_mode == "active_mpc":
        if args.delay_calibration is None:
            raise SystemExit("active MPC requires --delay-calibration from a shadow run")
        delay = json.loads(Path(args.delay_calibration).read_text(encoding="utf-8"))
        if (int(delay.get("samples", 0)) < 2000 or delay.get("method") != "p99.5" or
                float(delay.get("late_drop_rate", 1.0)) >= .01 or float(delay.get("packet_expiry_rate", 1.0)) >= .01):
            raise SystemExit("active MPC delay gate requires >=2000 samples and late-drop/expiry rates below 1%")
        args.anticipation_delay_steps = int(delay["anticipation_delay_steps"])
    verify_real_artifact_identity(args.checkpoint, args.normalizer, hardware.plant_identity())
    # Use the same residual bound in the live runner that the planner uses
    # during candidate generation.  This must not be hard-coded: otherwise a
    # nominal 0.5/1.0 degree authority sweep would still execute with the
    # historical 2 degree runtime clamp.
    runtime_residual_max = simulation_cli._parse_joint_vector(
        args.residual_max, args.n_joints, "residual_max"
    )
    assert runtime_residual_max is not None
    print("active MPC residual cap [deg]: "
          + " ".join(f"{value:.4f}" for value in np.rad2deg(runtime_residual_max)))
    reference, dq_reference, ddq_reference, execution_steps = simulation_cli._load_joint_file_reference(args)
    required_reference = execution_steps + args.horizon + args.anticipation_delay_steps + int(args.mpc_preview_nominal_steps) + 1
    if reference.shape[0] < required_reference:
        raise SystemExit(f"joint reference needs at least {required_reference} rows for horizon+delay padding")
    # Freeze the reference identity before any torque: SHA-256 must match the
    # derived MPC manifest, and the JointFilePlayer gates (hardware envelope /
    # home start / per-sample step) must pass on the full padded playback file.
    # mpc_reference_prevalidated=True is only ever set after both succeed.
    shape_name, entry, sha = find_reference_manifest_entry(args.reference_manifest, args.reference_file)
    print(f"reference matched frozen MPC artifact [{shape_name}] sha256={sha[:16]}...")
    player = JointFilePlayer(reference, config=hardware)
    # The padded beginning of the formal reference is deliberately static.
    # Keep active MPC disabled until the reference has moved by at least 1e-3
    # rad from its first pose; otherwise CEM can inject arbitrary residuals
    # into the home hold and build up projector velocity before the task starts.
    reference_origin = np.asarray(reference[0], dtype=np.float32)
    motion_rows = np.flatnonzero(np.max(np.abs(reference - reference_origin), axis=1) > 1e-3)
    active_start_tick = int(motion_rows[0]) if motion_rows.size else 0
    if args.real_mode == "active_mpc":
        print(f"active MPC startup gate: direct IK until tick {active_start_tick}")
    print(f"joint reference: {reference.shape[0]} rows @ {hardware.control_dt:.4f}s = "
          f"{reference.shape[0] * hardware.control_dt:.2f}s (envelope-validated)")
    snapshots, packets, results = LatestSnapshotStore(), PlanPacketStore(), PlannerResultStore()
    stop = threading.Event()
    envelope = None
    if args.ood_envelope is not None:
        ood = json.loads(Path(args.ood_envelope).read_text(encoding="utf-8"))
        envelope = RobustEnvelope(np.asarray(ood["median"], dtype=np.float32),
                                  np.asarray(ood["scale"], dtype=np.float32), float(ood["threshold"]))
    if args.real_mode == "active_mpc":
        if envelope is None:
            raise SystemExit("active MPC requires --ood-envelope calibrated from Model-A and shadow data")
        if any(float(ood.get(key, 0.0)) < .99 for key in
               ("executed_history_coverage", "selected_action_coverage", "predicted_state_coverage")):
            raise SystemExit("active MPC requires >=99% executed/action/predicted-state OOD coverage")
    backend = make_so101_backend(args.hardware_config)
    adapter = ASAPStorePlannerAdapter(snapshots, packets, 5, ood_envelope=envelope,
                                      control_dt=hardware.control_dt)
    worker = None
    # Nominal playback = the manifest-matched, gate-validated joint reference;
    # holds the final pose after the last row (JointFilePlayer semantics).
    nominal = player
    try:
        backend.connect()
        backend.startup_to_home(home_tolerance_rad=float(np.deg2rad(args.home_tolerance_deg)))
        # Freeze the exact runtime envelope/calibration/state machine after
        # startup.  The CUDA planner must use this object, not the dynamics
        # RobotSpec or a second hand-written projector.
        executable_command_spec = backend.executable_command_spec("hardware")
        # The worker consumes a warmup snapshot before it can publish.  It must
        # use the measured, frozen-home hardware state—not simulation's zero
        # state—so its history semantics match the live control thread.
        startup = backend.read_state(tick_index=0)
        adapter.submit(
            0, startup.timestamp_ns, startup.vector[None, :],
            startup.q_ctrl[None, :], startup.history_generation,
            executable_command_state=backend.executable_command_state(),
        )
        worker = ASAPPlannerWorker(args, vars(simulation_cli), snapshots, packets, results, stop,
                                   executable_command_spec.joint_low,
                                   executable_command_spec.joint_high,
                                   reference, dq_reference, ddq_reference,
                                   executable_command_spec=executable_command_spec)
        worker.start()
        if not worker.ready.wait(60) or worker.status().failure_reason:
            stop.set(); snapshots.wake()
            raise RuntimeError(worker.status().failure_reason or "planner worker initialization timeout")
        runner = RealTimeRunner(backend, nominal, mode=RealControlMode(args.real_mode), planner=adapter,
                                history_len=worker.history_len or 8, residual_limit_rad=runtime_residual_max,
                                # The reference spans the data-collection envelope (e.g. the 4 cm
                                # circle needs pan +/-6.5 deg), not the first-motion +/-3 deg
                                # authority; JointFilePlayer validated it above, so hardware
                                # envelope playback keeps commands inside the measured distribution.
                                command_envelope="hardware", mpc_reference_prevalidated=True,
                                active_start_tick=active_start_tick)
        records = runner.run(min(execution_steps, args.max_execution_steps or execution_steps))
        packet_available = np.asarray([record.planner_packet_available for record in records], dtype=bool)
        ever_packet = np.maximum.accumulate(packet_available) if packet_available.size else packet_available
        event_rows = results.drain()
        # Per-tick planner duration for the shared diagnostic plot.  Events
        # are published about once per control tick and drained in
        # publication order, so event i aligns with tick i; ticks beyond the
        # event count are left at zero (no plan was produced there).
        planning_times = np.zeros(len(records), dtype=np.float32)
        for index, event in enumerate(event_rows[:len(records)]):
            planning_times[index] = float(event.planning_time_s)
        def event_phase(name: str) -> np.ndarray:
            return np.asarray(
                [float(event.candidate_diagnostics.get(name, np.nan)) for event in event_rows],
                dtype=np.float32,
            )
        # OOD tokens for calibrate_real_ood.py: executed = per-tick
        # [state, previous q_ref] (15-dim); selected_action/predicted_state =
        # flattened [predicted_state; q_ref] future windows the runtime
        # OOD-checks per activated packet ((6,15) at H=6 -> M*6 rows).
        executed_tokens = np.asarray(adapter.executed_tokens, dtype=np.float32) if adapter.executed_tokens else np.empty((0, 10 + 5), dtype=np.float32)
        future_rows = [row for window in adapter.future_tokens for row in window]
        future_tokens = np.asarray(future_rows, dtype=np.float32) if future_rows else np.empty((0, 10 + 5), dtype=np.float32)
        command_velocity = np.asarray([
            record.command.diagnostics.get("command_velocity", np.zeros(hardware.n_joints, dtype=np.float32))
            for record in records
        ], dtype=np.float32)
        command_acceleration = np.zeros_like(command_velocity)
        if len(records) > 1:
            command_acceleration[1:] = np.diff(command_velocity, axis=0) / float(hardware.control_dt)
        # A one-count encoder quantisation step can exceed the numerical
        # acceleration threshold at 30 Hz.  Keep that diagnostic separate from
        # actual runtime safety faults; the transmitted command itself is already
        # checked by the canonical projector.
        quantization_acceleration_exceedance = np.asarray([
            bool(np.any(np.abs(value) > np.asarray(hardware.command_acceleration_limit) + 1e-7))
            for value in command_acceleration
        ], dtype=bool)
        actual_residual = np.asarray([
            record.command.transmitted_q_ref - record.nominal for record in records
        ], dtype=np.float32)
        arrays = {
            "actual_states": np.asarray([record.state.vector for record in records], dtype=np.float32),
            "observed_states": np.asarray([record.state.vector for record in records], dtype=np.float32),
            "state_timestamp_ns": np.asarray([record.state.timestamp_ns for record in records], dtype=np.int64),
            "q_des": np.asarray([record.nominal for record in records], dtype=np.float32),
            "actuator_q_ref": np.asarray([record.command.transmitted_q_ref for record in records], dtype=np.float32),
            "requested_absolute_command": np.asarray([record.command.requested_q_ref for record in records], dtype=np.float32),
            "projected_absolute_command": np.asarray([record.command.projected_q_ref for record in records], dtype=np.float32),
            "transmitted_goal_position_raw": np.asarray([record.command.tx_goal_position_raw for record in records], dtype=np.int64),
            "planner_expected_raw": np.asarray([
                record.command.diagnostics.get("planner_expected_raw", np.full(5, -1, dtype=np.int64))
                for record in records
            ], dtype=np.int64),
            "planner_expected_raw_match": np.asarray([
                bool(record.command.diagnostics.get("planner_expected_raw_match", True))
                for record in records
            ], dtype=bool),
            "planner_raw_mismatch_direct_fallback": np.asarray([
                bool(record.command.diagnostics.get("planner_raw_mismatch_direct_fallback", False))
                for record in records
            ], dtype=bool),
            "planner_raw_mismatch_live_reproject": np.asarray([
                bool(record.command.diagnostics.get("planner_raw_mismatch_live_reproject", False))
                for record in records
            ], dtype=bool),
            "fallback_q_ref": np.asarray([
                record.command.diagnostics.get("fallback_q_ref", np.full(5, np.nan, dtype=np.float32))
                for record in records
            ], dtype=np.float32),
            "command_velocity": command_velocity,
            "executable_command_velocity": command_velocity.copy(),
            "command_acceleration": command_acceleration,
            "planner_requested_residual": np.asarray([record.planner_residual for record in records], dtype=np.float32),
            "requested_mpc_residual": np.asarray([record.planner_residual for record in records], dtype=np.float32),
            # This is the residual that was actually transmitted relative to
            # the tick's nominal reference.  It includes the final canonical
            # projector and encoder quantisation; it is intentionally kept
            # separate from the planner's requested residual.
            "executed_residual": actual_residual,
            "command_nominal_offset": actual_residual.copy(),
            "safety_projection_offset": np.asarray([
                record.command.transmitted_q_ref - record.command.projected_q_ref for record in records
            ], dtype=np.float32),
            "projection_discrepancy": np.asarray([
                record.command.transmitted_q_ref - record.command.requested_q_ref for record in records
            ], dtype=np.float32),
            "requested_correction": np.asarray([
                record.command.requested_q_ref - record.nominal for record in records
            ], dtype=np.float32),
            "residual_max": np.asarray(runtime_residual_max, dtype=np.float32),
            "residual_saturated": np.asarray([
                bool(np.any(np.abs(record.planner_residual) >= 0.999 * runtime_residual_max))
                for record in records
            ], dtype=bool),
            "cem_num_samples": np.asarray(args.num_samples, dtype=np.int64),
            "cem_iters": np.asarray(args.cem_iters, dtype=np.int64),
            "cem_horizon": np.asarray(args.horizon, dtype=np.int64),
            "cem_seed": np.asarray(args.seed, dtype=np.int64),
            "cem_uniform_sample_ratio": np.asarray(args.uniform_sample_ratio, dtype=np.float32),
            "cem_reset_std_each_step": np.asarray(args.reset_std_each_step),
            "mpc_policy": np.asarray(args.mpc_policy),
            "cost_profile": np.asarray(args.cost_profile),
            "planner_projection": np.asarray(args.planner_projection),
            "nominal_command_semantics": np.asarray(args.nominal_command_semantics),
            "packet_residual_semantics": np.asarray("planner_residual_added_to_projected_nominal"),
            "residual_feasibility_semantics": np.asarray("runtime_jointwise_hard_clip"),
            "tx_local_success": np.asarray([record.command.tx_local_success for record in records], dtype=bool),
            "command_delivery_uncertain": np.asarray([
                record.command.command_delivery_uncertain for record in records
            ], dtype=bool),
            "projection_flags": np.asarray([
                "|".join(record.command.projection_flags) for record in records
            ], dtype=str),
            "state_validity_flags": np.asarray([
                "|".join(record.state.validity_flags) for record in records
            ], dtype=str),
            "safety_mode": np.asarray([record.safety_mode.value for record in records], dtype=str),
            "safety_guard_failure": np.asarray([
                record.safety_guard_failure or "" for record in records
            ], dtype=str),
            "planner_applied": np.asarray([record.planner_applied for record in records], dtype=bool),
            "command_velocity_violation_flags": np.asarray([
                bool(np.any(np.abs(value) > np.asarray(hardware.command_velocity_limit) + 1e-7))
                for value in command_velocity
            ], dtype=bool),
            "command_acceleration_violation_flags": np.zeros(len(records), dtype=bool),
            "command_acceleration_quantization_exceedance_flags": quantization_acceleration_exceedance,
            "command_acceleration_flag_semantics": np.asarray(
                "runtime_fault_only; quantization_exceedance_separate"
            ),
            "motor_voltage_v": np.asarray([
                record.state.diagnostics.get("motor_voltage_v", np.full(6, np.nan)) for record in records
            ]),
            "motor_temperature_c": np.asarray([
                record.state.diagnostics.get("motor_temperature", np.full(6, np.nan)) for record in records
            ]),
            "motor_current_raw": np.asarray([
                record.state.diagnostics.get("motor_current_raw", np.full(6, np.nan)) for record in records
            ]),
            "motor_load_raw": np.asarray([
                record.state.diagnostics.get("motor_load_raw", np.full(6, np.nan)) for record in records
            ]),
            "diagnostic_sample_age_s": np.asarray([
                record.state.diagnostics.get("diagnostic_sample_age_s", np.nan) for record in records
            ]),
            "control_wakeup_lateness_s": np.asarray([record.wake_lateness_s for record in records]),
            "control_deadline_miss": np.asarray([record.skipped_ticks > 0 for record in records]),
            "packet_expired": ever_packet & ~packet_available,
            "planner_end_to_end_latency_s": np.asarray([event.end_to_end_latency_s for event in event_rows]),
            "planner_worker_queue_wait_ms": event_phase("worker_queue_wait_ms"),
            "planner_anchor_forecast_ms": event_phase("anchor_forecast_ms"),
            "planner_cem_search_wall_ms": event_phase("cem_search_wall_ms"),
            "planner_packet_postprocess_ms": event_phase("packet_postprocess_ms"),
            "planner_late_drop": np.asarray([event.result_type == "success_late_dropped" for event in event_rows]),
            # Scalar planner counters are consumed by mpc.logging's threaded
            # run summary.  Keep them explicit; otherwise a real ASAP run is
            # incorrectly reported as having zero late drops.
            "planner_solve_count": np.asarray(len(event_rows), dtype=np.int64),
            "planner_late_drop_count": np.asarray(sum(event.result_type == "success_late_dropped" for event in event_rows), dtype=np.int64),
            "planner_failure_count": np.asarray(sum(event.result_type == "failure" for event in event_rows), dtype=np.int64),
            "packet_expiration_count": np.asarray(int(np.sum(ever_packet & ~packet_available)), dtype=np.int64),
            "planner_ood_valid": np.asarray([record.planner_ood_valid for record in records]),
            "planning_time": planning_times,
            "executed_tokens": executed_tokens,
            # Both future arrays are the same [predicted_state; q_ref] window
            # that the runtime OOD gate checks -- one distribution, two names.
            "selected_action_tokens": future_tokens,
            "predicted_state_tokens": future_tokens.copy(),
            "controller_mode": np.asarray(args.real_mode),
            "multirate_mode": np.asarray("threaded_asap"),
            "action_semantics": np.asarray("software_transmitted_absolute_position_target"),
            "motor_acknowledged": np.asarray(False),
            "plant_identity_sha256": np.asarray(hardware.config_sha256),
            "tau_actuator_available": np.asarray(False),
            "true_state_available": np.asarray(False),
            "active_start_tick": np.asarray(active_start_tick, dtype=np.int64),
            "execution_steps": np.asarray(len(records), dtype=np.int64),
            "control_dt_s": np.asarray(hardware.control_dt, dtype=np.float32),
        }
        rows = [{"tick": index, "safety_mode": record.safety_mode.value,
                 "planner_applied": record.planner_applied,
                 "planner_expected_raw_match": bool(record.command.diagnostics.get("planner_expected_raw_match", True)),
                 "planner_raw_mismatch_live_reproject": bool(record.command.diagnostics.get("planner_raw_mismatch_live_reproject", False)),
                 "planner_raw_mismatch_direct_fallback": bool(record.command.diagnostics.get("planner_raw_mismatch_direct_fallback", False)),
                 "tx_local_success": record.command.tx_local_success,
                 "command_delivery_uncertain": record.command.command_delivery_uncertain,
                 "projection_flags": "|".join(record.command.projection_flags)}
                for index, record in enumerate(records)]
        planner_events = [asdict(event) for event in event_rows]
        backend.move_to_configuration(robot.home_q, 3.0)
        save_mpc_run(simulation_cli.resolve_runtime_path(args.save_dir), arrays, rows, planner_events)
    finally:
        stop.set(); snapshots.wake()
        if worker is not None:
            worker.join(timeout=10)
        backend.close()


if __name__ == "__main__": main()
