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


def parser() -> argparse.ArgumentParser:
    value = simulation_cli.build_arg_parser()
    value.description = "Run joint-space SO101 shadow or low-authority active ASAP-MPC."
    value.add_argument("--hardware-config", required=True)
    value.add_argument("--real-mode", choices=["shadow_mpc", "active_mpc"], default="shadow_mpc")
    value.add_argument("--enable-motion", action="store_true")
    value.add_argument("--operator-supported-shutdown", action="store_true")
    value.add_argument("--delay-calibration", default=None)
    value.add_argument("--ood-envelope", default=None)
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.enable_motion: raise SystemExit("refusing hardware connection without --enable-motion")
    if not args.operator_supported_shutdown: raise SystemExit("refusing torque enable without --operator-supported-shutdown")
    if args.reference_mode != "joint_file": raise SystemExit("real MPC currently requires a prevalidated --reference_mode joint_file")
    if not args.checkpoint or not args.normalizer: raise SystemExit("real MPC requires --checkpoint and --normalizer")
    args.multirate_mode = "threaded_asap"
    args.controller_mode = "mpc"
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
    reference, dq_reference, ddq_reference, execution_steps = simulation_cli._load_joint_file_reference(args)
    required_reference = execution_steps + args.horizon + args.anticipation_delay_steps + int(args.mpc_preview_nominal_steps) + 1
    if reference.shape[0] < required_reference:
        raise SystemExit(f"joint reference needs at least {required_reference} rows for horizon+delay padding")
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
    adapter = ASAPStorePlannerAdapter(snapshots, packets, 5, ood_envelope=envelope)
    worker = None
    def nominal(tick, state):
        return reference[min(tick, reference.shape[0] - 1)]
    try:
        backend.connect()
        backend.startup_to_home()
        # The worker consumes a warmup snapshot before it can publish.  It must
        # use the measured, frozen-home hardware state—not simulation's zero
        # state—so its history semantics match the live control thread.
        startup = backend.read_state(tick_index=0)
        adapter.submit(0, startup.timestamp_ns, startup.vector[None, :], robot.home_q[None, :],
                       startup.history_generation)
        worker = ASAPPlannerWorker(args, vars(simulation_cli), snapshots, packets, results, stop,
                                   robot.data_target_low, robot.data_target_high,
                                   reference, dq_reference, ddq_reference)
        worker.start()
        if not worker.ready.wait(60) or worker.status().failure_reason:
            stop.set(); snapshots.wake()
            raise RuntimeError(worker.status().failure_reason or "planner worker initialization timeout")
        runner = RealTimeRunner(backend, nominal, mode=RealControlMode(args.real_mode), planner=adapter,
                                history_len=worker.history_len or 8, residual_limit_rad=np.deg2rad(2.0))
        records = runner.run(min(execution_steps, args.max_execution_steps or execution_steps))
        packet_available = np.asarray([record.planner_packet_available for record in records], dtype=bool)
        ever_packet = np.maximum.accumulate(packet_available) if packet_available.size else packet_available
        event_rows = results.drain()
        arrays = {
            "actual_states": np.asarray([record.state.vector for record in records], dtype=np.float32),
            "observed_states": np.asarray([record.state.vector for record in records], dtype=np.float32),
            "q_des": np.asarray([record.nominal for record in records], dtype=np.float32),
            "actuator_q_ref": np.asarray([record.command.transmitted_q_ref for record in records], dtype=np.float32),
            "requested_absolute_command": np.asarray([record.command.requested_q_ref for record in records], dtype=np.float32),
            "planner_requested_residual": np.asarray([record.planner_residual for record in records], dtype=np.float32),
            "control_wakeup_lateness_s": np.asarray([record.wake_lateness_s for record in records]),
            "control_deadline_miss": np.asarray([record.skipped_ticks > 0 for record in records]),
            "packet_expired": ever_packet & ~packet_available,
            "planner_end_to_end_latency_s": np.asarray([event.end_to_end_latency_s for event in event_rows]),
            "planner_late_drop": np.asarray([event.result_type == "success_late_dropped" for event in event_rows]),
            "planner_ood_valid": np.asarray([record.planner_ood_valid for record in records]),
            "controller_mode": np.asarray(args.real_mode),
            "multirate_mode": np.asarray("threaded_asap"),
            "action_semantics": np.asarray("software_transmitted_absolute_position_target"),
            "motor_acknowledged": np.asarray(False),
            "plant_identity_sha256": np.asarray(hardware.config_sha256),
            "tau_actuator_available": np.asarray(False),
            "true_state_available": np.asarray(False),
        }
        rows = [{"tick": index, "safety_mode": record.safety_mode.value,
                 "planner_applied": record.planner_applied,
                 "tx_local_success": record.command.tx_local_success,
                 "command_delivery_uncertain": record.command.command_delivery_uncertain}
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
