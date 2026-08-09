#!/usr/bin/env python3
"""Standalone CUDA benchmark for the real SO101 executable CEM path.

This intentionally does not connect to hardware.  It includes the canonical
runtime command spec so the measured planner path is the same projection used
by real ASAP-MPC.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Running a file from ``scripts/`` puts that directory first on sys.path;
# explicitly put the repository root first so ``scripts.run_cem_mpc`` refers
# to this checkout rather than an unrelated ROS package named ``scripts``.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dynamics_modeling") not in sys.path:
    sys.path.insert(0, str(ROOT / "dynamics_modeling"))

import scripts.run_cem_mpc as cli
from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.cost_functions import JointSpaceCostConfig
from mpc.executable_rollout import ExecutableRolloutEngine
from mpc.planner_rollout import LearnedDynamicsPlanner, PlannerRolloutConfig
from mpc.utils import build_history_tensor
from neural_dynamics.rollout import load_dynamics_bundle
from robot_runtime.config import load_hardware_config
from robot_runtime.executable_command import (
    ExecutableCommandState,
    make_executable_command_spec,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--robot-config", default="configs/robots/so101.yaml")
    parser.add_argument("--history-len", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--cem-iters", type=int, default=2)
    parser.add_argument("--plans", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--executable-rollout-backend", choices=["auto", "cuda_graph", "eager"], default="auto")
    parser.add_argument("--selection-validation", choices=["none", "exact_final_pool"], default="exact_final_pool")
    parser.add_argument("--delay-steps", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise SystemExit("CUDA is required for this benchmark")

    n = 5
    device = torch.device(args.device)
    robot_args = cli.build_arg_parser().parse_args(
        ["--robot_config", args.robot_config, "--n_joints", str(n)]
    )
    robot = cli._resolve_robot_from_args(robot_args)
    bundle = load_dynamics_bundle(
        cli.resolve_runtime_path(args.checkpoint),
        cli.resolve_runtime_path(args.normalizer),
        "gru",
        n,
        device,
        history_len=args.history_len,
        expected_robot_spec=robot,
    )
    ref_args = cli.build_arg_parser().parse_args(
        [
            "--n_joints", str(n), "--reference_mode", "joint_file",
            "--reference_file", args.reference, "--horizon", str(args.horizon),
            "--multirate_mode", "synchronous",
        ]
    )
    q, dq, ddq, _ = cli._load_joint_file_reference(ref_args)
    horizon = args.horizon
    hardware = load_hardware_config(args.hardware_config)
    calibration = json.loads(Path(hardware.calibration_path).read_text(encoding="utf-8"))
    names = hardware.joint_names
    cal_low = np.asarray([calibration[name]["range_min"] for name in names], dtype=np.float64)
    cal_high = np.asarray([calibration[name]["range_max"] for name in names], dtype=np.float64)
    spec = make_executable_command_spec(
        joint_low=hardware.hardware_joint_low,
        joint_high=hardware.hardware_joint_high,
        velocity_limit=hardware.command_velocity_limit,
        acceleration_limit=hardware.command_acceleration_limit,
        relative_limit=hardware.hardware_joint_high - hardware.hardware_joint_low,
        raw_low=hardware.raw_low[:n], raw_high=hardware.raw_high[:n],
        calibration_low=cal_low, calibration_high=cal_high,
        control_dt=hardware.control_dt,
    )
    scales = cli._reference_calibration(
        q, dq, ddq, hardware.command_velocity_limit.astype(np.float32),
        hardware.command_acceleration_limit.astype(np.float32),
    )
    residual_max = np.full(n, np.deg2rad(2.0), dtype=np.float32)
    qv = scales["q_ref_velocity_limit"]
    qa = scales["q_ref_acceleration_limit"]
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    cost = JointSpaceCostConfig(
        cost_mode="residual", q_tracking_scale=tensor(scales["q_tracking_scale"]),
        dq_tracking_scale=tensor(scales["dq_tracking_scale"]),
        residual_scale=tensor(0.5 * residual_max), servo_scale=torch.ones(n, device=device),
        residual_velocity_scale=tensor(residual_max / bundle.control_dt),
        residual_acceleration_scale=tensor(residual_max / bundle.control_dt**2),
        qref_velocity_scale=tensor(qv), qref_acceleration_scale=tensor(qa),
        state_velocity_limit=tensor(hardware.measured_velocity_emergency),
        control_dt=bundle.control_dt,
    )
    rollout = PlannerRolloutConfig(
        mpc_policy="residual", q_ref_velocity_limit=tensor(qv),
        q_ref_acceleration_limit=tensor(qa), residual_max=tensor(residual_max),
        rollout_batch_size=args.num_samples, project_residual_kinematics=False,
        residual_parameterization="full",
    )
    executable_engine = ExecutableRolloutEngine(
        model=bundle.model, normalizer=bundle.normalizer, model_type=bundle.model_type,
        state_dim=bundle.state_dim, target_mode=bundle.target_mode,
        control_dt=bundle.control_dt, spec=spec, backend=args.executable_rollout_backend,
    )
    state = np.concatenate([q[0], dq[0]]).astype(np.float32)
    history = build_history_tensor([state] * args.history_len, [q[0]] * args.history_len, args.history_len, device)
    planner = LearnedDynamicsPlanner(
        model=bundle.model, normalizer=bundle.normalizer, model_type="gru",
        state_dim=bundle.state_dim, target_mode=bundle.target_mode, control_dt=bundle.control_dt,
        initial_history=history, q_des=tensor(q[1 : 1 + horizon]),
        dq_des=tensor(dq[1 : 1 + horizon]), nominal_q_ref=tensor(q[:horizon]),
        previous_q_ref=tensor(q[0]), previous_q_ref_velocity=torch.zeros(n, device=device),
        previous_residual=torch.zeros(n, device=device), previous_residual_velocity=torch.zeros(n, device=device),
        joint_low=tensor(hardware.hardware_joint_low), joint_high=tensor(hardware.hardware_joint_high),
        cost_config=cost, rollout_config=rollout, executable_command_spec=spec,
        executable_command_state=ExecutableCommandState.anchored(q[0]),
        executable_rollout_engine=executable_engine,
    )
    delay = max(0, int(args.delay_steps))
    forecast_history = history.unsqueeze(0)
    forecast_requested = torch.zeros((1, delay, n), dtype=torch.float32, device=device)
    forecast_previous = torch.as_tensor(q[0], dtype=torch.float32, device=device).view(1, -1)
    forecast_velocity = torch.zeros_like(forecast_previous)
    forecast_expected = torch.zeros((1, delay, n), dtype=torch.int64, device=device)
    forecast_mask = torch.zeros((1, delay), dtype=torch.bool, device=device)
    forecast_times = []
    if delay > 0:
        for _ in range(args.warmup):
            executable_engine.run(
                initial_history=forecast_history,
                requested_q_ref=forecast_requested,
                previous_q_ref=forecast_previous,
                previous_velocity=forecast_velocity,
                fallback_q_ref=forecast_requested,
                expected_raw=forecast_expected,
                expected_raw_mask=forecast_mask,
                fail_closed=True,
                exact=True,
            )
        torch.cuda.synchronize()
        for _ in range(args.plans):
            start = time.perf_counter()
            executable_engine.run(
                initial_history=forecast_history,
                requested_q_ref=forecast_requested,
                previous_q_ref=forecast_previous,
                previous_velocity=forecast_velocity,
                fallback_q_ref=forecast_requested,
                expected_raw=forecast_expected,
                expected_raw_mask=forecast_mask,
                fail_closed=True,
                exact=True,
            )
            torch.cuda.synchronize()
            forecast_times.append((time.perf_counter() - start) * 1000.0)
    controller = CEMMPCController(
        CEMMPCConfig(
            horizon=horizon, action_dim=n, decision_horizon=horizon,
            num_samples=args.num_samples, cem_iters=args.cem_iters,
            force_baseline_candidate=True, execute="lowest_cost", seed=args.seed,
            device=str(device), selection_validation=args.selection_validation,
        ),
        planner, hardware.hardware_joint_low, hardware.hardware_joint_high,
    )
    for _ in range(args.warmup):
        controller.plan(state, q[0])
    torch.cuda.synchronize()
    samples = []
    end_to_end_samples = []
    for _ in range(args.plans):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = controller.plan(state, q[0])
        torch.cuda.synchronize()
        cem_done = time.perf_counter()
        # The real worker now carries exact raw counts from the final pool.
        # Keep the compatibility replay only for planners without that field.
        if result.selected_expected_raw_sequence.shape != result.selected_q_ref_sequence.shape:
            requested = planner.nominal_sequence() + torch.as_tensor(
                result.selected_residual_sequence, dtype=torch.float32, device=device
            )
            planner.exact_executable_command_sequence(
                requested.unsqueeze(0),
                torch.as_tensor(result.selected_predicted_state_sequence, dtype=torch.float32, device=device).unsqueeze(0),
            )
        torch.cuda.synchronize()
        samples.append((cem_done - start) * 1000.0)
        end_to_end_samples.append((time.perf_counter() - start) * 1000.0)
    print(json.dumps({
        "benchmark": "canonical_executable_cem_fast_search",
        "checkpoint": args.checkpoint, "horizon": horizon,
        "num_samples": args.num_samples, "cem_iters": args.cem_iters,
        "plans": len(samples), "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
        "max_ms": float(np.max(samples)),
        "cem_plus_exact_mean_ms": float(np.mean(end_to_end_samples)),
        "cem_plus_exact_p95_ms": float(np.percentile(end_to_end_samples, 95)),
        "cem_plus_exact_max_ms": float(np.max(end_to_end_samples)),
        "exact_float64_fallbacks": int(planner._exact_projection_fallbacks),
        "delay_steps": delay,
        "delay_forecast_mean_ms": float(np.mean(forecast_times)) if forecast_times else 0.0,
        "delay_forecast_p95_ms": float(np.percentile(forecast_times, 95)) if forecast_times else 0.0,
        "reported_last_planning_ms": float(result.planning_time * 1000.0),
    }, indent=2))


if __name__ == "__main__":
    main()
