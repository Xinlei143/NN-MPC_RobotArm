"""Deterministic virtual-time runner for delayed multi-rate CEM-MPC.

It receives the host script namespace to avoid a circular import with the CLI
entry point.  The virtual schedule is deliberate: CEM is measured normally but
its result only becomes eligible after ``anticipation_delay_steps`` control
ticks, so controller behaviour is reproducible without Python-thread jitter.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from neural_dynamics.rollout import rollout_dynamics_batch
from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.cost_functions import JointSpaceCostConfig
from mpc.delay_aware import DelayedPlanPacket, corrected_direct_ik_command, feedback_correction
from mpc.planner_rollout import LearnedDynamicsPlanner, PlannerRolloutConfig


def run(args: Any, api: dict[str, Any]) -> dict[str, Any]:
    if args.controller_mode != "mpc" or args.mpc_policy != "residual":
        raise ValueError("virtual delay-aware modes require --controller_mode mpc --mpc_policy residual")
    if args.visualize:
        raise ValueError("--visualize is not supported by virtual delay-aware modes")
    delay = args.anticipation_delay_steps or args.replan_interval_steps
    if delay <= 0:
        raise ValueError("anticipation_delay_steps must be positive")
    device = api["resolve_device"](args.device)
    api["set_seed"](args.seed)
    bundle = api["load_dynamics_bundle"](
        checkpoint_path=api["resolve_runtime_path"](args.checkpoint), normalizer_path=api["resolve_runtime_path"](args.normalizer),
        model_type=args.model_type, n_joints=args.n_joints, device=device, history_len=args.history_len,
    )
    env = api["MuJoCoArmEnv"](str(api["resolve_runtime_path"](args.model_xml)), n_joints=args.n_joints, seed=args.seed)
    stack = api["_stack_records"]
    try:
        state = env.reset_to_configuration(api["MPC_HOME_Q"][: args.n_joints])
        previous_command = np.asarray(state[: args.n_joints], dtype=np.float32).copy()
        previous_velocity = np.zeros(args.n_joints, dtype=np.float32)
        for _ in range(args.settle_steps):
            state = env.step(previous_command)
        states_history, command_history = [state.copy()], [previous_command.copy()]
        reference, dq_reference, ddq_reference, execution_steps, task_reference = api["_reference_for_run"](
            args=args, state=state, env=env, control_dt=bundle.control_dt
        )
        if args.max_execution_steps is not None:
            execution_steps = min(execution_steps, args.max_execution_steps)
        execution_steps = min(execution_steps, reference.shape[0] - args.horizon - delay - 1)
        if execution_steps <= 0:
            raise ValueError("reference is too short for horizon plus anticipation delay")
        parse = api["_parse_joint_vector"]
        physical_v = parse(args.command_velocity_physical_limit, args.n_joints, "command_velocity_physical_limit")
        physical_a = parse(args.command_acceleration_physical_limit, args.n_joints, "command_acceleration_physical_limit")
        residual_max = parse(args.residual_max, args.n_joints, "residual_max")
        feedback_max = parse(args.feedback_max, args.n_joints, "feedback_max")
        calibration = api["_reference_calibration"](reference, dq_reference, ddq_reference, physical_v, physical_a)
        t = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
        cost = JointSpaceCostConfig(
            cost_mode="residual", w_q=args.w_q, w_dq=args.w_dq, w_residual=args.w_residual, w_servo=args.w_servo,
            w_residual_velocity=args.w_residual_velocity, w_residual_acceleration=args.w_residual_acceleration,
            w_first=args.w_first, w_qref_velocity=args.w_qref_velocity, w_qref_acceleration=args.w_qref_acceleration,
            w_terminal=args.w_terminal, w_joint_limit=args.w_joint_limit, w_dq_limit=args.w_dq_limit,
            q_tracking_scale=t(calibration["q_tracking_scale"]), dq_tracking_scale=t(calibration["dq_tracking_scale"]),
            residual_scale=t(0.5 * residual_max), servo_scale=t(parse(args.servo_scale, args.n_joints, "servo_scale")),
            residual_velocity_scale=t(residual_max / bundle.control_dt), residual_acceleration_scale=t(residual_max / bundle.control_dt**2),
            qref_velocity_scale=t(physical_v), qref_acceleration_scale=t(physical_a), temporal_discount=args.temporal_discount,
            barrier_max_weight=args.barrier_max_weight, state_velocity_limit=t(parse(args.state_velocity_limit, args.n_joints, "state_velocity_limit")),
            joint_limit_safe_margin=args.joint_limit_safe_margin, joint_limit_temp=args.joint_limit_temp,
            dq_limit_temp=args.dq_limit_temp, control_dt=bundle.control_dt, velocity_cost_mode=args.velocity_cost_mode,
        )
        rollout = PlannerRolloutConfig(
            mpc_policy="residual", q_ref_velocity_limit=t(physical_v), q_ref_acceleration_limit=t(physical_a),
            residual_max=t(residual_max), joint_limit_margin=args.joint_limit_margin,
            rollout_batch_size=args.rollout_batch_size, project_residual_kinematics=False,
        )
        joint_low, joint_high = t(env.joint_low), t(env.joint_high)
        controller: CEMMPCController | None = None
        active: DelayedPlanPacket | None = None
        pending: dict[int, DelayedPlanPacket] = {}
        rec: dict[str, list[Any]] = {key: [] for key in (
            "actual_states next_states q_des dq_des actuator_q_ref delta_q_ref command_velocity command_acceleration planning_time replan_time mpc_replanned replan_deadline_miss control_step_wall_time buffer_index buffer_length best_cost mean_cost baseline_cost selected_cost elite_mean_cost selection_mode failure_flags joint_limit_violation_flags command_velocity_violation_flags command_acceleration_violation_flags realized_tracking_error nominal_q_ref buffered_residual executed_residual feedback_correction predicted_feedback_state packet_age packet_event tau_actuator tau_gravity tau_total desired_ee_positions desired_ee_rotations actual_ee_positions actual_ee_rotations ee_position_errors ee_orientation_errors segment_ids lap_ids".split()
        )}
        rows: list[dict[str, Any]] = []

        def active_action(absolute_step: int) -> np.ndarray:
            nominal = reference[absolute_step + 1]
            if args.multirate_mode == "virtual_asap" and active is not None:
                index = active.index_at(absolute_step)
                if index is not None:
                    return nominal + active.residual_sequence[index]
            return nominal

        def prediction_context(step: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
            history = api["build_history_tensor"](states_history, command_history, bundle.history_len, device)
            actions = np.stack([active_action(step + i) for i in range(delay)]).astype(np.float32)
            predicted = rollout_dynamics_batch(
                model=bundle.model, normalizer=bundle.normalizer, model_type=bundle.model_type, initial_history=history,
                future_q_ref=t(actions).unsqueeze(0), state_dim=bundle.state_dim, target_mode=bundle.target_mode,
                control_dt=bundle.control_dt,
            )[0].detach().cpu().numpy().astype(np.float32)
            tokens = [np.concatenate([s, q]).astype(np.float32) for s, q in zip(states_history, command_history)]
            tokens.extend(np.concatenate([predicted[i + 1], actions[i]]).astype(np.float32) for i in range(delay))
            future_history = np.stack(tokens[-bundle.history_len :])
            while future_history.shape[0] < bundle.history_len:
                future_history = np.concatenate([future_history[:1], future_history], axis=0)
            velocity = previous_velocity if delay == 1 else (actions[-1] - actions[-2]) / bundle.control_dt
            return t(future_history), predicted[-1], actions[-1], velocity.astype(np.float32)

        for step in range(execution_steps):
            started = api["time"].perf_counter()
            event = ""
            if step in pending:
                active = pending.pop(step); event = "packet_activated"
            nominal = np.asarray(reference[step + 1], dtype=np.float32)
            age = -1
            plan_residual = np.zeros(args.n_joints, dtype=np.float32)
            predicted_feedback = np.full(2 * args.n_joints, np.nan, dtype=np.float32)
            if active is not None:
                age = active.index_at(step)
                if age is None:
                    active = None; age = -1
                else:
                    plan_residual = active.residual_sequence[age].copy()
                    predicted_feedback = active.predicted_state_sequence[age].copy()
            feedback = np.zeros(args.n_joints, dtype=np.float32) if age < 0 else feedback_correction(
                predicted_feedback, state, args.feedback_kq, args.feedback_kdq, feedback_max
            )
            proposed = np.clip(plan_residual + feedback, -residual_max - feedback_max, residual_max + feedback_max)
            command_t, correction_t = corrected_direct_ik_command(
                t(nominal), t(proposed), t(previous_command), t(previous_velocity), joint_low, joint_high,
                args.joint_limit_margin, t(physical_v), t(physical_a), bundle.control_dt,
            )
            command, executed = command_t.detach().cpu().numpy().astype(np.float32), correction_t.detach().cpu().numpy().astype(np.float32)
            planning_time = float("nan"); replanned = 0; failure = 0
            best = mean = baseline = selected = elite = float("nan")
            selection = "direct_ik_nominal" if age < 0 else "delayed_packet_feedback"
            if step % args.replan_interval_steps == 0 and step + delay + args.horizon < reference.shape[0]:
                future_history, anchor_state, anchor_command, anchor_velocity = prediction_context(step)
                anchor = step + delay
                future_q = t(reference[anchor + 1 : anchor + 1 + args.horizon])
                planner = LearnedDynamicsPlanner(
                    model=bundle.model, normalizer=bundle.normalizer, model_type=bundle.model_type, state_dim=bundle.state_dim,
                    target_mode=bundle.target_mode, control_dt=bundle.control_dt, initial_history=future_history, q_des=future_q,
                    dq_des=t(dq_reference[anchor + 1 : anchor + 1 + args.horizon]), nominal_q_ref=future_q,
                    previous_q_ref=t(anchor_command), previous_q_ref_velocity=t(anchor_velocity),
                    previous_residual=torch.zeros(args.n_joints, dtype=torch.float32, device=device),
                    previous_residual_velocity=torch.zeros(args.n_joints, dtype=torch.float32, device=device),
                    joint_low=joint_low, joint_high=joint_high, cost_config=cost, rollout_config=rollout,
                )
                if controller is None:
                    controller = CEMMPCController(CEMMPCConfig(
                        horizon=args.horizon, action_dim=args.n_joints, num_samples=args.num_samples, num_elites=args.num_elites,
                        elite_ratio=args.elite_ratio, cem_iters=args.cem_iters, init_std=args.init_std, min_std=args.min_std,
                        smoothing_alpha=args.smoothing_alpha, temporal_noise_alpha=args.temporal_noise_alpha,
                        reset_std_each_step=args.reset_std_each_step, uniform_sample_ratio=args.uniform_sample_ratio,
                        force_baseline_candidate=True, execute=args.cem_execute, seed=args.seed, device=str(device),
                    ), planner, env.joint_low, env.joint_high)
                    # CUDA's first rollout is a runtime initialisation artefact,
                    # not a representative asynchronous-plan delay.
                    if args.mpc_warmup_plans:
                        generator_state = controller.generator.get_state()
                        for _ in range(args.mpc_warmup_plans):
                            controller.plan(anchor_state, anchor_command)
                        controller.generator.set_state(generator_state)
                        controller.reset()
                else:
                    controller.planner = planner
                result = controller.plan(anchor_state, anchor_command)
                planning_time, replanned, failure = float(result.planning_time), 1, int(result.failure)
                best, mean, baseline, selected, elite, selection = result.best_cost, result.mean_cost, result.baseline_cost, result.selected_cost, result.elite_mean_cost, result.selection_mode
                if result.failure:
                    event = "planner_failure"
                elif planning_time > delay * bundle.control_dt:
                    event = "late_plan_dropped"
                else:
                    pending[anchor] = DelayedPlanPacket(step, anchor, result.selected_residual_sequence.copy(), result.selected_predicted_state_sequence.copy(), planning_time, args.multirate_mode)
                    event = (event + ";" if event else "") + "packet_scheduled"
            delta = command - previous_command
            velocity, acceleration = delta / bundle.control_dt, (delta / bundle.control_dt - previous_velocity) / bundle.control_dt
            torque = env.compute_torque_components(command)
            rec["actual_states"].append(state.copy()); rec["q_des"].append(reference[step].copy()); rec["dq_des"].append(dq_reference[step].copy())
            if task_reference is not None:
                dp, dr = np.asarray(task_reference.task_positions_des[step], dtype=np.float32), np.asarray(task_reference.task_rotations_des[step], dtype=np.float32)
                ap, ar = api["site_pose"](env.model, env.data, args.ee_site_name); ap, ar = np.asarray(ap, dtype=np.float32), np.asarray(ar, dtype=np.float32)
                rec["desired_ee_positions"].append(dp); rec["desired_ee_rotations"].append(dr); rec["actual_ee_positions"].append(ap); rec["actual_ee_rotations"].append(ar)
                rec["ee_position_errors"].append(float(np.linalg.norm(ap - dp))); rec["ee_orientation_errors"].append(api["_orientation_error"](dr, ar)); rec["segment_ids"].append(int(task_reference.segment_ids[step])); rec["lap_ids"].append(int(task_reference.lap_ids[step]))
            state = env.step(command)
            rec["next_states"].append(state.copy()); rec["actuator_q_ref"].append(command); rec["delta_q_ref"].append(delta); rec["command_velocity"].append(velocity); rec["command_acceleration"].append(acceleration)
            rec["planning_time"].append(0.0 if not np.isfinite(planning_time) else planning_time); rec["replan_time"].append(planning_time); rec["mpc_replanned"].append(replanned); rec["replan_deadline_miss"].append(int(np.isfinite(planning_time) and planning_time > args.replan_interval_steps * bundle.control_dt)); rec["control_step_wall_time"].append(api["time"].perf_counter() - started)
            rec["buffer_index"].append(age); rec["buffer_length"].append(args.horizon if active is not None else 0); rec["best_cost"].append(best); rec["mean_cost"].append(mean); rec["baseline_cost"].append(baseline); rec["selected_cost"].append(selected); rec["elite_mean_cost"].append(elite); rec["selection_mode"].append(selection); rec["failure_flags"].append(failure); rec["joint_limit_violation_flags"].append(0); rec["command_velocity_violation_flags"].append(int(np.any(np.abs(velocity) > physical_v + 1e-6))); rec["command_acceleration_violation_flags"].append(int(np.any(np.abs(acceleration) > physical_a + 1e-6)))
            rec["nominal_q_ref"].append(nominal); rec["buffered_residual"].append(plan_residual); rec["executed_residual"].append(executed); rec["feedback_correction"].append(feedback); rec["predicted_feedback_state"].append(predicted_feedback); rec["packet_age"].append(age); rec["packet_event"].append(event)
            for source, target in (("actuator_tau", "tau_actuator"), ("gravity_tau", "tau_gravity"), ("total_tau", "tau_total")):
                rec[target].append(torque[source].astype(np.float32))
            tracking = float(np.linalg.norm(state[: args.n_joints] - reference[step + 1])); rec["realized_tracking_error"].append(tracking)
            rows.append({"step": step, "controller_mode": "mpc", "tracking_error": tracking, "planning_time": planning_time, "replan_time": planning_time, "mpc_replanned": replanned, "replan_deadline_miss": rec["replan_deadline_miss"][-1], "multirate_mode": args.multirate_mode, "packet_event": event, "packet_age": age, "feedback_correction_norm": float(np.linalg.norm(feedback)), "executed_residual_norm": float(np.linalg.norm(executed)), "selection_mode": selection})
            previous_command, previous_velocity = command.copy(), velocity.astype(np.float32)
            states_history.append(state.copy()); command_history.append(command.copy())
    finally:
        env.close()
    int_keys = {"mpc_replanned", "replan_deadline_miss", "buffer_index", "buffer_length", "failure_flags", "joint_limit_violation_flags", "command_velocity_violation_flags", "command_acceleration_violation_flags", "packet_age", "segment_ids", "lap_ids"}
    string_keys = {"selection_mode", "packet_event"}
    arrays = {
        key: stack(value, dtype=(str if key in string_keys else np.int64 if key in int_keys else np.float32))
        for key, value in rec.items()
    }
    arrays.update({
        "controller_mode": np.asarray("mpc"), "mpc_policy": np.asarray("residual"), "cost_profile": np.asarray(args.cost_profile),
        "replan_interval_steps": np.asarray(args.replan_interval_steps, dtype=np.int64), "replan_deadline_s": np.asarray(args.replan_interval_steps * bundle.control_dt, dtype=np.float32),
        "multirate_mode": np.asarray(args.multirate_mode), "anticipation_delay_steps": np.asarray(delay, dtype=np.int64), "feedback_kq": np.asarray(args.feedback_kq, dtype=np.float32), "feedback_kdq": np.asarray(args.feedback_kdq, dtype=np.float32), "feedback_max": feedback_max, "residual_max": residual_max, "q_ref_velocity_limit": physical_v, "q_ref_acceleration_limit": physical_a,
        "recovery_active_flags": np.zeros(len(rec["actuator_q_ref"]), dtype=np.int64), "recovery_trigger_reasons": np.asarray([""] * len(rec["actuator_q_ref"])),
        "cem_reset_std_each_step": np.asarray(args.reset_std_each_step), "cem_uniform_sample_ratio": np.asarray(args.uniform_sample_ratio, dtype=np.float32), "cem_uniform_sample_count": np.asarray(int(round((args.num_samples - 2) * args.uniform_sample_ratio)), dtype=np.int64),
        "ddq_des": stack([ddq_reference[i] for i in range(len(rec["q_des"]))]),
    })
    if task_reference is not None:
        arrays["execution_steps"] = np.asarray(execution_steps, dtype=np.int64)
    return {"arrays": arrays, "rows": rows, "failure_reasons": []}
