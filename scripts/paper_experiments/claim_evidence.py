"""Diagnostics for the paper's history-alignment and learned-model claims."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
_ROOT = Path(__file__).resolve().parents[2]
_DYNAMICS_ROOT = _ROOT / "dynamics_modeling"
for _path in (_ROOT, _DYNAMICS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from mpc.cost_functions import joint_space_tracking_cost
from mpc.history import history_tokens
from mpc.model_c.branches import execute_open_loop_action_branch, project_activation_q_ref_sequence
from mpc.task_space_cost import ExactTaskSpaceCost, TaskSpaceCostConfig
from neural_dynamics.rollout import rollout_dynamics_batch
from scripts.experiment_utils import load_json
from scripts.paper_experiments.evaluation import write_json
from scripts.paper_experiments.workflow import ROOT, RUNNER


TRAJECTORIES = ("circle", "figure8", "fast_ellipse")
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _args_from_manifest(manifest: dict[str, Any]) -> argparse.Namespace:
    args = RUNNER.parse_args([])
    for key, value in manifest["base_run_args"].items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.multirate_mode = "virtual_asap"
    args.delay_protocol = "full"
    args.anticipation_delay_steps = int(manifest["delay_calibration"]["anticipation_delay_steps"])
    args.controller_mode = "mpc"
    args.device = "cuda"
    return args


def _total_cost(
    states: np.ndarray,
    actions: np.ndarray,
    q_des: np.ndarray,
    dq_des: np.ndarray,
    previous_command: np.ndarray,
    previous_velocity: np.ndarray,
    current_nominal: np.ndarray,
    current_nominal_velocity: np.ndarray,
    cost_config: Any,
    exact_task_cost: ExactTaskSpaceCost,
    task_positions: np.ndarray,
    task_rotations: np.ndarray,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
) -> float:
    device = cost_config.q_tracking_scale.device
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    residual = actions - q_des
    previous_residual = previous_command - current_nominal
    previous_residual_velocity = previous_velocity - current_nominal_velocity
    base = joint_space_tracking_cost(
        tensor(states).unsqueeze(0), tensor(q_des).unsqueeze(0), tensor(dq_des).unsqueeze(0),
        tensor(actions).unsqueeze(0), tensor(previous_command), tensor(previous_velocity),
        tensor(joint_low), tensor(joint_high), cost_config,
        nominal_q_ref=tensor(q_des), requested_residual=tensor(residual).unsqueeze(0),
        previous_residual=tensor(previous_residual),
        previous_residual_velocity=tensor(previous_residual_velocity),
    )[0]
    n_joints = actions.shape[1]
    terms = exact_task_cost.evaluate(
        tensor(states[1:, :n_joints]).unsqueeze(0), task_positions, task_rotations
    )
    config = exact_task_cost.config
    return float((
        base + config.w_position * terms["task_position"][0]
        + config.w_orientation * terms["task_orientation"][0]
    ).detach().cpu())


def _target_steps(reference: dict[str, np.ndarray], count: int = 6) -> list[int]:
    loop = np.flatnonzero(reference["segment_ids"] == 3)
    if len(loop) < 40:
        raise ValueError("reference has no sufficiently long steady task segment")
    if count <= 0:
        raise ValueError("target count must be positive")
    indices = np.linspace(0.10, 0.90, count)
    return [int(loop[round(value * (len(loop) - 1))]) for value in indices]


def collect_candidate_ranking(
    output: Path, manifest_path: Path, resume: bool,
    case_limit: int | None = None, target_count: int = 6,
) -> None:
    manifest = load_json(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    completed_cases = 0
    for trajectory in TRAJECTORIES:
        reference_path = Path(manifest["references"][trajectory]["path"])
        with np.load(reference_path, allow_pickle=False) as archive:
            reference = {key: np.asarray(archive[key]) for key in archive.files if key != "metadata_json"}
        targets = _target_steps(reference, target_count)
        for seed in (0, 1, 2):
            if case_limit is not None and completed_cases >= case_limit:
                break
            completed_cases += 1
            shard = output / "candidate_ranking" / f"{trajectory}_seed{seed}.npz"
            if shard.is_file() and resume:
                with np.load(shard, allow_pickle=False) as archive:
                    for value in archive["rows_json"].tolist():
                        rows.append(json.loads(str(value)))
                continue
            args = _args_from_manifest(manifest)
            args.reference_mode = "task"
            args.reference_file = str(reference_path)
            args.seed = seed
            physical_v = RUNNER._parse_joint_vector(
                args.command_velocity_physical_limit, args.n_joints, "command_velocity_physical_limit"
            )
            physical_a = RUNNER._parse_joint_vector(
                args.command_acceleration_physical_limit, args.n_joints, "command_acceleration_physical_limit"
            )
            model = mujoco.MjModel.from_xml_path(str(RUNNER.resolve_runtime_path(args.model_xml)))
            task_config = TaskSpaceCostConfig(
                w_position=args.w_task_position, w_orientation=args.w_task_orientation,
                position_scale_m=args.task_position_scale_m,
                orientation_scale_rad=args.task_orientation_scale_rad,
                temporal_discount=args.temporal_discount,
            )
            exact_task = ExactTaskSpaceCost(
                model, ee_site_name=args.ee_site_name, n_joints=args.n_joints, config=task_config
            )
            cursor = 0
            shard_rows: list[dict[str, Any]] = []

            def observer(**kwargs: Any) -> None:
                nonlocal cursor
                step = int(kwargs["step"])
                if cursor >= len(targets) or step < targets[cursor]:
                    return
                group_index = cursor
                cursor += 1
                candidates = tuple(kwargs["packet"].branch_candidates)
                if not candidates:
                    return
                before = kwargs["env"].capture_full_state().copy()
                state_history = np.asarray(kwargs["states_history"], dtype=np.float32)
                command_history = np.asarray(kwargs["command_history"], dtype=np.float32)
                bundle = kwargs["dynamics_bundle"]
                initial_history = torch.as_tensor(
                    history_tokens(state_history, command_history, bundle.history_len),
                    dtype=torch.float32, device=bundle.device,
                )
                q_des = np.asarray(kwargs["q_des_sequence"], dtype=np.float32)
                dq_des = np.asarray(kwargs["dq_des_sequence"], dtype=np.float32)
                positions = reference["task_positions_des"][step + 1 : step + 1 + args.horizon]
                rotations = reference["task_rotations_des"][step + 1 : step + 1 + args.horizon]
                for candidate in candidates:
                    planned = np.asarray(candidate.q_ref_sequence, dtype=np.float32)
                    executed = project_activation_q_ref_sequence(
                        planned, kwargs["previous_command"], kwargs["previous_velocity"],
                        kwargs["env"].joint_low, kwargs["env"].joint_high, args.joint_limit_margin,
                        physical_v, physical_a, kwargs["env"].control_dt,
                    )
                    with torch.no_grad():
                        predicted = rollout_dynamics_batch(
                            model=bundle.model, normalizer=bundle.normalizer,
                            model_type=bundle.model_type, initial_history=initial_history,
                            future_q_ref=torch.as_tensor(
                                executed, dtype=torch.float32, device=bundle.device
                            ).unsqueeze(0),
                            state_dim=bundle.state_dim, target_mode=bundle.target_mode,
                            control_dt=bundle.control_dt,
                        )[0].detach().cpu().numpy().astype(np.float32)
                    predicted_cost = _total_cost(
                        predicted, executed, q_des, dq_des,
                        kwargs["previous_command"], kwargs["previous_velocity"],
                        reference["q_des"][step], reference["dq_des"][step],
                        kwargs["cost_config"], exact_task, positions, rotations,
                        kwargs["env"].joint_low, kwargs["env"].joint_high,
                    )
                    branch = execute_open_loop_action_branch(
                        parent_env=kwargs["env"], full_state=before,
                        planned_q_ref_sequence=planned, executed_q_ref_sequence=executed,
                        predicted_state_sequence=predicted, predicted_cost=predicted_cost,
                        role_mask=tuple(candidate.role_mask), actual_state=kwargs["state"],
                        previous_command=kwargs["previous_command"],
                        previous_velocity=kwargs["previous_velocity"],
                    )
                    realized_cost = _total_cost(
                        branch.realized_state_sequence, executed, q_des, dq_des,
                        kwargs["previous_command"], kwargs["previous_velocity"],
                        reference["q_des"][step], reference["dq_des"][step],
                        kwargs["cost_config"], exact_task, positions, rotations,
                        kwargs["env"].joint_low, kwargs["env"].joint_high,
                    )
                    q_error = np.sqrt(np.mean(
                        (predicted[:, :args.n_joints] - branch.realized_state_sequence[:, :args.n_joints]) ** 2,
                        axis=1,
                    ))
                    shard_rows.append({
                        "trajectory": trajectory, "seed": seed, "snapshot": group_index,
                        "activation_step": step, "roles": list(candidate.role_mask),
                        "predicted_cost": predicted_cost, "realized_cost": realized_cost,
                        "relative_regret_placeholder": 0.0,
                        "projection_active": bool(np.max(np.abs(planned - executed)) > 1e-6),
                        "projection_rms_rad": float(np.sqrt(np.mean((planned - executed) ** 2))),
                        **{
                            f"q_rmse_k{k}_rad": float(q_error[k])
                            for k in (1, 5, 10, 20)
                        },
                    })
                after = kwargs["env"].capture_full_state()
                if not np.array_equal(before, after):
                    raise RuntimeError("counterfactual branch mutated the parent MuJoCo environment")

            RUNNER.run_closed_loop_mpc(deepcopy(args), activation_observer=observer)
            if cursor != len(targets):
                raise RuntimeError(f"{trajectory} seed {seed}: collected {cursor}/{len(targets)} snapshots")
            shard.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                shard, rows_json=np.asarray([json.dumps(row, sort_keys=True) for row in shard_rows])
            )
            rows.extend(shard_rows)
    _summarize_candidate_rows(rows, output)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def _summarize_candidate_rows(rows: list[dict[str, Any]], output: Path) -> None:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["trajectory"], int(row["seed"]), int(row["snapshot"])), []).append(row)
    group_rows: list[dict[str, Any]] = []
    for key, candidates in groups.items():
        pred = np.asarray([row["predicted_cost"] for row in candidates], dtype=float)
        real = np.asarray([row["realized_cost"] for row in candidates], dtype=float)
        corr = float(np.corrcoef(_rank(pred), _rank(real))[0, 1]) if len(candidates) >= 3 else float("nan")
        selected = next((row for row in candidates if "selected" in row["roles"]), None)
        baseline = next((row for row in candidates if "baseline" in row["roles"]), None)
        alternative = next((row for row in candidates if "alternative_elite" in row["roles"]), None)
        best_real = float(np.min(real))
        group_rows.append({
            "trajectory": key[0], "seed": key[1], "snapshot": key[2],
            "candidate_count": len(candidates), "within_snapshot_spearman": corr,
            "all_pair_concordance": float(np.mean([
                (pred[i] <= pred[j]) == (real[i] <= real[j])
                for i in range(len(pred)) for j in range(i + 1, len(pred))
            ])),
            "selected_vs_baseline_correct": float(
                selected is not None and baseline is not None and baseline is not selected
                and ((selected["predicted_cost"] <= baseline["predicted_cost"])
                     == (selected["realized_cost"] <= baseline["realized_cost"]))
            ) if baseline is not None and baseline is not selected else float("nan"),
            "selected_vs_alternative_correct": float(
                selected is not None and alternative is not None and alternative is not selected
                and ((selected["predicted_cost"] <= alternative["predicted_cost"])
                     == (selected["realized_cost"] <= alternative["realized_cost"]))
            ) if alternative is not None and alternative is not selected else float("nan"),
            "selected_baseline_duplicate": bool(baseline is selected and selected is not None),
            "selected_alternative_duplicate": bool(alternative is selected and selected is not None),
            "selected_additive_regret": float(selected["realized_cost"] - best_real) if selected else float("nan"),
            "selected_relative_regret": float(
                (selected["realized_cost"] - best_real) / max(abs(best_real), 1e-9)
            ) if selected else float("nan"),
            "projection_active": bool(any(row["projection_active"] for row in candidates)),
            **{
                f"q_rmse_k{k}_rad": float(np.mean([row[f"q_rmse_k{k}_rad"] for row in candidates]))
                for k in (1, 5, 10, 20)
            },
        })
    summary: dict[str, Any] = {
        "snapshot_count": len(group_rows), "candidate_count": len(rows),
        "selected_baseline_duplicate_count": int(sum(row["selected_baseline_duplicate"] for row in group_rows)),
        "selected_alternative_duplicate_count": int(sum(row["selected_alternative_duplicate"] for row in group_rows)),
        "definitions": {
            "spearman": "Spearman computed within each activation snapshot, then summarized.",
            "primary_branch": "activation-projected; predicted and realized rollouts consume identical commands.",
        },
    }
    for field in (
        "within_snapshot_spearman", "all_pair_concordance",
        "selected_vs_baseline_correct", "selected_vs_alternative_correct",
        "selected_additive_regret", "selected_relative_regret",
        "q_rmse_k1_rad", "q_rmse_k5_rad", "q_rmse_k10_rad", "q_rmse_k20_rad",
    ):
        values = np.asarray([row[field] for row in group_rows], dtype=float)
        values = values[np.isfinite(values)]
        summary[field] = {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
            "n": int(len(values)),
        }
        rng = np.random.default_rng(20260780)
        boot: list[float] = []
        trajectories = sorted({str(row["trajectory"]) for row in group_rows})
        for _ in range(10_000):
            sampled: list[float] = []
            for trajectory in rng.choice(trajectories, size=len(trajectories), replace=True):
                trajectory_rows = [row for row in group_rows if row["trajectory"] == trajectory]
                seeds = sorted({int(row["seed"]) for row in trajectory_rows})
                for seed in rng.choice(seeds, size=len(seeds), replace=True):
                    seed_rows = [
                        row for row in trajectory_rows
                        if int(row["seed"]) == int(seed) and np.isfinite(float(row[field]))
                    ]
                    if seed_rows:
                        picked = rng.choice(len(seed_rows), size=len(seed_rows), replace=True)
                        sampled.extend(float(seed_rows[index][field]) for index in picked)
            if sampled:
                boot.append(float(np.mean(sampled)))
        summary[field]["hierarchical_bootstrap_95ci"] = [
            float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        ] if boot else [float("nan"), float("nan")]
    for active in (False, True):
        subset = [row for row in group_rows if bool(row["projection_active"]) == active]
        summary[f"projection_{'active' if active else 'inactive'}"] = {
            "snapshot_count": len(subset),
            "spearman_mean": float(np.nanmean([row["within_snapshot_spearman"] for row in subset]))
            if subset else float("nan"),
            "pair_concordance_mean": float(np.nanmean([row["all_pair_concordance"] for row in subset]))
            if subset else float("nan"),
        }
    destination = output / "summaries"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "candidate_ranking_groups.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    write_json(destination / "candidate_ranking.json", summary)
    figure, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    labels = ("Spearman", "Pairwise", "Sel.–zero", "Sel.–elite")
    fields = (
        "within_snapshot_spearman", "all_pair_concordance",
        "selected_vs_baseline_correct", "selected_vs_alternative_correct",
    )
    axes[0].bar(
        np.arange(len(fields)),
        [summary[field]["mean"] for field in fields],
        color=("#1d4ed8", "#0f766e", "#b45309", "#7c3aed"),
    )
    axes[0].set_xticks(np.arange(len(fields)), labels, rotation=25, ha="right")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("Ranking agreement")
    horizons = (1, 5, 10, 20)
    axes[1].plot(
        horizons,
        [1000.0 * summary[f"q_rmse_k{k}_rad"]["mean"] for k in horizons],
        marker="o", color="#0f766e",
    )
    axes[1].set_xlabel("Replay horizon [steps]")
    axes[1].set_ylabel("Joint prediction RMSE [mrad]")
    axes[1].set_xticks(horizons)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination / "candidate_ranking.pdf", bbox_inches="tight")
    plt.close(figure)


def _percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _effort_row(
    robot: str, entry: dict[str, Any], xml_path: Path,
) -> dict[str, Any]:
    with np.load(Path(entry["run_dir"]) / "rollout.npz", allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    tau = np.asarray(arrays["tau_actuator"], dtype=np.float64)
    acceleration = np.asarray(arrays["command_acceleration"], dtype=np.float64)
    states = np.asarray(arrays["actual_states"], dtype=np.float64)
    n_joints = tau.shape[1]
    dq = states[: len(tau), n_joints : 2 * n_joints]
    jerk = np.diff(acceleration, axis=0) / 0.01
    slew = np.diff(tau, axis=0) / 0.01
    absolute_power = np.sum(np.abs(tau * dq), axis=1)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    ranges = np.max(np.abs(np.asarray(model.actuator_forcerange[:n_joints], dtype=np.float64)), axis=1)
    limited = np.asarray(model.actuator_forcelimited[:n_joints], dtype=bool)
    ranges = np.where(limited & (ranges > 0.0), ranges, np.nan)
    utilization = np.abs(tau) / ranges[None, :]
    tcp_error = np.asarray(arrays.get("ee_position_errors", np.empty(0)), dtype=np.float64)
    return {
        "robot": robot, "label": entry["label"], "trajectory": entry["trajectory"],
        "seed": int(entry["seed"]), "run_dir": entry["run_dir"],
        "tcp_rmse_mm": float(1000.0 * np.sqrt(np.mean(tcp_error ** 2))) if tcp_error.size else float("nan"),
        "command_acceleration_rms_rad_s2": float(np.sqrt(np.mean(acceleration ** 2))),
        "command_acceleration_p95_rad_s2": _percentile(np.abs(acceleration), 95),
        "command_acceleration_max_rad_s2": _percentile(np.abs(acceleration), 100),
        "command_jerk_rms_rad_s3": float(np.sqrt(np.mean(jerk ** 2))),
        "command_jerk_p95_rad_s3": _percentile(np.abs(jerk), 95),
        "command_jerk_max_rad_s3": _percentile(np.abs(jerk), 100),
        "torque_rms_nm": float(np.sqrt(np.mean(tau ** 2))),
        "torque_abs_p95_nm": _percentile(np.abs(tau), 95),
        "torque_abs_max_nm": _percentile(np.abs(tau), 100),
        "torque_slew_rms_nm_s": float(np.sqrt(np.mean(slew ** 2))),
        "torque_slew_p95_nm_s": _percentile(np.abs(slew), 95),
        "torque_slew_max_nm_s": _percentile(np.abs(slew), 100),
        "torque_utilization_p95": _percentile(utilization, 95),
        "torque_utilization_max": _percentile(utilization, 100),
        "absolute_mechanical_power_mean_w": float(np.mean(absolute_power)),
        "absolute_mechanical_power_p95_w": _percentile(absolute_power, 95),
        "absolute_mechanical_power_max_w": _percentile(absolute_power, 100),
    }


def effort_postprocess(
    output: Path, abb_index: Path, ur5e_index: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    configurations = (
        (
            "ABB", abb_index, ROOT / "dynamics_modeling" / "ABB_IRB2400.xml",
            {"DirectIK", "FullVirtual", "ThreadedASAP"},
        ),
        (
            "UR5e", ur5e_index,
            ROOT / "dynamics_modeling" / "robots" / "ur5e" / "ur5e_project.xml",
            {"ProjectedDirectIK", "FullVirtual", "ThreadedAsync"},
        ),
    )
    for robot, index_path, xml_path, labels in configurations:
        entries = load_json(index_path)["entries"]
        for entry in entries:
            if entry["label"] in labels:
                rows.append(_effort_row(robot, entry, xml_path))
    destination = output / "summaries"
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "effort_peak_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = [key for key in rows[0] if key not in {"robot", "label", "trajectory", "seed", "run_dir"}]
    aggregate: list[dict[str, Any]] = []
    for robot, label in sorted({(str(row["robot"]), str(row["label"])) for row in rows}):
        subset = [row for row in rows if row["robot"] == robot and row["label"] == label]
        aggregate.append({
            "robot": robot, "label": label, "n": len(subset),
            **{
                metric: float(np.nanmean([float(row[metric]) for row in subset]))
                for metric in metrics
            },
        })
    with (destination / "effort_peak_aggregate.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)


def pareto_figures(output: Path, pareto_index: Path, abb_main_index: Path) -> None:
    abb_xml = ROOT / "dynamics_modeling" / "ABB_IRB2400.xml"
    entries = load_json(pareto_index)["entries"]
    rows = [_effort_row("ABB", entry, abb_xml) | {"effort_scale": float(entry["effort_scale"])} for entry in entries]
    direct_rows = [
        _effort_row("ABB", entry, abb_xml)
        for entry in load_json(abb_main_index)["entries"]
        if entry["label"] == "DirectIK" and entry["trajectory"] in TRAJECTORIES
    ]
    destination = output / "summaries"
    destination.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict[str, Any]] = []
    for scale in SCALES:
        subset = [row for row in rows if math.isclose(row["effort_scale"], scale)]
        aggregate_row: dict[str, Any] = {
            "effort_scale": scale, "n": len(subset),
            **{
                metric: float(np.mean([row[metric] for row in subset]))
                for metric in (
                    "tcp_rmse_mm", "command_acceleration_rms_rad_s2", "torque_rms_nm",
                    "torque_abs_p95_nm", "torque_abs_max_nm",
                    "absolute_mechanical_power_p95_w",
                )
            },
        }
        rng = np.random.default_rng(20260790)
        for metric in ("tcp_rmse_mm", "command_acceleration_rms_rad_s2", "torque_rms_nm"):
            bootstrap: list[float] = []
            for _ in range(10_000):
                sampled: list[float] = []
                for trajectory in rng.choice(TRAJECTORIES, size=len(TRAJECTORIES), replace=True):
                    trajectory_rows = [row for row in subset if row["trajectory"] == trajectory]
                    picked = rng.choice(len(trajectory_rows), size=len(trajectory_rows), replace=True)
                    sampled.extend(float(trajectory_rows[index][metric]) for index in picked)
                bootstrap.append(float(np.mean(sampled)))
            aggregate_row[f"{metric}_ci95_low"] = float(np.percentile(bootstrap, 2.5))
            aggregate_row[f"{metric}_ci95_high"] = float(np.percentile(bootstrap, 97.5))
        aggregate.append(aggregate_row)
    with (destination / "effort_pareto_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (destination / "effort_pareto_aggregate.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    baseline = {
        metric: float(np.mean([row[metric] for row in direct_rows]))
        for metric in ("tcp_rmse_mm", "command_acceleration_rms_rad_s2", "torque_rms_nm")
    }
    write_json(destination / "effort_pareto_projected_ik_baseline.json", {
        "trajectories": list(TRAJECTORIES), "n": len(direct_rows), **baseline,
    })
    for x_metric, x_label, filename in (
        ("command_acceleration_rms_rad_s2", r"Command acceleration RMS [rad/s$^2$]", "pareto_command_acceleration.pdf"),
        ("torque_rms_nm", "Actuator torque RMS [Nm]", "pareto_torque.pdf"),
    ):
        figure, axis = plt.subplots(figsize=(4.8, 3.3))
        x = [row[x_metric] for row in aggregate]
        y = [row["tcp_rmse_mm"] for row in aggregate]
        axis.plot(x, y, marker="o", color="#0f766e", linewidth=1.6)
        axis.errorbar(
            x, y,
            yerr=[
                [row["tcp_rmse_mm"] - row["tcp_rmse_mm_ci95_low"] for row in aggregate],
                [row["tcp_rmse_mm_ci95_high"] - row["tcp_rmse_mm"] for row in aggregate],
            ],
            fmt="none", ecolor="#0f766e", alpha=0.45, capsize=2,
        )
        for row in aggregate:
            axis.annotate(f"{row['effort_scale']:g}×", (row[x_metric], row["tcp_rmse_mm"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        axis.scatter([baseline[x_metric]], [baseline["tcp_rmse_mm"]], marker="s", color="#b45309", label="Projected IK")
        axis.set_xlabel(x_label)
        axis.set_ylabel("TCP RMSE [mm]")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
        figure.tight_layout()
        figure.savefig(destination / filename, bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/paper_claim_evidence_v1")
    sub = parser.add_subparsers(dest="command", required=True)
    candidate = sub.add_parser("candidate-ranking")
    candidate.add_argument("--manifest", required=True)
    candidate.add_argument("--resume", action="store_true")
    candidate.add_argument("--case-limit", type=int, default=None)
    candidate.add_argument("--target-count", type=int, default=6)
    effort = sub.add_parser("effort-postprocess")
    effort.add_argument("--abb-index", required=True)
    effort.add_argument("--ur5e-index", required=True)
    pareto = sub.add_parser("pareto")
    pareto.add_argument("--pareto-index", required=True)
    pareto.add_argument("--abb-main-index", required=True)
    args = parser.parse_args()
    output = Path(args.output_root)
    if not output.is_absolute():
        output = ROOT / output
    if args.command == "candidate-ranking":
        collect_candidate_ranking(
            output, Path(args.manifest), args.resume, args.case_limit, args.target_count
        )
    elif args.command == "effort-postprocess":
        effort_postprocess(output, Path(args.abb_index), Path(args.ur5e_index))
    elif args.command == "pareto":
        pareto_figures(output, Path(args.pareto_index), Path(args.abb_main_index))


if __name__ == "__main__":
    main()
