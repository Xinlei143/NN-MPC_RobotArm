"""One reproducible entry point for the ROBIO delay-aware MPC experiments."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np
import torch

from mpc.logging import save_mpc_run
from mpc.reference_pipeline import ReferenceConfig, build_reference, save_reference_bundle
from mpc.robot_config import load_robot_spec
from scripts.experiment_utils import (
    environment_snapshot, file_identity, load_completed_rollout, load_json,
    paired_bootstrap_rows, run_fingerprint, write_immutable_json,
)
from scripts.paper_experiments.evaluation import aggregate_rows, latency_recovery, summarize_arrays, write_csv, write_json
from scripts.paper_experiments.reanalyze_gru_history_windows import parse_horizons, reanalyze
from scripts.paper_experiments.statistics import delay_sweep_report, main_endpoint, projection_report
from scripts.robustness._runtime import ROOT, load_runner


RUNNER = load_runner("paper_delay_aware_runner")
DEFAULT_ROOT = ROOT / "outputs" / "paper_delay_aware_two_stage_v2"
LEGACY_PAPER_ROOT = ROOT / "outputs" / "paper_delay_aware_two_stage_v1"
CHECKPOINT = ROOT / "dynamics_modeling" / "outputs" / "checkpoints" / "gru_20260717_182930" / "best_model.pt"
NORMALIZER = ROOT / "dynamics_modeling" / "outputs" / "checkpoints" / "gru_20260717_182930" / "normalizer.pt"
MODEL_XML = ROOT / "dynamics_modeling" / "ABB_IRB2400.xml"
TRAINING_DATASET = ROOT / "dynamics_modeling" / "outputs" / "datasets" / "irb2400_parallel_data copy.npz"
LEGACY_MPC_ROOT = ROOT / "outputs" / "robustness" / "paper_three_ik_l036_s5_v1" / "mpc_architectures"
LEGACY_IK_ROOT = ROOT / "outputs" / "robustness" / "paper_three_ik_l036_s5_v1"
TRAJECTORIES = ("circle", "figure8", "fast_ellipse", "rounded_square")
# This post-freeze diagnostic is evaluated on exactly the same nominal
# trajectory--seed grid as the primary FullVirtual experiment.  The absolute
# 1x point is therefore a directly matched sensitivity reference, rather than
# a three-trajectory exploratory subset.
EFFORT_PARETO_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    top.add_argument("--output-root", default=str(DEFAULT_ROOT))
    sub = top.add_subparsers(dest="command", required=True)

    calibration = sub.add_parser("generate-calibration-reference")
    calibration.add_argument("--steps", type=int, default=2200)
    calibration.add_argument("--overwrite", action="store_true")

    delay = sub.add_parser("calibrate-delay")
    delay.add_argument("--samples", type=int, default=500)
    delay.add_argument("--provisional-delay", type=int, default=10)
    delay.add_argument("--guard-ms", type=float, default=5.0)
    delay.add_argument("--exact-task-space-cost", choices=["on", "off"], default="on")
    delay.add_argument(
        "--variant",
        choices=(
            "task_space_two_stage",
            "joint_only_two_stage",
            "joint_only_projection_off",
            "joint_only_full_compiled",
        ),
        default=None,
    )
    delay.add_argument("--smoke", action="store_true")

    references = sub.add_parser("generate-references")
    references.add_argument("--overwrite", action="store_true")

    preview = sub.add_parser("calibrate-preview")
    preview.add_argument("--preview-values", default="0,1,2,3,4")
    preview.add_argument(
        "--orientation-tolerance",
        type=float,
        default=0.10,
        help="Maximum relative orientation-RMSE increase over the best candidate when selecting preview.",
    )
    preview.add_argument("--smoke", action="store_true")

    validation = sub.add_parser("validate-model")
    validation.add_argument("--num-rollouts", type=int, default=20)
    validation.add_argument("--rollout-len", type=int, default=200)

    history_windows = sub.add_parser("reanalyze-model-validation")
    history_windows.add_argument(
        "--input-dir",
        default=str(ROOT / "outputs" / "paper_final" / "diagnostics" / "gru_validation"),
        help="Frozen validation rollout directory to reanalyse without rerunning MuJoCo.",
    )
    history_windows.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "paper_model_validation_history_windows_v1"),
    )
    history_windows.add_argument("--horizons", type=parse_horizons, default=(1, 5, 10, 20))
    history_windows.add_argument("--window-stride", type=int, default=20)
    history_windows.add_argument("--overwrite", action="store_true")

    manifest = sub.add_parser("build-manifest")
    manifest.add_argument("--allow-dirty", action="store_true")
    manifest.add_argument("--profile", choices=["paper", "smoke"], default="paper")
    manifest.add_argument("--reference-manifest", default=str(LEGACY_PAPER_ROOT / "references" / "manifest.json"))
    manifest.add_argument("--preview-calibration", default=str(LEGACY_PAPER_ROOT / "calibration" / "preview.json"))
    manifest.add_argument("--paper-control-commit", default=None)
    manifest.add_argument("--analysis-commit", default=None)

    run = sub.add_parser("run")
    run.add_argument("--manifest", default=None)
    run.add_argument("--suite", choices=["main", "ablation", "history_alignment", "effort_pareto", "delay_sweep", "delay_sweep_components", "projection_choice", "preview", "oracle", "task_cost", "all"], default="main")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--case-limit", type=int, default=None)
    run.add_argument("--legacy-mpc-root", default=str(LEGACY_MPC_ROOT))
    run.add_argument("--legacy-ik-root", default=str(LEGACY_IK_ROOT))

    summary = sub.add_parser("summarize")
    summary.add_argument("--suite", choices=["main", "ablation", "history_alignment", "effort_pareto", "delay_sweep", "delay_sweep_components", "projection_choice", "preview", "oracle", "task_cost", "smoke", "all"], default="all")
    summary.add_argument("--bootstrap-samples", type=int, default=10000)

    sub.add_parser("smoke")
    return top


def _require_models() -> None:
    for path in (CHECKPOINT, NORMALIZER, MODEL_XML):
        if not path.is_file():
            raise FileNotFoundError(path)


def generate_calibration_reference(output: Path, steps: int, overwrite: bool) -> Path:
    if steps < 100:
        raise ValueError("--steps must be at least 100")
    destination = output / "calibration" / "task_space_delay_calibration" / "reference.npz"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {destination}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    robot = load_robot_spec()
    # This is intentionally distinct from every reported trajectory.  It must
    # nevertheless exercise the exact task-space final-pool scoring path.
    config = ReferenceConfig(
        shape_name="ellipse", repeat_count=1, lap_duration=max((steps - 300) * 0.01, 4.0),
        ellipse_axis_a=0.022, ellipse_axis_b=0.014,
        plane_axis_u=(1.0, 0.0, 0.0), plane_axis_v=(0.0, 0.0, 1.0),
        start_hold_duration=0.5, joint_departure_duration=1.0, approach_duration=1.0,
        return_duration=1.0, joint_return_duration=1.0, final_hold_duration=0.5,
        safe_departure_mode="auto",
    )
    bundle = build_reference(
        config, model, robot.home_q, 0.01, 20, 16, robot_spec=robot
    )
    if bundle.execution_steps < steps:
        raise RuntimeError("Task-space calibration reference is shorter than requested")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return save_reference_bundle(bundle, destination.parent)


def _base_args() -> argparse.Namespace:
    args = RUNNER.parse_args([])
    args.checkpoint = str(CHECKPOINT); args.normalizer = str(NORMALIZER)
    args.model_type = "gru"; args.history_len = 16; args.device = "cuda"
    args.horizon = 20; args.num_samples = 128; args.cem_iters = 2
    args.rollout_batch_size = 128
    args.replan_interval_steps = 5; args.mpc_warmup_plans = 1
    args.controller_mode = "mpc"; args.mpc_policy = "residual"
    args.delay_protocol = "full"; args.dynamics_backend = "learned"
    # The final paper method uses cheap population screening followed by exact
    # compiled projection and re-evaluation of the selected candidates.  The
    # 100 Hz execution layer independently applies the shared physical filter.
    args.planner_projection = "on"
    args.planner_projection_backend = "compiled"
    args.planner_projection_strategy = "two_stage"
    args.residual_cost_semantics = "requested"
    args.packet_residual_semantics = "requested"
    args.residual_feasibility_semantics = "finite"
    args.nominal_command_semantics = "raw_ik"
    args.visualize = False; args.settle_steps = 50
    args.ik_command_projection = "physical"
    args.exact_task_space_cost = "on"
    args.residual_parameterization = "full"
    args.stage_one_task_space_cost = "off"
    args.stage_one_task_compile = "off"
    args.cem_execute = "lowest_cost"
    args.mpc_preview_nominal_steps = 0
    args.w_task_position = 1.0; args.w_task_orientation = 0.25
    args.task_position_scale_m = 0.05; args.task_orientation_scale_rad = float(np.deg2rad(5.0))
    args.uncertainty_mode = "off"; args.cost_profile = "blackbox"
    args.payload_level = 0; args.actuator_gain_level = 0
    args.force_pulse_level = 0; args.observation_noise_level = 0
    return args


CALIBRATION_VARIANTS = {
    "task_space_two_stage": {
        "exact_task_space_cost": "on",
        "planner_projection": "on",
        "planner_projection_backend": "compiled",
        "planner_projection_strategy": "two_stage",
    },
    "joint_only_two_stage": {
        "exact_task_space_cost": "off",
        "planner_projection": "on",
        "planner_projection_backend": "compiled",
        "planner_projection_strategy": "two_stage",
    },
    "joint_only_projection_off": {
        "exact_task_space_cost": "off",
        "planner_projection": "off",
        "planner_projection_backend": "compiled",
        "planner_projection_strategy": "full",
    },
    "joint_only_full_compiled": {
        "exact_task_space_cost": "off",
        "planner_projection": "on",
        "planner_projection_backend": "compiled",
        "planner_projection_strategy": "full",
    },
}


def calibrate_delay(
    output: Path,
    samples: int,
    provisional: int,
    guard_ms: float,
    smoke: bool,
    exact_task_space_cost: str = "on",
    variant: str | None = None,
) -> Path:
    if samples <= 0 or provisional <= 0 or guard_ms < 0:
        raise ValueError("samples/provisional delay must be positive and guard non-negative")
    if variant is None:
        variant = "task_space_two_stage" if exact_task_space_cost == "on" else "joint_only_two_stage"
    configuration = CALIBRATION_VARIANTS[variant]
    suffix = f"delay_{variant}{'_smoke' if smoke else ''}.json"
    path = output / "calibration" / suffix
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable calibration: {path}")
    reference = output / "calibration" / "task_space_delay_calibration" / "reference.npz"
    if not reference.is_file():
        raise FileNotFoundError("Run generate-calibration-reference first")
    args = _base_args()
    for key, value in configuration.items():
        setattr(args, key, value)
    args.multirate_mode = "threaded_asap"; args.anticipation_delay_steps = provisional
    args.planner_guard_ms = 0.0; args.reference_mode = "task"; args.reference_file = str(reference)
    args.max_execution_steps = 40 if smoke else None
    args.horizon = 3 if smoke else args.horizon; args.num_samples = 8 if smoke else args.num_samples
    args.cem_iters = 1 if smoke else args.cem_iters
    collected: list[float] = []
    episode = 0
    target = min(samples, 3) if smoke else samples
    while len(collected) < target:
        args.seed = episode
        result = RUNNER.run_closed_loop_mpc(deepcopy(args))["arrays"]
        values = np.asarray(result.get("planner_end_to_end_latency_s", np.empty(0)), dtype=np.float64)
        collected.extend(values[np.isfinite(values)].tolist())
        episode += 1
        if smoke and episode >= 3 and not collected:
            raise RuntimeError("Threaded smoke produced no E2E latency samples")
    values = np.asarray(collected[:target])
    p95 = float(np.percentile(values, 95))
    delay = int(math.ceil((p95 + guard_ms / 1000.0) / 0.01))
    write_immutable_json(path, {
        "definition": "ceil((P95(snapshot_to_publication)+guard)/control_dt)",
        "variant": variant,
        "samples": values.tolist(), "p50_s": float(np.percentile(values, 50)), "p95_s": p95,
        "guard_ms": guard_ms, "control_dt_s": 0.01, "anticipation_delay_steps": delay,
        "provisional_delay_steps": provisional, "source": "planner_end_to_end_latency_s",
        "calibration_reference": file_identity(reference),
        "checkpoint": file_identity(CHECKPOINT), "normalizer": file_identity(NORMALIZER),
        "method": {key: getattr(args, key) for key in (
            "exact_task_space_cost", "planner_projection", "planner_projection_backend",
            "planner_projection_strategy", "horizon", "num_samples", "cem_iters",
            "rollout_batch_size", "residual_parameterization", "stage_one_task_space_cost",
            "stage_one_task_compile", "cem_execute", "mpc_preview_nominal_steps",
        )},
    })
    return path


def _reference_configs() -> dict[str, ReferenceConfig]:
    common = dict(repeat_count=3, safe_departure_mode="auto", start_hold_duration=0.5,
                  joint_departure_duration=2.0, approach_duration=2.0, return_duration=2.0,
                  joint_return_duration=2.0, final_hold_duration=0.5)
    return {
        "circle": ReferenceConfig(shape_name="circle", lap_duration=3.0, circle_radius=0.05, **common),
        "figure8": ReferenceConfig(shape_name="figure8", lap_duration=3.0, figure8_axis_a=0.05, figure8_axis_b=0.03, **common),
        "fast_ellipse": ReferenceConfig(shape_name="ellipse", lap_duration=2.0, ellipse_axis_a=0.055, ellipse_axis_b=0.03, **common),
        "rounded_square": ReferenceConfig(shape_name="rounded_square", lap_duration=3.0, square_half_side=0.03, rounded_square_corner_radius=0.008, **common),
        "preview_calibration": ReferenceConfig(shape_name="ellipse", repeat_count=1, lap_duration=4.0,
            ellipse_axis_a=0.035, ellipse_axis_b=0.02, plane_axis_u=(1.0, 0.0, 0.0),
            plane_axis_v=(0.0, 0.0, 1.0), safe_departure_mode="auto"),
    }


def generate_references(output: Path, overwrite: bool) -> None:
    delay_path = output / "calibration" / "delay.json"
    if not delay_path.is_file():
        raise FileNotFoundError("Run calibrate-delay first")
    delay = int(load_json(delay_path)["anticipation_delay_steps"])
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    robot = load_robot_spec()
    records: list[dict[str, Any]] = []
    for label, config in _reference_configs().items():
        destination = output / "references" / label
        path = destination / "reference.npz"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
        bundle = build_reference(
            config,
            model,
            robot.home_q,
            0.01,
            20,
            max(delay, 8, 4),
            robot_spec=robot,
        )
        saved = save_reference_bundle(bundle, destination)
        records.append({"label": label, "file": file_identity(saved), "config": bundle.metadata["config"]})
    manifest = output / "references" / "manifest.json"
    if manifest.exists() and overwrite:
        manifest.unlink()
    write_immutable_json(manifest, {"schema_version": 1, "references": records})


def _select_preview(rows: list[dict[str, Any]], orientation_tolerance: float) -> tuple[dict[str, Any], float]:
    if orientation_tolerance < 0.0:
        raise ValueError("orientation tolerance must be non-negative")
    orientations = np.asarray([float(row["orientation_rmse_rad"]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(orientations)):
        raise ValueError("preview calibration produced a non-finite orientation RMSE")
    limit = float(np.min(orientations) * (1.0 + orientation_tolerance))
    eligible = [row for row in rows if float(row["orientation_rmse_rad"]) <= limit]
    selected = min(eligible, key=lambda row: (float(row["tcp_rmse_m"]), int(row["preview_steps"])))
    return selected, limit


def calibrate_preview(output: Path, values: list[int], smoke: bool, orientation_tolerance: float = 0.10) -> Path:
    path = output / "calibration" / ("preview_smoke.json" if smoke else "preview.json")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable calibration: {path}")
    reference = output / "references" / "preview_calibration" / "reference.npz"
    if not reference.is_file():
        raise FileNotFoundError("Run generate-references first")
    rows: list[dict[str, Any]] = []
    for preview in values:
        if preview < 0:
            raise ValueError("preview values must be non-negative")
        args = _base_args(); args.controller_mode = "ik_direct"; args.multirate_mode = "synchronous"
        args.checkpoint = None; args.normalizer = None; args.reference_mode = "task"; args.reference_file = str(reference)
        args.exact_task_space_cost = "off"
        args.ik_preview_steps = preview; args.max_execution_steps = 10 if smoke else None
        result = RUNNER.run_closed_loop_mpc(args)
        run_dir = output / "calibration" / "preview" / f"p{preview}"
        save_mpc_run(run_dir, result["arrays"], result["rows"])
        row = summarize_arrays(f"PreviewIK_{preview}", result["arrays"]); row["preview_steps"] = preview; rows.append(row)
    selected, orientation_limit = _select_preview(rows, orientation_tolerance)
    write_csv(output / "calibration" / "preview" / "summary.csv", rows)
    write_immutable_json(path, {
        "candidate_steps": values,
        "selected_steps": int(selected["preview_steps"]),
        "criterion": "minimum calibration TCP RMSE subject to an orientation-RMSE tolerance; ties choose smaller preview",
        "orientation_tolerance_fraction": orientation_tolerance,
        "orientation_rmse_limit_rad": orientation_limit,
        "selected_tcp_rmse_m": float(selected["tcp_rmse_m"]),
        "selected_orientation_rmse_rad": float(selected["orientation_rmse_rad"]),
    })
    return path


def validate_model(output: Path, num_rollouts: int, rollout_len: int) -> None:
    _require_models()
    destination = output / "diagnostics" / "gru_validation"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen model validation: {destination}")
    command = [
        sys.executable, str(ROOT / "dynamics_modeling" / "scripts" / "eval_dynamics.py"),
        "--checkpoint", str(CHECKPOINT), "--normalizer", str(NORMALIZER),
        "--model_type", "gru", "--history_len", "16",
        "--num_rollouts", str(num_rollouts), "--rollout_len", str(rollout_len),
        "--horizons", "1,5,10,20", "--teacher_forcing", "--seed", "20260730",
        "--action_std_groups", "0.5:10,0.8:10",
        "--save_dir", str(destination),
    ]
    if num_rollouts != 20:
        command[command.index("--action_std_groups") + 1] = f"0.5:{num_rollouts}"
    subprocess.run(command, cwd=ROOT, check=True)
    files = sorted(destination.glob("evaluation_rollout_*.npz"))
    validation_manifest = {
        "description": "Post-checkpoint frozen evaluation rollouts; not used for training",
        "checkpoint": file_identity(CHECKPOINT), "normalizer": file_identity(NORMALIZER),
        "training_dataset": file_identity(TRAINING_DATASET),
        "training_action_std_normalized_support": [0.5, 0.8],
        "evaluation_action_std_groups": {"0.5": 10, "0.8": 10} if num_rollouts == 20 else {"0.5": num_rollouts},
        "rollouts": [file_identity(path) for path in files], "seed": 20260730,
        "horizons": [1, 5, 10, 20], "one_step_name": "ground_truth_history_one_step",
        "divergence": "nonfinite, q outside joint bounds by >0.05 rad, or |dq|>25 rad/s",
    }
    write_json(destination / "validation_manifest.json", validation_manifest)
    aggregate_rows: list[dict[str, Any]] = []
    aggregate_json: dict[str, Any] = {"groups": {}}
    for filename in ("horizon_metrics.csv", "teacher_forcing_metrics.csv"):
        with (destination / filename).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in rows:
            key = (str(row["mode"]), str(row["action_std"]), str(row["horizon"]))
            groups.setdefault(key, []).append(row)
        for key, members in groups.items():
            group_name = ":".join(key)
            aggregate_json["groups"].setdefault(group_name, {})
            ignored = {"rollout", "mode", "horizon", "action_std", "warmup_steps"}
            for metric in sorted(set(members[0]) - ignored):
                values = np.asarray([float(member[metric]) for member in members], dtype=np.float64)
                finite_mask = np.isfinite(values)
                finite = values[finite_mask]
                if not finite.size:
                    continue
                finite_indices = np.flatnonzero(finite_mask)
                worst_index = int(finite_indices[int(np.argmax(finite))])
                stats = {
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                    "median": float(np.median(finite)),
                    "p95": float(np.percentile(finite, 95)),
                    "worst": float(np.max(finite)),
                    "worst_rollout": int(members[worst_index]["rollout"]),
                }
                aggregate_json["groups"][group_name][metric] = stats
                aggregate_rows.append({
                    "mode": key[0], "action_std": key[1], "horizon": key[2],
                    "metric": metric, **stats,
                })
    write_csv(destination / "gru_validation_aggregate.csv", aggregate_rows)
    write_json(destination / "gru_validation_aggregate.json", aggregate_json)


def build_manifest(
    output: Path,
    allow_dirty: bool,
    profile: str,
    reference_manifest: Path | None = None,
    preview_calibration: Path | None = None,
    paper_control_commit: str | None = None,
    analysis_commit: str | None = None,
) -> Path:
    _require_models()
    suffix = "_smoke" if profile == "smoke" else ""
    calibration_files = {
        name: output / "calibration" / f"delay_{name}{suffix}.json"
        for name in CALIBRATION_VARIANTS
    }
    reference_manifest = resolve(reference_manifest or LEGACY_PAPER_ROOT / "references" / "manifest.json")
    preview_file = resolve(preview_calibration or LEGACY_PAPER_ROOT / "calibration" / "preview.json")
    for path in (*calibration_files.values(), preview_file, reference_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    environment = environment_snapshot(ROOT)
    if environment["git_dirty"] and not allow_dirty:
        raise RuntimeError("Formal paper manifest requires a clean Git worktree; use --allow-dirty only for smoke")
    base = vars(_base_args()).copy()
    for key in ("save_dir", "seed", "reference_file", "reference_mode", "controller_mode", "multirate_mode"):
        base.pop(key, None)
    reference_payload = load_json(reference_manifest)
    reference_records = {record["label"]: record["file"] for record in reference_payload["references"]}
    references = {name: reference_records[name] for name in (*TRAJECTORIES, "preview_calibration")}
    frozen_method = {
        key: base[key]
        for key in (
            "model_type", "history_len", "horizon", "num_samples", "rollout_batch_size", "cem_iters",
            "replan_interval_steps", "planner_projection", "planner_projection_backend",
            "planner_projection_strategy", "residual_cost_semantics", "packet_residual_semantics",
            "residual_feasibility_semantics", "nominal_command_semantics", "ik_command_projection",
            "feedback_kq", "feedback_kdq", "feedback_max", "planner_guard_ms",
            "exact_task_space_cost", "w_task_position", "w_task_orientation",
            "task_position_scale_m", "task_orientation_scale_rad", "uncertainty_mode",
            "cost_profile", "payload_level", "actuator_gain_level", "force_pulse_level",
            "observation_noise_level", "residual_parameterization",
            "stage_one_task_space_cost", "stage_one_task_compile", "cem_execute",
            "mpc_preview_nominal_steps",
        )
    }
    frozen_method["control_dt_s"] = 0.01
    frozen_method["exact_final_pool_backend"] = "mujoco_fk"
    frozen_method["exact_final_pool_candidate_policy"] = "final_elites_mean_best_baseline"
    payload = {
        "schema_version": 6, "kind": "paper_delay_aware", "profile": profile,
        "control_semantics_version": 2, "projection_semantics_version": 2,
        "projection_backend": "shared_physical_v2", "projection_tolerance": 1e-6,
        "environment": environment, "checkpoint": file_identity(CHECKPOINT), "normalizer": file_identity(NORMALIZER),
        "paper_control_commit": paper_control_commit or environment["git_commit"],
        "analysis_commit": analysis_commit or environment["git_commit"],
        "model_xml": file_identity(MODEL_XML),
        "delay_calibrations": {name: load_json(path) for name, path in calibration_files.items()},
        "delay_calibration": load_json(calibration_files["task_space_two_stage"]),
        "joint_only_delay_calibration": load_json(calibration_files["joint_only_two_stage"]),
        "preview_calibration": load_json(preview_file), "frozen_method": frozen_method,
        "base_run_args": base, "references": references,
        "reference_manifest": file_identity(reference_manifest),
        "projection_common_delay_steps": 6,
        "projection_noninferiority_margin_fraction": 0.05,
        "paired_cem_seeds": [0, 1, 2, 3, 4], "delay_sweep_seeds": [0, 1, 2],
        "delay_sweep_steps": [0, 2, 4, 6, 8], "bootstrap_seed": 20260722,
    }
    path = output / "manifests" / f"{profile}.json"
    write_immutable_json(path, payload)
    return path


def _case(label: str, trajectory: str, seed: int, mode: str, protocol: str, delay: int, **extra: Any) -> dict[str, Any]:
    return {"label": label, "trajectory": trajectory, "seed": seed, "multirate_mode": mode,
            "delay_protocol": protocol, "delay_steps": delay, **extra}


def suite_cases(manifest: dict[str, Any], suite: str) -> list[dict[str, Any]]:
    delay = int(manifest["delay_calibration"]["anticipation_delay_steps"])
    joint_only_delay = int(manifest.get("joint_only_delay_calibration", manifest["delay_calibration"])["anticipation_delay_steps"])
    seeds = [int(value) for value in manifest["paired_cem_seeds"]]
    cases: list[dict[str, Any]] = []
    if suite == "main":
        for trajectory in TRAJECTORIES:
            cases.append(_case("DirectIK", trajectory, 0, "synchronous", "full", 0, controller="ik_direct"))
            for seed in seeds:
                cases.extend([
                    _case("IdealZeroDelay", trajectory, seed, "virtual_asap", "full", 0),
                    _case("NaiveDelayed", trajectory, seed, "virtual_asap", "naive_delayed", delay),
                    _case("FullVirtual", trajectory, seed, "virtual_asap", "full", delay),
                    _case("ThreadedASAP", trajectory, seed, "threaded_asap", "full", delay),
                ])
    elif suite == "ablation":
        labels = (("FullVirtual", "full"), ("NoFutureAlignment", "no_future_alignment"),
                  ("NoReanchor", "no_reanchor"), ("NoFeedback", "no_feedback"))
        for trajectory in TRAJECTORIES:
            for seed in seeds:
                cases.extend(_case(label, trajectory, seed, "virtual_asap", protocol, delay) for label, protocol in labels)
    elif suite == "history_alignment":
        labels = (
            ("FullAlignment", "full"),
            ("StaleHistory", "stale_history"),
            ("NoAlignment", "no_future_alignment"),
        )
        for trajectory in ("circle", "figure8", "fast_ellipse"):
            for seed in seeds:
                cases.extend(
                    _case(label, trajectory, seed, "virtual_asap", protocol, delay)
                    for label, protocol in labels
                )
    elif suite == "effort_pareto":
        for trajectory in TRAJECTORIES:
            for seed in seeds:
                for scale in EFFORT_PARETO_SCALES:
                    cases.append(_case(
                        f"EffortScale{scale:g}", trajectory, seed,
                        "virtual_asap", "full", delay, effort_scale=scale,
                    ))
    elif suite == "delay_sweep":
        for trajectory in ("circle", "fast_ellipse"):
            for seed in manifest["delay_sweep_seeds"]:
                for imposed in manifest["delay_sweep_steps"]:
                    cases.append(_case(f"Full_D{imposed}", trajectory, int(seed), "virtual_asap", "full", int(imposed)))
                    cases.append(_case(f"Naive_D{imposed}", trajectory, int(seed), "virtual_asap", "naive_delayed", int(imposed)))
    elif suite == "delay_sweep_components":
        labels = (
            ("Naive", "naive_delayed"),
            ("AnchorOnly", "anchor_only"),
            ("AnchorReanchor", "no_feedback"),
            ("Full", "full"),
        )
        for trajectory in ("circle", "fast_ellipse"):
            for seed in manifest["delay_sweep_seeds"]:
                for imposed in (2, 4, 6, 8):
                    cases.extend(
                        _case(
                            f"{label}_D{imposed}", trajectory, int(seed),
                            "virtual_asap", protocol, imposed,
                        )
                        for label, protocol in labels
                    )
    elif suite == "preview":
        preview = int(manifest["preview_calibration"]["selected_steps"])
        for trajectory in TRAJECTORIES:
            cases.append(_case("PreviewIK", trajectory, 0, "synchronous", "full", 0, controller="ik_direct", preview_steps=preview))
    elif suite == "oracle":
        for trajectory in ("circle", "fast_ellipse"):
            for seed in (0, 1, 2):
                cases.append(_case("LearnedFull", trajectory, seed, "virtual_asap", "full", delay))
                cases.append(_case("OracleUpperBound", trajectory, seed, "virtual_asap", "full", delay, dynamics_backend="mujoco_oracle"))
    elif suite == "task_cost":
        # Fixed-D rows isolate scoring; threaded rows include each method's
        # deployment latency cost.  Use matched seeds, never 100 Hz ticks.
        for trajectory in ("circle", "figure8"):
            for seed in (0, 1, 2):
                cases.extend([
                    _case("JointOnlyFixedD6", trajectory, seed, "virtual_asap", "full", 6, exact_task_space_cost="off"),
                    _case("TaskSpaceFixedD6", trajectory, seed, "virtual_asap", "full", 6, exact_task_space_cost="on"),
                    _case("JointOnlyDeployed", trajectory, seed, "threaded_asap", "full", joint_only_delay, exact_task_space_cost="off"),
                    _case("TaskSpaceDeployed", trajectory, seed, "threaded_asap", "full", delay, exact_task_space_cost="on"),
                ])
    elif suite == "projection_choice":
        calibrations = manifest["delay_calibrations"]
        common_delay = int(manifest["projection_common_delay_steps"])
        variants = (
            ("ProjectionOff", "off", "compiled", "full", "joint_only_projection_off"),
            ("FullCompiled", "on", "compiled", "full", "joint_only_full_compiled"),
            ("TwoStageCompiled", "on", "compiled", "two_stage", "joint_only_two_stage"),
        )
        for trajectory in ("circle", "figure8"):
            for seed in (0, 1, 2):
                for label, projection, backend, strategy, calibration_name in variants:
                    common = {
                        "exact_task_space_cost": "off",
                        "planner_projection": projection,
                        "planner_projection_backend": backend,
                        "planner_projection_strategy": strategy,
                    }
                    cases.append(_case(
                        f"{label}CommonD6", trajectory, seed, "virtual_asap", "full",
                        common_delay, evaluation_set="common_d", variant=label, **common,
                    ))
                    deployed_delay = int(calibrations[calibration_name]["anticipation_delay_steps"])
                    cases.append(_case(
                        f"{label}Deployed", trajectory, seed, "threaded_asap", "full",
                        deployed_delay, evaluation_set="deployed", variant=label, **common,
                    ))
    else:
        raise ValueError(suite)
    return cases


@lru_cache(maxsize=4)
def _legacy_records(root_text: str, kind: str) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for path in Path(root_text).glob("**/run_fingerprint.json"):
        try:
            fingerprint = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        payload = fingerprint.get("payload", {})
        if payload.get("kind") != kind:
            continue
        records.append({
            "path": path.parent,
            "sha256": fingerprint.get("sha256", ""),
            "payload": payload,
        })
    return tuple(records)


def _legacy_case(
    manifest: dict[str, Any],
    case: dict[str, Any],
    suite: str,
    legacy_mpc_root: Path,
    legacy_ik_root: Path,
) -> dict[str, Any] | None:
    label = str(case["label"])
    trajectory = str(case["trajectory"])
    seed = int(case["seed"])
    if case.get("controller") == "ik_direct" or label == "DirectIK":
        records = _legacy_records(str(legacy_ik_root), "direct_ik_robustness")
        wanted_method = "preview" if int(case.get("preview_steps", 0)) else "physical"
        wanted_case = f"{trajectory}_seed_0"
    else:
        mapping = {
            "IdealZeroDelay": "IdealZeroDelay",
            "NaiveDelayed": "NaiveDelayed",
            "FullVirtual": "VirtualDelayAware",
            "ThreadedASAP": "ThreadedAsync",
        }
        if suite == "delay_sweep_components" and int(case["delay_steps"]) == 6:
            mapping = {
                **mapping,
                "Full_D6": "VirtualDelayAware",
                "Naive_D6": "NaiveDelayed",
            }
        wanted_method = mapping.get(label)
        if wanted_method is None:
            return None
        records = _legacy_records(str(legacy_mpc_root), "delay_aware_mpc_robustness")
        wanted_case = f"{trajectory}_seed_{seed}"
    expected_reference = str(manifest["references"][trajectory]["sha256"])
    for record in records:
        payload = record["payload"]
        method = str(payload.get("method", payload.get("projection", "")))
        condition = payload.get("condition", {})
        if method != wanted_method or payload.get("case_id") != wanted_case or condition.get("name") != "nominal":
            continue
        if payload.get("reference", {}).get("sha256") != expected_reference:
            continue
        config = payload.get("run_config", {})
        compatibility = {
            "horizon": 20, "num_samples": 128, "cem_iters": 2,
            "rollout_batch_size": 128, "planner_projection": "on",
            "planner_projection_backend": "compiled",
            "planner_projection_strategy": "two_stage",
            "residual_cost_semantics": "requested",
            "packet_residual_semantics": "requested",
            "residual_feasibility_semantics": "finite",
            "nominal_command_semantics": "raw_ik",
            "exact_task_space_cost": "off" if wanted_method in {"physical", "preview"} else "on",
        }
        if any(config.get(key) != value for key, value in compatibility.items()):
            continue
        backfilled = _legacy_compatibility_backfill(config)
        if backfilled is None:
            continue
        if not (record["path"] / "rollout.npz").is_file():
            continue
        return {
            **case,
            "suite": suite,
            "fingerprint": record["sha256"],
            "run_dir": str(record["path"]),
            "evidence_cohort": "historical_compatible",
            "compatibility_backfill": backfilled,
        }
    return None


def _legacy_compatibility_backfill(config: dict[str, Any]) -> dict[str, Any] | None:
    backfilled = {
        "residual_parameterization": config.get("residual_parameterization", "full"),
        "stage_one_task_space_cost": config.get("stage_one_task_space_cost", "off"),
        "stage_one_task_compile": config.get("stage_one_task_compile", "off"),
        "mpc_preview_nominal_steps": config.get("mpc_preview_nominal_steps", 0),
    }
    expected = {
        "residual_parameterization": "full",
        "stage_one_task_space_cost": "off",
        "stage_one_task_compile": "off",
        "mpc_preview_nominal_steps": 0,
    }
    return backfilled if backfilled == expected else None


def _run_case(
    output: Path,
    manifest: dict[str, Any],
    case: dict[str, Any],
    resume: bool,
    suite: str,
    legacy_mpc_root: Path,
    legacy_ik_root: Path,
) -> dict[str, Any]:
    args = RUNNER.parse_args([])
    for key, value in manifest["base_run_args"].items():
        if hasattr(args, key): setattr(args, key, value)
    reference = Path(manifest["references"][case["trajectory"]]["path"])
    args.reference_mode = "task"; args.reference_file = str(reference); args.seed = int(case["seed"])
    args.multirate_mode = case["multirate_mode"]; args.delay_protocol = case["delay_protocol"]
    args.anticipation_delay_steps = int(case["delay_steps"]); args.controller_mode = case.get("controller", "mpc")
    args.ik_preview_steps = int(case.get("preview_steps", 0)); args.dynamics_backend = case.get("dynamics_backend", "learned")
    for key in (
        "exact_task_space_cost", "planner_projection",
        "planner_projection_backend", "planner_projection_strategy",
    ):
        if key in case:
            setattr(args, key, str(case[key]))
    if "effort_scale" in case:
        scale = float(case["effort_scale"])
        for key in ("w_servo", "w_residual_velocity", "w_residual_acceleration", "w_first"):
            setattr(args, key, float(getattr(args, key)) * scale)
    if args.controller_mode == "ik_direct":
        args.checkpoint = args.normalizer = None
        args.exact_task_space_cost = "off"
    fingerprint_case = {key: value for key, value in case.items() if key != "label"}
    payload = {"manifest_commit": manifest["environment"]["git_commit"], "case": fingerprint_case,
               "run_args": {key: value for key, value in vars(args).items() if key != "save_dir"},
               "reference": file_identity(reference), "checkpoint": None if not args.checkpoint else manifest["checkpoint"],
               "normalizer": None if not args.normalizer else manifest["normalizer"]}
    fingerprint = run_fingerprint(payload)
    run_dir = output / "runs" / "cache" / fingerprint["sha256"]
    arrays = load_completed_rollout(run_dir, fingerprint)
    if arrays is None:
        legacy = _legacy_case(manifest, case, suite, legacy_mpc_root, legacy_ik_root)
        if legacy is not None:
            if resume:
                print(f"Reused compatible historical rollout: {legacy['run_dir']}")
            return legacy
        args.save_dir = str(run_dir)
        result = RUNNER.run_closed_loop_mpc(args); arrays = result["arrays"]
        save_mpc_run(run_dir, arrays, result["rows"], result.get("planner_events"))
        write_json(run_dir / "run_fingerprint.json", fingerprint)
    elif resume:
        print(f"Reused completed rollout: {run_dir}")
    return {**case, "suite": suite, "fingerprint": fingerprint["sha256"], "run_dir": str(run_dir)}


def run_suite(
    output: Path,
    manifest_path: Path,
    suite: str,
    resume: bool,
    case_limit: int | None,
    legacy_mpc_root: Path = LEGACY_MPC_ROOT,
    legacy_ik_root: Path = LEGACY_IK_ROOT,
) -> None:
    manifest = load_json(manifest_path)
    if int(manifest.get("schema_version", 0)) < 6:
        raise ValueError("Formal paper runs require a schema-v6 manifest")
    frozen = manifest.get("frozen_method")
    required_frozen = {
        "model_type": "gru", "history_len": 16, "horizon": 20, "num_samples": 128,
        "cem_iters": 2, "planner_projection": "on", "planner_projection_backend": "compiled",
        "planner_projection_strategy": "two_stage", "residual_cost_semantics": "requested",
        "packet_residual_semantics": "requested", "residual_feasibility_semantics": "finite",
        "nominal_command_semantics": "raw_ik", "control_dt_s": 0.01,
        "exact_task_space_cost": "on", "w_task_position": 1.0, "w_task_orientation": 0.25,
        "task_position_scale_m": 0.05, "task_orientation_scale_rad": float(np.deg2rad(5.0)),
        "uncertainty_mode": "off", "cost_profile": "blackbox", "payload_level": 0,
        "actuator_gain_level": 0, "force_pulse_level": 0, "observation_noise_level": 0,
        "rollout_batch_size": 128, "residual_parameterization": "full",
        "stage_one_task_space_cost": "off", "stage_one_task_compile": "off",
        "cem_execute": "lowest_cost", "mpc_preview_nominal_steps": 0,
        "exact_final_pool_backend": "mujoco_fk",
        "exact_final_pool_candidate_policy": "final_elites_mean_best_baseline",
    }
    if not isinstance(frozen, dict) or any(frozen.get(key) != value for key, value in required_frozen.items()):
        raise ValueError("Formal paper runs require the compiled two-stage frozen method")
    suites = (
        "main", "ablation", "history_alignment", "effort_pareto", "delay_sweep", "delay_sweep_components",
        "projection_choice", "preview", "oracle", "task_cost",
    ) if suite == "all" else (suite,)
    for name in suites:
        cases = suite_cases(manifest, name)
        if case_limit is not None: cases = cases[:case_limit]
        entries = [
            _run_case(output, manifest, case, resume, name, legacy_mpc_root, legacy_ik_root)
            for case in cases
        ]
        write_json(output / "runs" / "indexes" / f"{name}.json", {"suite": name, "entries": entries})
        summarize(output, name, 10000)


def summarize(output: Path, suite: str, bootstrap_samples: int) -> None:
    suites = (
        "main", "ablation", "history_alignment", "effort_pareto", "delay_sweep", "delay_sweep_components",
        "projection_choice", "preview", "oracle", "task_cost", "smoke",
    ) if suite == "all" else (suite,)
    for name in suites:
        index_path = output / "runs" / "indexes" / f"{name}.json"
        if not index_path.is_file():
            continue
        entries = load_json(index_path)["entries"]
        rows: list[dict[str, Any]] = []
        for entry in entries:
            with np.load(Path(entry["run_dir"]) / "rollout.npz", allow_pickle=False) as archive:
                arrays = {key: np.asarray(archive[key]) for key in archive.files}
            event_path = Path(entry["run_dir"]) / "planner_events.jsonl"
            events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()] if event_path.is_file() else []
            row = summarize_arrays(str(entry["label"]), arrays, events)
            row.update({key: entry[key] for key in ("trajectory", "seed", "label")})
            for key in ("evaluation_set", "variant", "effort_scale"):
                if key in entry:
                    row[key] = entry[key]
            row["case_id"] = f"{entry['trajectory']}:{entry['seed']}"
            rows.append(row)
        control_versions = {int(row["control_semantics_version"]) for row in rows}
        projection_versions = {int(row["projection_semantics_version"]) for row in rows}
        if len(control_versions) != 1 or len(projection_versions) != 1:
            raise ValueError(
                f"Refusing to mix semantics versions in {name}: "
                f"control={sorted(control_versions)}, projection={sorted(projection_versions)}"
            )
        write_csv(output / "summaries" / f"{name}.csv", rows)
        groups = (
            ("delay_protocol", "trajectory", "delay_steps")
            if name in {"delay_sweep", "delay_sweep_components"}
            else ("label", "trajectory")
        )
        write_csv(output / "summaries" / f"{name}_aggregate.csv", aggregate_rows(rows, groups))
        if name == "main":
            comparisons = {
                "FullVirtual_minus_NaiveDelayed": paired_bootstrap_rows(
                    rows, left="NaiveDelayed", right="FullVirtual",
                    metrics=("tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad", "failure_rate"),
                    samples=bootstrap_samples, seed=20260722),
                "ThreadedASAP_minus_NaiveDelayed": paired_bootstrap_rows(
                    rows, left="NaiveDelayed", right="ThreadedASAP",
                    metrics=("tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad", "failure_rate"),
                    samples=bootstrap_samples, seed=20260723),
                "ThreadedASAP_minus_FullVirtual": paired_bootstrap_rows(
                    rows, left="FullVirtual", right="ThreadedASAP",
                    metrics=("tcp_rmse_m", "lap_tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad", "failure_rate"),
                    samples=bootstrap_samples, seed=20260724),
            }
            write_json(output / "summaries" / "main_paired_bootstrap.json", comparisons)
            write_json(output / "summaries" / "latency_recovery.json", latency_recovery(rows))
            main_endpoint(rows, output / "summaries", bootstrap_samples)
        elif name == "ablation":
            comparisons = {}
            for index, variant in enumerate(("NoFutureAlignment", "NoReanchor", "NoFeedback")):
                comparisons[f"{variant}_minus_FullVirtual"] = paired_bootstrap_rows(
                    rows, left="FullVirtual", right=variant,
                    metrics=(
                        "tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad",
                        "failure_rate", "projection_discrepancy_rms_rad",
                        "planner_execution_qref_error_rms_rad", "residual_rms_rad",
                        "feedback_saturation_rate", "residual_saturation_rate",
                        "packet_expiration_count", "velocity_violation_count",
                        "acceleration_violation_count",
                    ),
                    samples=bootstrap_samples, seed=20260722 + index,
                )
            write_json(output / "summaries" / "ablation_paired_bootstrap.json", comparisons)
        elif name == "history_alignment":
            comparisons = {}
            for index, variant in enumerate(("StaleHistory", "NoAlignment")):
                comparisons[f"{variant}_minus_FullAlignment"] = paired_bootstrap_rows(
                    rows, left="FullAlignment", right=variant,
                    metrics=(
                        "tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad",
                        "joint_rmse_rad", "failure_rate", "residual_saturation_rate",
                        "velocity_violation_count", "acceleration_violation_count",
                    ),
                    samples=bootstrap_samples, seed=20260760 + index,
                )
            write_json(output / "summaries" / "history_alignment_paired_bootstrap.json", comparisons)
        elif name == "task_cost":
            comparisons = {}
            for index, (left, right) in enumerate((("JointOnlyFixedD6", "TaskSpaceFixedD6"), ("JointOnlyDeployed", "TaskSpaceDeployed"))):
                comparisons[f"{right}_minus_{left}"] = paired_bootstrap_rows(
                    rows, left=left, right=right,
                    metrics=(
                        "tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad", "joint_rmse_rad",
                        "solve_p95_s", "solve_max_s", "e2e_p95_s", "e2e_max_s",
                        "exact_selection_changed_rate", "baseline_selection_rate",
                        "exact_pool_candidate_count_mean", "exact_pool_valid_count_mean",
                        "exact_final_pool_time_p95_s", "exact_final_pool_time_max_s",
                    ),
                    samples=bootstrap_samples, seed=20260740 + index,
                )
            write_json(output / "summaries" / "task_cost_paired_bootstrap.json", comparisons)
        elif name == "delay_sweep_components":
            delay_sweep_report(rows, output / "summaries", bootstrap_samples)
        elif name == "projection_choice":
            projection_report(
                rows, output / "summaries", bootstrap_samples,
                float(load_json(output / "manifests" / "paper.json")["projection_noninferiority_margin_fraction"]),
            )


def smoke(output: Path) -> None:
    _require_models()
    reference = ROOT / "outputs" / "references" / "circle_3laps" / "reference.npz"
    if not reference.is_file(): raise FileNotFoundError(reference)
    base = _base_args(); base.reference_mode = "task"; base.reference_file = str(reference)
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        base.device = "cpu"
    base.horizon = 3; base.num_samples = 8; base.cem_iters = 1; base.max_execution_steps = 3
    base.replan_interval_steps = 1; base.anticipation_delay_steps = 1; base.mpc_warmup_plans = 0
    specs = [
        ("IdealZeroDelay", "virtual_asap", "full", 0, "mpc"),
        ("FullVirtual", "virtual_asap", "full", 1, "mpc"),
        ("NaiveDelayed", "virtual_asap", "naive_delayed", 1, "mpc"),
        ("AnchorOnly", "virtual_asap", "anchor_only", 1, "mpc"),
        ("NoFutureAlignment", "virtual_asap", "no_future_alignment", 1, "mpc"),
        ("NoReanchor", "virtual_asap", "no_reanchor", 1, "mpc"),
        ("NoFeedback", "virtual_asap", "no_feedback", 1, "mpc"),
        ("DirectIK", "synchronous", "full", 0, "ik_direct"),
    ]
    if cuda_available:
        specs.append(("ThreadedASAP", "threaded_asap", "full", 1, "mpc"))
    entries: list[dict[str, Any]] = []
    for label, mode, protocol, delay, controller in specs:
        args = deepcopy(base); args.multirate_mode = mode; args.delay_protocol = protocol
        args.anticipation_delay_steps = delay; args.controller_mode = controller
        if controller == "ik_direct":
            args.checkpoint = args.normalizer = None
            args.exact_task_space_cost = "off"
        payload = {"smoke": True, "label": label, "args": {key: value for key, value in vars(args).items() if key != "save_dir"}}
        fingerprint = run_fingerprint(payload); run_dir = output / "smoke" / label
        result = RUNNER.run_closed_loop_mpc(args); save_mpc_run(run_dir, result["arrays"], result["rows"], result.get("planner_events"))
        write_json(run_dir / "run_fingerprint.json", fingerprint)
        entries.append({"label": label, "trajectory": "circle_smoke", "seed": 0, "suite": "smoke", "fingerprint": fingerprint["sha256"], "run_dir": str(run_dir)})
    write_json(output / "runs" / "indexes" / "smoke.json", {"suite": "smoke", "entries": entries})
    write_json(output / "smoke" / "environment_status.json", {
        "cuda_available": cuda_available,
        "threaded_smoke": "completed" if cuda_available else "skipped_cuda_unavailable",
    })
    summarize(output, "smoke", 100)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv); output = resolve(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    if args.command == "generate-calibration-reference":
        print(generate_calibration_reference(output, args.steps, args.overwrite))
    elif args.command == "calibrate-delay":
        print(calibrate_delay(
            output, args.samples, args.provisional_delay, args.guard_ms,
            args.smoke, args.exact_task_space_cost, args.variant,
        ))
    elif args.command == "generate-references": generate_references(output, args.overwrite)
    elif args.command == "calibrate-preview":
        values = [int(value) for value in args.preview_values.split(",") if value.strip()]
        print(calibrate_preview(output, values, args.smoke, args.orientation_tolerance))
    elif args.command == "validate-model": validate_model(output, args.num_rollouts, args.rollout_len)
    elif args.command == "reanalyze-model-validation":
        print(reanalyze(
            resolve(args.input_dir),
            resolve(args.output_dir),
            ROOT / "configs" / "robots" / "abb_irb2400.yaml",
            args.horizons,
            args.window_stride,
            args.overwrite,
        ))
    elif args.command == "build-manifest":
        print(build_manifest(
            output,
            args.allow_dirty,
            args.profile,
            Path(args.reference_manifest),
            Path(args.preview_calibration),
            args.paper_control_commit,
            args.analysis_commit,
        ))
    elif args.command == "run":
        manifest = resolve(args.manifest) if args.manifest else output / "manifests" / "paper.json"
        run_suite(
            output, manifest, args.suite, args.resume, args.case_limit,
            resolve(args.legacy_mpc_root), resolve(args.legacy_ik_root),
        )
    elif args.command == "summarize": summarize(output, args.suite, args.bootstrap_samples)
    elif args.command == "smoke": smoke(output)


if __name__ == "__main__":
    main()
