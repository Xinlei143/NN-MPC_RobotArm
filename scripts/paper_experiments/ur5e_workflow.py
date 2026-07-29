"""Reproducible nominal portability validation on the UR5e MuJoCo platform."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mujoco
import numpy as np
import torch

from neural_dynamics.mujoco_env import MuJoCoArmEnv
from neural_dynamics.rollout import load_dynamics_bundle
from mpc.logging import save_mpc_run
from mpc.reference_pipeline import ReferenceConfig, build_reference, save_reference_bundle
from mpc.robot_config import file_sha256, load_robot_spec, validate_robot_model_contract
from scripts.experiment_utils import (
    canonical_sha256,
    environment_snapshot,
    file_identity,
    load_completed_rollout,
    paired_bootstrap_rows,
    run_fingerprint,
    write_immutable_json,
)
from scripts.paper_experiments.evaluation import aggregate_rows, summarize_arrays, write_csv, write_json
from scripts.paper_experiments.reanalyze_gru_history_windows import parse_horizons, reanalyze
from scripts.robustness._runtime import load_runner


RUNNER = load_runner("paper_ur5e_runner")
ROBOT_CONFIG = ROOT / "configs" / "robots" / "ur5e.yaml"
DEFAULT_ROOT = ROOT / "outputs" / "paper_ur5e_v1"
TRAJECTORIES = ("circle", "figure8", "fast_ellipse", "rounded_square")
MPC_LABELS = ("IdealZeroDelay", "NaiveDelayed", "FullVirtual", "ThreadedAsync")


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    top.add_argument("--output-root", default=str(DEFAULT_ROOT))
    top.add_argument("--checkpoint", default=None)
    top.add_argument("--normalizer", default=None)
    top.add_argument("--dataset-manifest", default=None)
    sub = top.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-robot")
    references = sub.add_parser("generate-references")
    references.add_argument(
        "--calibration-steps",
        type=int,
        default=2400,
        help="Minimum requested length used to size the independent delay-calibration trajectory near the ABB reference's 2405 executed steps.",
    )
    references.add_argument("--overwrite", action="store_true")

    validation = sub.add_parser("validate-model")
    validation.add_argument("--num-rollouts", type=int, default=20)
    validation.add_argument("--rollout-len", type=int, default=200)

    history_windows = sub.add_parser("reanalyze-model-validation")
    history_windows.add_argument(
        "--input-dir",
        default=str(ROOT / "outputs" / "paper_ur5e_v2" / "diagnostics" / "gru_validation"),
        help="Frozen validation rollout directory to reanalyse without rerunning MuJoCo.",
    )
    history_windows.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "paper_ur5e_history_windows_v1"),
    )
    history_windows.add_argument("--horizons", type=parse_horizons, default=(1, 5, 10, 20))
    history_windows.add_argument("--window-stride", type=int, default=20)
    history_windows.add_argument("--overwrite", action="store_true")

    delay = sub.add_parser("calibrate-delay")
    delay.add_argument("--samples", type=int, default=500)
    delay.add_argument("--guard-ms", type=float, default=5.0)
    delay.add_argument("--provisional-delay", type=int, default=10)
    delay.add_argument("--smoke", action="store_true")

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--max-steps", type=int, default=5)

    manifest = sub.add_parser("build-manifest")
    manifest.add_argument("--allow-dirty", action="store_true")

    upgrade = sub.add_parser("upgrade-manifest")
    upgrade.add_argument("--manifest", default=None)

    run = sub.add_parser("run")
    run.add_argument("--manifest", default=None)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--case-limit", type=int, default=None)

    summary = sub.add_parser("summarize")
    summary.add_argument("--bootstrap-samples", type=int, default=10000)

    audit = sub.add_parser("audit-freeze")
    audit.add_argument("--manifest", default=None)
    return top


def _require_artifacts(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.checkpoint or not args.normalizer:
        raise ValueError("--checkpoint and --normalizer are required")
    checkpoint = resolve(args.checkpoint)
    normalizer = resolve(args.normalizer)
    if not checkpoint.is_file() or not normalizer.is_file():
        raise FileNotFoundError("UR5e dynamics checkpoint or normalizer does not exist")
    robot = load_robot_spec(ROBOT_CONFIG)
    load_dynamics_bundle(
        checkpoint,
        normalizer,
        "gru",
        robot.n_joints,
        "cpu",
        16,
        expected_robot_spec=robot,
    )
    return checkpoint, normalizer


def _reference_configs(calibration_steps: int = 2400) -> dict[str, ReferenceConfig]:
    if calibration_steps < 100:
        raise ValueError("--calibration-steps must be at least 100")
    common = dict(
        repeat_count=3,
        safe_departure_mode="auto",
        start_hold_duration=0.5,
        joint_departure_duration=2.0,
        approach_duration=2.0,
        return_duration=2.0,
        joint_return_duration=2.0,
        final_hold_duration=0.5,
    )
    return {
        "circle": ReferenceConfig(shape_name="circle", lap_duration=3.0, circle_radius=0.05, **common),
        "figure8": ReferenceConfig(
            shape_name="figure8", lap_duration=3.0,
            figure8_axis_a=0.05, figure8_axis_b=0.03, **common,
        ),
        "fast_ellipse": ReferenceConfig(
            shape_name="ellipse", lap_duration=2.0,
            ellipse_axis_a=0.055, ellipse_axis_b=0.03, **common,
        ),
        "rounded_square": ReferenceConfig(
            shape_name="rounded_square", lap_duration=3.0,
            square_half_side=0.03, rounded_square_corner_radius=0.008, **common,
        ),
        "delay_calibration": ReferenceConfig(
            # Match the ABB delay-reference sizing rule.  Keeping the timing
            # trajectory long enough avoids rebuilding the threaded worker
            # merely to collect the formal 500-sample calibration target.
            shape_name="ellipse", repeat_count=1,
            lap_duration=max((calibration_steps - 300) * 0.01, 4.0),
            ellipse_axis_a=0.022, ellipse_axis_b=0.014,
            plane_axis_u=(1.0, 0.0, 0.0), plane_axis_v=(0.0, 0.0, 1.0),
            start_hold_duration=0.5, joint_departure_duration=1.0,
            approach_duration=1.0, return_duration=1.0,
            joint_return_duration=1.0, final_hold_duration=0.5,
            safe_departure_mode="auto",
        ),
    }


def validate_robot(output: Path) -> Path:
    robot = load_robot_spec(ROBOT_CONFIG, validate_model=True)
    report = validate_robot_model_contract(robot)
    env = MuJoCoArmEnv(
        str(robot.model_xml),
        n_joints=robot.n_joints,
        frame_skip=robot.frame_skip,
        home_q=robot.home_q,
        ee_site_name=robot.ee_site_name,
        gravity_compensation=robot.gravity_compensation,
        gravity_compensation_zero_indices=robot.gravity_compensation_zero_indices,
        seed=0,
    )
    try:
        env.reset_to_configuration(robot.home_q)
        for _ in range(500):
            env.step(robot.home_q)
        state = env.get_state()
        deviation = float(np.max(np.abs(state[: robot.n_joints] - robot.home_q)))
        if not np.all(np.isfinite(state)) or deviation > 0.01:
            raise RuntimeError(f"UR5e home hold contract failed: deviation={deviation}")
        report["hold_steps"] = 500
        report["hold_max_abs_error_rad"] = deviation
        report["home_tcp_position_m"] = env.get_ee_position().tolist()
    finally:
        env.close()
    path = output / "diagnostics" / "robot_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report)
    return path


def generate_references(output: Path, overwrite: bool, calibration_steps: int = 2400) -> Path:
    robot = load_robot_spec(ROBOT_CONFIG, validate_model=True)
    model = mujoco.MjModel.from_xml_path(str(robot.model_xml))
    records = []
    for label, initial_config in _reference_configs(calibration_steps).items():
        config = ReferenceConfig(**{
            **initial_config.__dict__,
            "ee_site_name": robot.ee_site_name,
            "max_joint_velocity": tuple(float(value) for value in robot.command_velocity_limit),
            "max_joint_acceleration": tuple(
                float(value) for value in robot.command_acceleration_limit
            ),
        })
        directory = output / "references" / label
        path = directory / "reference.npz"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
        bundle = build_reference(
            config,
            model,
            robot.home_q,
            robot.expected_control_dt,
            20,
            16,
            robot_spec=robot,
        )
        saved = save_reference_bundle(bundle, directory)
        records.append({
            "label": label,
            "file": file_identity(saved),
            "validation": bundle.metadata["validation"],
            "config": bundle.metadata["config"],
        })
    path = output / "references" / "manifest.json"
    if path.exists() and overwrite:
        path.unlink()
    write_immutable_json(path, {
        "schema_version": 2,
        "robot_identity": robot.artifact_identity(),
        "references": records,
    })
    return path


def _base_args(checkpoint: Path, normalizer: Path) -> argparse.Namespace:
    args = RUNNER.parse_args([])
    args.robot_config = str(ROBOT_CONFIG)
    args.checkpoint = str(checkpoint)
    args.normalizer = str(normalizer)
    args.model_type = "gru"
    args.history_len = 16
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.horizon = 20
    args.num_samples = 128
    args.cem_iters = 2
    args.rollout_batch_size = 128
    args.replan_interval_steps = 5
    args.mpc_warmup_plans = 1
    args.controller_mode = "mpc"
    args.mpc_policy = "residual"
    args.delay_protocol = "full"
    args.dynamics_backend = "learned"
    args.planner_projection = "on"
    args.planner_projection_backend = "compiled"
    args.planner_projection_strategy = "two_stage"
    args.ik_command_projection = "physical"
    args.exact_task_space_cost = "on"
    args.residual_parameterization = "full"
    args.stage_one_task_space_cost = "off"
    args.stage_one_task_compile = "off"
    args.cem_execute = "lowest_cost"
    args.mpc_preview_nominal_steps = 0
    args.w_task_position = 1.0
    args.w_task_orientation = 0.25
    args.task_position_scale_m = 0.05
    args.task_orientation_scale_rad = float(np.deg2rad(5.0))
    args.uncertainty_mode = "off"
    args.cost_profile = "blackbox"
    args.payload_level = 0
    args.actuator_gain_level = 0
    args.force_pulse_level = 0
    args.observation_noise_level = 0
    args.visualize = False
    args.settle_steps = 50
    return args


def resolved_robot_contract(robot) -> dict[str, Any]:
    """Return the RobotSpec-resolved contract applied by the runtime.

    ``base_run_args`` is intentionally preserved in frozen manifests because it
    participates in historical run fingerprints. It is captured before the
    runner resolves a RobotSpec, so legacy defaults such as an ABB XML path are
    not evidence of the robot model used by a UR5e rollout.
    """
    return {
        **robot.artifact_identity(),
        "robot_config": str(ROBOT_CONFIG.relative_to(ROOT)),
        "command_velocity_limit": [float(value) for value in robot.command_velocity_limit],
        "command_acceleration_limit": [float(value) for value in robot.command_acceleration_limit],
        "state_velocity_limit": [float(value) for value in robot.state_velocity_limit],
        "residual_max": [float(value) for value in robot.residual_max],
        "servo_scale": [float(value) for value in robot.servo_scale],
    }


def formal_case_delay_steps(manifest: dict[str, Any]) -> dict[str, Any]:
    """Summarize the formal per-method delays recorded in the main index."""
    delay = int(manifest["delay_calibration"]["anticipation_delay_steps"])
    return {
        "ProjectedDirectIK": 0,
        "IdealZeroDelay": 0,
        "NaiveDelayed": delay,
        "FullVirtual": delay,
        "ThreadedAsync": delay,
        "source": "runs/indexes/main.json",
    }


def upgrade_manifest_payload(manifest: dict[str, Any], robot=None) -> dict[str, Any]:
    """Add RobotSpec resolution metadata without changing fingerprints."""
    if manifest.get("kind") != "ur5e_nominal_portability":
        raise ValueError("Manifest is not a UR5e nominal-portability manifest")
    if "base_run_args" not in manifest:
        raise ValueError("Legacy base_run_args is required for fingerprint compatibility")
    resolved_robot = robot or load_robot_spec(ROBOT_CONFIG, validate_model=True)
    if manifest.get("robot_identity") != resolved_robot.artifact_identity():
        raise ValueError("Manifest robot identity does not match the resolved UR5e RobotSpec")
    upgraded = deepcopy(manifest)
    upgraded["schema_version"] = 2
    upgraded["base_run_args_semantics"] = (
        "Workflow base arguments captured before RobotSpec resolution. Retained "
        "verbatim for frozen run-fingerprint compatibility; these fields are not "
        "the executed robot contract."
    )
    upgraded["resolved_robot_contract"] = resolved_robot_contract(resolved_robot)
    upgraded["formal_case_delay_steps"] = formal_case_delay_steps(upgraded)
    return upgraded


def upgrade_manifest(manifest_path: Path) -> Path:
    """Upgrade a frozen manifest's explanatory schema without rerunning control."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    upgraded = upgrade_manifest_payload(manifest)
    if manifest.get("schema_version") == 2:
        if manifest != upgraded:
            raise ValueError("Existing schema-v2 manifest conflicts with the current UR5e contract")
        return manifest_path
    if manifest.get("schema_version") != 1:
        raise ValueError("Only schema-v1 UR5e manifests can be upgraded")
    write_json(manifest_path, upgraded)
    return manifest_path


def validate_model(
    output: Path,
    checkpoint: Path,
    normalizer: Path,
    num_rollouts: int,
    rollout_len: int,
) -> Path:
    destination = output / "diagnostics" / "gru_validation"
    command = [
        sys.executable,
        str(ROOT / "dynamics_modeling" / "scripts" / "eval_dynamics.py"),
        "--robot_config", str(ROBOT_CONFIG),
        "--checkpoint", str(checkpoint),
        "--normalizer", str(normalizer),
        "--model_type", "gru",
        "--history_len", "16",
        "--num_rollouts", str(num_rollouts),
        "--rollout_len", str(rollout_len),
        "--horizons", "1,5,10,20",
        "--teacher_forcing",
        "--seed", "20260730",
        "--action_std_groups", (
            "0.5:10,0.8:10" if num_rollouts == 20 else f"0.5:{num_rollouts}"
        ),
        "--save_dir", str(destination),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    manifest = {
        "schema_version": 1,
        "robot_identity": load_robot_spec(ROBOT_CONFIG).artifact_identity(),
        "checkpoint": file_identity(checkpoint),
        "normalizer": file_identity(normalizer),
        "horizons": [1, 5, 10, 20],
        "num_rollouts": num_rollouts,
        "rollout_len": rollout_len,
        "seed": 20260730,
        "formal_control_data_excluded": True,
    }
    path = destination / "validation_manifest.json"
    write_json(path, manifest)
    return path


def calibrate_delay(
    output: Path,
    checkpoint: Path,
    normalizer: Path,
    samples: int,
    guard_ms: float,
    provisional: int,
    smoke: bool,
) -> Path:
    reference = output / "references" / "delay_calibration" / "reference.npz"
    if not reference.is_file():
        raise FileNotFoundError("Run generate-references first")
    args = _base_args(checkpoint, normalizer)
    args.multirate_mode = "threaded_asap"
    args.anticipation_delay_steps = provisional
    args.planner_guard_ms = 0.0
    args.reference_mode = "task"
    args.reference_file = str(reference)
    args.max_execution_steps = 40 if smoke else None
    if smoke:
        args.horizon = 3
        # The runner validates replan_interval_steps <= horizon even though
        # threaded ASAP launches plans whenever the worker is ready.  Keep the
        # reduced smoke configuration internally consistent with that contract.
        args.replan_interval_steps = 1
        args.num_samples = 8
        args.cem_iters = 1
    target = min(samples, 3) if smoke else samples
    collected: list[float] = []
    episode = 0
    while len(collected) < target:
        args.seed = episode
        arrays = RUNNER.run_closed_loop_mpc(deepcopy(args))["arrays"]
        values = np.asarray(arrays.get("planner_end_to_end_latency_s", []), dtype=np.float64)
        collected.extend(values[np.isfinite(values)].tolist())
        episode += 1
        if episode >= 5 and not collected:
            raise RuntimeError("Delay calibration produced no end-to-end planner samples")
    values = np.asarray(collected[:target], dtype=np.float64)
    p95 = float(np.percentile(values, 95))
    delay = int(math.ceil((p95 + guard_ms / 1000.0) / 0.01))
    path = output / "calibration" / ("delay_smoke.json" if smoke else "delay.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(path, {
        "schema_version": 1,
        "definition": "ceil((P95(snapshot_to_publication)+guard)/control_dt)",
        "samples": values.tolist(),
        "p50_s": float(np.percentile(values, 50)),
        "p95_s": p95,
        "guard_ms": guard_ms,
        "control_dt_s": 0.01,
        "anticipation_delay_steps": delay,
        "checkpoint": file_identity(checkpoint),
        "normalizer": file_identity(normalizer),
        "calibration_reference": file_identity(reference),
    })
    return path


def _delay(output: Path, smoke_ok: bool = False) -> int:
    candidates = [output / "calibration" / "delay.json"]
    if smoke_ok:
        candidates.append(output / "calibration" / "delay_smoke.json")
    for path in candidates:
        if path.is_file():
            return int(json.loads(path.read_text(encoding="utf-8"))["anticipation_delay_steps"])
    raise FileNotFoundError("Run calibrate-delay first")


def _case_args(
    base: argparse.Namespace,
    *,
    label: str,
    trajectory: str,
    seed: int,
    delay: int,
) -> argparse.Namespace:
    args = deepcopy(base)
    args.seed = seed
    args.reference_mode = "task"
    args.controller_mode = "mpc"
    args.multirate_mode = "virtual_asap"
    args.delay_protocol = "full"
    args.anticipation_delay_steps = delay
    if label == "ProjectedDirectIK":
        args.controller_mode = "ik_direct"
        args.checkpoint = None
        args.normalizer = None
        args.multirate_mode = "synchronous"
        args.exact_task_space_cost = "off"
        args.anticipation_delay_steps = 0
    elif label == "IdealZeroDelay":
        args.anticipation_delay_steps = 0
    elif label == "NaiveDelayed":
        args.delay_protocol = "naive_delayed"
    elif label == "ThreadedAsync":
        args.multirate_mode = "threaded_asap"
    return args


def preflight(output: Path, checkpoint: Path, normalizer: Path, max_steps: int) -> Path:
    validate_robot(output)
    delay = _delay(output, smoke_ok=True)
    reference = output / "references" / "circle" / "reference.npz"
    if not reference.is_file():
        raise FileNotFoundError("Run generate-references first")
    base = _base_args(checkpoint, normalizer)
    entries = []
    for label in ("ProjectedDirectIK", *MPC_LABELS):
        args = _case_args(
            base,
            label=label,
            trajectory="circle",
            seed=0,
            delay=delay,
        )
        args.reference_file = str(reference)
        args.max_execution_steps = max_steps
        args.settle_steps = 2
        if label != "ProjectedDirectIK":
            args.horizon = 3
            args.num_samples = 8
            args.cem_iters = 1
            args.replan_interval_steps = 1
        result = RUNNER.run_closed_loop_mpc(args)
        arrays = result["arrays"]
        for key in (
            "actual_states",
            "actuator_q_ref",
            "command_velocity_violation_flags",
            "command_acceleration_violation_flags",
            "joint_limit_violation_flags",
        ):
            if key not in arrays:
                raise RuntimeError(f"Preflight result is missing {key}")
        if not np.all(np.isfinite(arrays["actual_states"])):
            raise RuntimeError(f"{label} preflight produced non-finite states")
        if any(np.any(arrays[key]) for key in (
            "command_velocity_violation_flags",
            "command_acceleration_violation_flags",
            "joint_limit_violation_flags",
        )):
            raise RuntimeError(f"{label} preflight produced a constraint violation")
        run_dir = output / "preflight" / label
        save_mpc_run(run_dir, arrays, result["rows"], result.get("planner_events"))
        entries.append({"label": label, "run_dir": str(run_dir)})
    path = output / "preflight" / "report.json"
    write_json(path, {"status": "passed", "delay_steps": delay, "entries": entries})
    return path


def build_manifest(
    output: Path,
    checkpoint: Path,
    normalizer: Path,
    dataset_manifest: Path,
    allow_dirty: bool,
) -> Path:
    environment = environment_snapshot(ROOT)
    if environment["git_dirty"] and not allow_dirty:
        raise RuntimeError("Formal UR5e manifest requires a clean worktree")
    robot = load_robot_spec(ROBOT_CONFIG, validate_model=True)
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    if dataset_payload.get("robot_identity") != robot.artifact_identity():
        raise ValueError("Dataset manifest robot identity mismatch")
    reference_manifest = output / "references" / "manifest.json"
    delay_path = output / "calibration" / "delay.json"
    if not reference_manifest.is_file() or not delay_path.is_file():
        raise FileNotFoundError("References and formal delay calibration must exist")
    base = _base_args(checkpoint, normalizer)
    payload = {
        "schema_version": 2,
        "kind": "ur5e_nominal_portability",
        "robot_identity": robot.artifact_identity(),
        "environment": environment,
        "checkpoint": file_identity(checkpoint),
        "normalizer": file_identity(normalizer),
        "dataset_manifest": file_identity(dataset_manifest),
        "references_manifest": file_identity(reference_manifest),
        "references": {
            item["label"]: item["file"]
            for item in json.loads(reference_manifest.read_text(encoding="utf-8"))["references"]
            if item["label"] in TRAJECTORIES
        },
        "delay_calibration": json.loads(delay_path.read_text(encoding="utf-8")),
        "paired_cem_seeds": [0, 1, 2, 3, 4],
        "trajectories": list(TRAJECTORIES),
        "methods": ["ProjectedDirectIK", *MPC_LABELS],
        "base_run_args": {
            key: value for key, value in vars(base).items()
            if key not in {"checkpoint", "normalizer", "save_dir", "seed", "_robot_spec"}
        },
        "base_run_args_semantics": (
            "Workflow base arguments captured before RobotSpec resolution. Retained "
            "verbatim for frozen run-fingerprint compatibility; these fields are not "
            "the executed robot contract."
        ),
        "resolved_robot_contract": resolved_robot_contract(robot),
        "claim_scope": (
            "Nominal architecture portability on a second 6-DoF position-controlled "
            "MuJoCo manipulator; not learned-model transfer."
        ),
    }
    payload["formal_case_delay_steps"] = formal_case_delay_steps(payload)
    path = output / "manifests" / "paper.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(path, payload)
    return path


def suite_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    delay = int(manifest["delay_calibration"]["anticipation_delay_steps"])
    for trajectory in manifest["trajectories"]:
        cases.append({
            "label": "ProjectedDirectIK", "trajectory": trajectory,
            "seed": 0, "delay_steps": 0,
        })
        for seed in manifest["paired_cem_seeds"]:
            for label in MPC_LABELS:
                cases.append({
                    "label": label, "trajectory": trajectory,
                    "seed": int(seed),
                    "delay_steps": 0 if label == "IdealZeroDelay" else delay,
                })
    return cases


def _case_fingerprint(manifest: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    return run_fingerprint({
        "kind": "ur5e_nominal_portability",
        "case": case,
        "robot_identity": manifest["robot_identity"],
        "checkpoint": manifest["checkpoint"],
        "normalizer": manifest["normalizer"],
        "reference": manifest["references"][str(case["trajectory"])],
        "base_run_args": manifest["base_run_args"],
    })


def _identity_matches(identity: dict[str, Any]) -> bool:
    path = identity.get("path")
    return isinstance(path, str) and file_identity(Path(path)) == identity


def audit_freeze(output: Path, manifest_path: Path) -> Path:
    """Audit the sealed UR5e main matrix without rerunning any controller."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_cases = suite_cases(manifest)
    expected_by_key = {
        (str(case["label"]), str(case["trajectory"]), int(case["seed"])): case
        for case in expected_cases
    }
    index_path = output / "runs" / "indexes" / "main.json"
    entries = json.loads(index_path.read_text(encoding="utf-8")).get("entries", []) if index_path.is_file() else []
    observed_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicate_cases: list[str] = []
    malformed_entries: list[str] = []
    for index, entry in enumerate(entries):
        try:
            key = (str(entry["label"]), str(entry["trajectory"]), int(entry["seed"]))
        except (KeyError, TypeError, ValueError):
            malformed_entries.append(str(index))
            continue
        if key in observed_by_key:
            duplicate_cases.append(":".join(map(str, key)))
        else:
            observed_by_key[key] = entry

    input_identities = {
        "checkpoint": manifest["checkpoint"],
        "normalizer": manifest["normalizer"],
        "dataset_manifest": manifest["dataset_manifest"],
        "references_manifest": manifest["references_manifest"],
        **{f"reference:{name}": identity for name, identity in manifest["references"].items()},
        "delay_checkpoint": manifest["delay_calibration"]["checkpoint"],
        "delay_normalizer": manifest["delay_calibration"]["normalizer"],
        "delay_reference": manifest["delay_calibration"]["calibration_reference"],
    }
    input_hash_mismatches = [name for name, identity in input_identities.items() if not _identity_matches(identity)]

    missing_cases = sorted(set(expected_by_key).difference(observed_by_key))
    unexpected_cases = sorted(set(observed_by_key).difference(expected_by_key))
    missing_files: list[str] = []
    wrong_run_directories: list[str] = []
    fingerprint_mismatches: list[str] = []
    fingerprint_hash_mismatches: list[str] = []
    nonfinite_runs: list[str] = []
    missing_required_rollout_fields: list[str] = []
    constraint_violation_runs: list[str] = []
    threaded_control_ticks = 0
    threaded_planner_publications = 0
    threaded_late_drops = 0
    threaded_deadline_misses = 0
    threaded_expirations = 0
    threaded_fallback_active_steps = 0
    threaded_late_drop_rates: list[float] = []
    threaded_deadline_miss_rates: list[float] = []
    threaded_control_lateness: list[np.ndarray] = []
    threaded_wakeup_lateness: list[np.ndarray] = []
    threaded_e2e_latency: list[np.ndarray] = []
    runs_root = (output / "runs").resolve()
    required_arrays = (
        "actual_states", "failure_flags", "joint_limit_violation_flags",
        "command_velocity_violation_flags", "command_acceleration_violation_flags",
    )
    optional_failure_arrays = ("planner_failure_event", "worker_failure")
    for key, case in expected_by_key.items():
        text_key = ":".join(map(str, key))
        entry = observed_by_key.get(key)
        if entry is None:
            continue
        expected_dir = output / "runs" / key[0] / key[1] / f"seed_{key[2]}"
        run_dir = Path(str(entry.get("run_dir", "")))
        if run_dir.resolve() != expected_dir.resolve() or not run_dir.resolve().is_relative_to(runs_root):
            wrong_run_directories.append(text_key)
            continue
        rollout = run_dir / "rollout.npz"
        summary = run_dir / "run_summary.json"
        fingerprint_path = run_dir / "run_fingerprint.json"
        for path in (rollout, summary, fingerprint_path):
            if not path.is_file():
                missing_files.append(f"{text_key}:{path.name}")
        if not rollout.is_file() or not fingerprint_path.is_file():
            continue
        actual_fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if actual_fingerprint.get("sha256") != canonical_sha256(actual_fingerprint.get("payload", {})):
            fingerprint_hash_mismatches.append(text_key)
        if actual_fingerprint != _case_fingerprint(manifest, case):
            fingerprint_mismatches.append(text_key)
        with np.load(rollout, allow_pickle=False) as archive:
            names = (*required_arrays, *optional_failure_arrays)
            if key[0] == "ThreadedAsync":
                names = (*names, "planner_solve_count", "packet_late_dropped",
                         "control_deadline_miss", "packet_expired_event", "fallback_active",
                         "control_lateness_s", "control_wakeup_lateness_s",
                         "planner_end_to_end_latency_s")
            arrays = {
                name: np.asarray(archive[name])
                for name in names
                if name in archive.files
            }
        if "actual_states" not in arrays or not np.all(np.isfinite(arrays["actual_states"])):
            nonfinite_runs.append(text_key)
        if any(name not in arrays for name in required_arrays):
            missing_required_rollout_fields.append(text_key)
        elif any(np.any(arrays[name]) for name in (*required_arrays[1:], *optional_failure_arrays) if name in arrays):
            constraint_violation_runs.append(text_key)
        if key[0] == "ThreadedAsync":
            ticks = int(arrays["actual_states"].shape[0])
            publications = int(np.max(arrays["planner_solve_count"]))
            late_drops = int(np.sum(arrays["packet_late_dropped"]))
            deadline_misses = int(np.sum(arrays["control_deadline_miss"]))
            threaded_control_ticks += ticks
            threaded_planner_publications += publications
            threaded_late_drops += late_drops
            threaded_deadline_misses += deadline_misses
            threaded_expirations += int(np.sum(arrays["packet_expired_event"]))
            threaded_fallback_active_steps += int(np.sum(arrays["fallback_active"]))
            threaded_late_drop_rates.append(late_drops / publications if publications else 0.0)
            threaded_deadline_miss_rates.append(deadline_misses / ticks if ticks else 0.0)
            threaded_control_lateness.append(np.ravel(arrays["control_lateness_s"]))
            threaded_wakeup_lateness.append(np.ravel(arrays["control_wakeup_lateness_s"]))
            threaded_e2e_latency.append(np.ravel(arrays["planner_end_to_end_latency_s"]))

    environment = manifest.get("environment", {})
    git_commit = str(environment.get("git_commit", ""))
    commit_exists = bool(git_commit) and subprocess.run(
        ["git", "cat-file", "-e", f"{git_commit}^{{commit}}"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    def _timing_quantiles(values: list[np.ndarray]) -> dict[str, float]:
        merged = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
        merged = merged[np.isfinite(merged)]
        if not merged.size:
            return {"samples": 0, "p99_s": float("nan"), "max_s": float("nan")}
        return {
            "samples": int(merged.size),
            "p99_s": float(np.percentile(merged, 99)),
            "max_s": float(np.max(merged)),
        }
    status = "passed" if not any((
        duplicate_cases, malformed_entries, missing_cases, unexpected_cases, missing_files,
        wrong_run_directories, fingerprint_mismatches, fingerprint_hash_mismatches,
        nonfinite_runs, missing_required_rollout_fields, constraint_violation_runs,
        input_hash_mismatches,
    )) and environment.get("git_dirty") is False and commit_exists else "failed"
    report = {
        "schema_version": 1,
        "status": status,
        "manifest": file_identity(manifest_path),
        "manifest_git_commit": git_commit,
        "manifest_git_dirty": environment.get("git_dirty"),
        "manifest_commit_exists": commit_exists,
        "expected_cases": len(expected_cases),
        "observed_cases": len(observed_by_key),
        "duplicate_cases": len(duplicate_cases),
        "missing_rollouts": sum(item.endswith(":rollout.npz") for item in missing_files),
        "missing_run_artifacts": missing_files,
        "malformed_index_entries": malformed_entries,
        "missing_cases": [":".join(map(str, key)) for key in missing_cases],
        "unexpected_cases": [":".join(map(str, key)) for key in unexpected_cases],
        "wrong_run_directories": wrong_run_directories,
        "fingerprint_mismatches": len(fingerprint_mismatches),
        "fingerprint_hash_mismatches": len(fingerprint_hash_mismatches),
        "input_hash_mismatches": input_hash_mismatches,
        "nonfinite_runs": len(nonfinite_runs),
        "missing_required_rollout_fields": missing_required_rollout_fields,
        "constraint_violation_runs": len(constraint_violation_runs),
        "threaded_timing": {
            "runs": sum(key[0] == "ThreadedAsync" for key in expected_by_key),
            "control_ticks": threaded_control_ticks,
            "planner_publications": threaded_planner_publications,
            "late_packet_drops": threaded_late_drops,
            "late_drop_rate": threaded_late_drops / threaded_planner_publications if threaded_planner_publications else float("nan"),
            "worst_single_run_late_drop_rate": max(threaded_late_drop_rates, default=float("nan")),
            "control_deadline_misses": threaded_deadline_misses,
            "control_deadline_miss_rate": threaded_deadline_misses / threaded_control_ticks if threaded_control_ticks else float("nan"),
            "worst_single_run_deadline_miss_rate": max(threaded_deadline_miss_rates, default=float("nan")),
            "packet_expirations": threaded_expirations,
            "packet_expiration_rate": threaded_expirations / threaded_planner_publications if threaded_planner_publications else float("nan"),
            "fallback_duty_cycle": threaded_fallback_active_steps / threaded_control_ticks if threaded_control_ticks else float("nan"),
            "control_lateness": _timing_quantiles(threaded_control_lateness),
            "wakeup_lateness": _timing_quantiles(threaded_wakeup_lateness),
            "planner_end_to_end_latency": _timing_quantiles(threaded_e2e_latency),
        },
    }
    path = output / "freeze_audit.json"
    write_json(path, report)
    return path


def run_suite(
    output: Path,
    manifest_path: Path,
    resume: bool,
    case_limit: int | None,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"]["path"])
    normalizer = Path(manifest["normalizer"]["path"])
    base = _base_args(checkpoint, normalizer)
    cases = suite_cases(manifest)
    if case_limit is not None:
        cases = cases[:case_limit]
    entries = []
    for case in cases:
        label = case["label"]
        trajectory = case["trajectory"]
        seed = int(case["seed"])
        run_dir = output / "runs" / label / trajectory / f"seed_{seed}"
        fingerprint = _case_fingerprint(manifest, case)
        if not (resume and load_completed_rollout(run_dir, fingerprint) is not None):
            args = _case_args(
                base,
                label=label,
                trajectory=trajectory,
                seed=seed,
                delay=int(case["delay_steps"]),
            )
            args.reference_file = manifest["references"][trajectory]["path"]
            result = RUNNER.run_closed_loop_mpc(args)
            save_mpc_run(run_dir, result["arrays"], result["rows"], result.get("planner_events"))
            write_json(run_dir / "run_fingerprint.json", fingerprint)
        entries.append({**case, "run_dir": str(run_dir)})
    path = output / "runs" / "indexes" / "main.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"suite": "main", "entries": entries})
    return path


def summarize(output: Path, bootstrap_samples: int) -> Path:
    index_path = output / "runs" / "indexes" / "main.json"
    entries = json.loads(index_path.read_text(encoding="utf-8"))["entries"]
    rows = []
    for entry in entries:
        run_dir = Path(entry["run_dir"])
        with np.load(run_dir / "rollout.npz", allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        event_path = run_dir / "planner_events.jsonl"
        events = (
            [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
            if event_path.is_file() else []
        )
        row = summarize_arrays(entry["label"], arrays, events)
        row.update({
            "label": entry["label"],
            "trajectory": entry["trajectory"],
            "seed": int(entry["seed"]),
            "case_id": f"{entry['trajectory']}:{entry['seed']}",
        })
        rows.append(row)
    path = output / "summaries" / "main.csv"
    write_csv(path, rows)
    write_csv(
        output / "summaries" / "main_aggregate.csv",
        aggregate_rows(rows, ("label", "trajectory")),
    )
    comparisons = {
        "FullVirtual_minus_NaiveDelayed": paired_bootstrap_rows(
            rows, left="NaiveDelayed", right="FullVirtual",
            metrics=("tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad", "failure_rate"),
            samples=bootstrap_samples, seed=20260722,
        ),
        "ThreadedAsync_minus_FullVirtual": paired_bootstrap_rows(
            rows, left="FullVirtual", right="ThreadedAsync",
            metrics=("tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad", "failure_rate"),
            samples=bootstrap_samples, seed=20260724,
        ),
    }
    write_json(output / "summaries" / "main_paired_bootstrap.json", comparisons)
    worst = sorted(
        (
            row for row in rows
            if row["label"] in {"NaiveDelayed", "FullVirtual", "ThreadedAsync"}
        ),
        key=lambda row: float(row["tcp_rmse_m"]),
        reverse=True,
    )
    write_json(output / "summaries" / "worst_seed.json", worst[0] if worst else {})
    return path


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    output = resolve(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "validate-robot":
        print(validate_robot(output))
    elif args.command == "generate-references":
        print(generate_references(output, args.overwrite, args.calibration_steps))
    elif args.command == "validate-model":
        checkpoint, normalizer = _require_artifacts(args)
        print(validate_model(output, checkpoint, normalizer, args.num_rollouts, args.rollout_len))
    elif args.command == "reanalyze-model-validation":
        print(reanalyze(
            resolve(args.input_dir),
            resolve(args.output_dir),
            ROBOT_CONFIG,
            args.horizons,
            args.window_stride,
            args.overwrite,
        ))
    elif args.command == "calibrate-delay":
        checkpoint, normalizer = _require_artifacts(args)
        print(calibrate_delay(
            output, checkpoint, normalizer, args.samples, args.guard_ms,
            args.provisional_delay, args.smoke,
        ))
    elif args.command == "preflight":
        checkpoint, normalizer = _require_artifacts(args)
        print(preflight(output, checkpoint, normalizer, args.max_steps))
    elif args.command == "build-manifest":
        checkpoint, normalizer = _require_artifacts(args)
        if not args.dataset_manifest:
            raise ValueError("--dataset-manifest is required")
        print(build_manifest(
            output, checkpoint, normalizer, resolve(args.dataset_manifest), args.allow_dirty,
        ))
    elif args.command == "upgrade-manifest":
        manifest = resolve(args.manifest or output / "manifests" / "paper.json")
        print(upgrade_manifest(manifest))
    elif args.command == "run":
        manifest = resolve(args.manifest or output / "manifests" / "paper.json")
        print(run_suite(output, manifest, args.resume, args.case_limit))
    elif args.command == "summarize":
        print(summarize(output, args.bootstrap_samples))
    elif args.command == "audit-freeze":
        manifest = resolve(args.manifest or output / "manifests" / "paper.json")
        print(audit_freeze(output, manifest))


if __name__ == "__main__":
    main()
