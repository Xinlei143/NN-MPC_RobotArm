#!/usr/bin/env python3
"""Short, staged SO101 real-robot counterfactual micro-test.

This is deliberately not a full tracking-only MPC run.  For each selected
anchor it starts from home, plays the frozen Direct reference up to that
anchor, injects exactly one candidate command (Direct, preview_6, or a
tracking-only CEM command), then returns to Direct for a three-tick real
response measurement.  The hardware runner retains its thermal, voltage,
timing, command-projector, and transmission safety gates.

The first stage is intentionally conservative: tracking-only CEM searches
with a 0.5 degree residual cap and only one CEM-selected command is applied.
Use --dry-run to validate the frozen reference, artifact identities, and
anchor plan without opening the hardware backend.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dynamics_modeling.neural_dynamics.rollout import load_dynamics_bundle
from mpc.analytical_candidates import build_preview_residual_candidates
from mpc.cem_controller import CEMMPCConfig, CEMMPCController
from mpc.cost_functions import JointSpaceCostConfig
from mpc.executable_rollout import ExecutableRolloutEngine
from mpc.history import history_tokens
from mpc.planner_rollout import LearnedDynamicsPlanner, PlannerRolloutConfig
from mpc.robot_config import load_robot_spec
from robot_runtime.artifacts import verify_real_artifact_identity
from robot_runtime.config import load_hardware_config
from robot_runtime.factory import make_so101_backend
from robot_runtime.runner import PlannerCommand, RealControlMode, RealTimeRunner
from scripts.candidate_cost_audit import (
    _make_executable_spec,
    _reference_calibration,
    _resolve_path,
)
from scripts.run_real_direct_control import JointFilePlayer, find_reference_manifest_entry


JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
CANDIDATES = ("Direct", "preview_6", "CEM_selected")
DEFAULT_ANCHORS = (145, 193, 243, 293, 343, 401, 493, 543, 593, 672, 743, 793, 843, 893, 943)


def _load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"q_des", "dq_des", "ddq_des", "execution_steps"}
        missing = required.difference(archive.files)
        if missing:
            raise KeyError(f"reference {path} is missing {sorted(missing)}")
        q_des = np.asarray(archive["q_des"], dtype=np.float32)
        dq_des = np.asarray(archive["dq_des"], dtype=np.float32)
        ddq_des = np.asarray(archive["ddq_des"], dtype=np.float32)
        execution_steps = int(np.asarray(archive["execution_steps"]).item())
    if q_des.shape != dq_des.shape or q_des.shape != ddq_des.shape or q_des.ndim != 2 or q_des.shape[1] != 5:
        raise ValueError(f"reference arrays must all have shape [N,5], got {q_des.shape}, {dq_des.shape}, {ddq_des.shape}")
    if execution_steps <= 0 or execution_steps > len(q_des):
        raise ValueError(f"invalid execution_steps={execution_steps} for reference length {len(q_des)}")
    if not np.all(np.isfinite(q_des)) or not np.all(np.isfinite(dq_des)) or not np.all(np.isfinite(ddq_des)):
        raise ValueError("reference contains non-finite values")
    return q_des, dq_des, ddq_des, execution_steps


def _parse_anchors(value: str | None) -> list[int]:
    if value is None:
        return list(DEFAULT_ANCHORS)
    try:
        anchors = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--anchors must be comma-separated integer ticks") from exc
    if not anchors:
        raise ValueError("--anchors must contain at least one tick")
    return list(dict.fromkeys(anchors))


def _anchor_metadata(anchors: list[int], dq_des: np.ndarray) -> list[dict[str, Any]]:
    metadata = []
    for anchor in anchors:
        dq = dq_des[anchor]
        metadata.append(
            {
                "anchor": int(anchor),
                "dq_des": dq.astype(float).tolist(),
                "shoulder_pan_direction": "positive" if dq[0] > 1e-4 else "negative" if dq[0] < -1e-4 else "turn/hold",
                "elbow_direction": "positive" if dq[2] > 1e-4 else "negative" if dq[2] < -1e-4 else "turn/hold",
                "speed_norm": float(np.linalg.norm(dq)),
            }
        )
    return metadata


def _validate_reference_and_anchors(
    reference: np.ndarray,
    execution_steps: int,
    anchors: list[int],
    *,
    horizon: int,
    preview_steps: int,
    probe_horizon: int,
) -> None:
    max_required = max(preview_steps, probe_horizon) + 1
    for anchor in anchors:
        if anchor < 1 or anchor + max_required >= execution_steps:
            raise ValueError(
                f"anchor {anchor} is outside the safe executable window "
                f"[1,{execution_steps - max_required - 1}]"
            )
        if anchor + horizon >= len(reference):
            raise ValueError(f"anchor {anchor} lacks CEM horizon {horizon}")


def _make_tracking_planner(
    *,
    states: np.ndarray,
    commands: np.ndarray,
    current_state: np.ndarray,
    command_state: Any,
    reference: np.ndarray,
    dq_des: np.ndarray,
    anchor: int,
    bundle: Any,
    spec: Any,
    engine: ExecutableRolloutEngine,
    hardware: Any,
    robot: Any,
    residual_cap: np.ndarray,
    horizon: int,
    num_samples: int,
    cem_iters: int,
    seed: int,
    device: torch.device,
) -> tuple[LearnedDynamicsPlanner, CEMMPCController]:
    if states.ndim != 2 or commands.ndim != 2 or len(states) != len(commands):
        raise ValueError(f"live history must have equal [T,D] arrays, got {states.shape}, {commands.shape}")
    initial_history = torch.as_tensor(
        history_tokens(states, commands, bundle.history_len), dtype=torch.float32, device=device
    ).unsqueeze(0)
    previous_q_ref = np.asarray(command_state.previous_transmitted_q_ref, dtype=np.float64)
    previous_velocity = np.asarray(command_state.previous_command_velocity, dtype=np.float64)
    nominal = reference[anchor : anchor + horizon]
    q_target = reference[anchor + 1 : anchor + 1 + horizon]
    dq_target = dq_des[anchor + 1 : anchor + 1 + horizon]
    if nominal.shape != (horizon, 5) or q_target.shape != (horizon, 5) or dq_target.shape != (horizon, 5):
        raise ValueError(f"CEM reference window is invalid at anchor {anchor}")
    calibration = _reference_calibration(
        reference,
        dq_des,
        np.gradient(dq_des, hardware.control_dt, axis=0),
        np.asarray(hardware.command_velocity_limit, dtype=np.float32),
        np.asarray(hardware.command_acceleration_limit, dtype=np.float32),
    )
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    cost = JointSpaceCostConfig(
        cost_mode="residual",
        w_q=1.0,
        w_dq=0.0,
        w_residual=0.0,
        w_servo=0.0,
        w_residual_velocity=0.0,
        w_residual_acceleration=0.0,
        w_first=0.0,
        w_terminal=0.0,
        w_joint_limit=0.0,
        w_dq_limit=0.0,
        q_tracking_scale=tensor(calibration["q_tracking_scale"]),
        dq_tracking_scale=tensor(calibration["dq_tracking_scale"]),
        residual_scale=tensor(0.5 * residual_cap),
        servo_scale=tensor(robot.servo_scale),
        residual_velocity_scale=tensor(residual_cap / bundle.control_dt),
        residual_acceleration_scale=tensor(residual_cap / bundle.control_dt**2),
        temporal_discount=0.95,
        barrier_max_weight=2.0,
        state_velocity_limit=tensor(robot.state_velocity_limit),
        joint_limit_safe_margin=0.08,
        joint_limit_temp=0.02,
        dq_limit_temp=0.1,
        control_dt=bundle.control_dt,
        velocity_cost_mode="track",
    )
    rollout = PlannerRolloutConfig(
        mpc_policy="residual",
        q_ref_velocity_limit=tensor(hardware.command_velocity_limit),
        q_ref_acceleration_limit=tensor(hardware.command_acceleration_limit),
        residual_max=tensor(residual_cap),
        joint_limit_margin=0.02,
        rollout_batch_size=num_samples,
        project_residual_kinematics=False,
        projection_backend="eager",
        projection_strategy="two_stage",
        residual_cost_semantics="requested",
        residual_feasibility_semantics="finite",
        residual_parameterization="full",
        residual_control_points=None,
    )
    planner = LearnedDynamicsPlanner(
        model=bundle.model,
        normalizer=bundle.normalizer,
        model_type=bundle.model_type,
        state_dim=bundle.state_dim,
        target_mode=bundle.target_mode,
        control_dt=bundle.control_dt,
        initial_history=initial_history,
        q_des=tensor(q_target),
        dq_des=tensor(dq_target),
        nominal_q_ref=tensor(nominal),
        previous_q_ref=tensor(previous_q_ref),
        previous_q_ref_velocity=tensor(previous_velocity),
        previous_residual=tensor(np.zeros(5, dtype=np.float32)),
        previous_residual_velocity=tensor(np.zeros(5, dtype=np.float32)),
        joint_low=tensor(spec.joint_low),
        joint_high=tensor(spec.joint_high),
        cost_config=cost,
        rollout_config=rollout,
        executable_command_spec=spec,
        executable_command_state=command_state,
        executable_rollout_engine=engine,
    )
    controller = CEMMPCController(
        CEMMPCConfig(
            horizon=horizon,
            action_dim=5,
            decision_horizon=horizon,
            num_samples=num_samples,
            num_elites=None,
            elite_ratio=0.08,
            cem_iters=cem_iters,
            init_std=0.5,
            min_std=0.25,
            smoothing_alpha=0.2,
            temporal_noise_alpha=0.8,
            reset_std_each_step=False,
            uniform_sample_ratio=0.15,
            force_baseline_candidate=True,
            seed=seed,
            device=str(device),
            execute="lowest_cost",
            selection_validation="exact_final_pool",
            stage_one_task_mode="off",
        ),
        planner,
        spec.joint_low,
        spec.joint_high,
    )
    return planner, controller


class _OneShotCEMPlanner:
    """Synchronous one-shot planner adapter for RealTimeRunner."""

    def __init__(
        self,
        *,
        anchor: int,
        reference: np.ndarray,
        dq_des: np.ndarray,
        bundle: Any,
        spec: Any,
        engine: ExecutableRolloutEngine,
        hardware: Any,
        robot: Any,
        residual_cap: np.ndarray,
        horizon: int,
        num_samples: int,
        cem_iters: int,
        seed: int,
        device: torch.device,
        original_residual_max: np.ndarray,
    ) -> None:
        self.anchor = int(anchor)
        self.reference = reference
        self.dq_des = dq_des
        self.bundle = bundle
        self.spec = spec
        self.engine = engine
        self.hardware = hardware
        self.robot = robot
        self.residual_cap = residual_cap
        self.horizon = horizon
        self.num_samples = num_samples
        self.cem_iters = cem_iters
        self.seed = seed
        self.device = device
        self.original_residual_max = original_residual_max
        self.packet: PlannerCommand | None = None
        self.consumed = False
        self.diagnostics: dict[str, Any] = {"planner_ran": False}

    def submit(self, tick_index: int, state_timestamp_ns: int, states: np.ndarray, commands: np.ndarray,
               history_generation: int, executable_command_state: object | None = None) -> None:
        if int(tick_index) != self.anchor or self.diagnostics.get("planner_ran"):
            return
        if executable_command_state is None:
            raise RuntimeError("one-shot CEM requires the backend executable command state")
        command_state = executable_command_state
        planner, controller = _make_tracking_planner(
            states=states,
            commands=commands,
            current_state=states[-1],
            command_state=command_state,
            reference=self.reference,
            dq_des=self.dq_des,
            anchor=self.anchor,
            bundle=self.bundle,
            spec=self.spec,
            engine=self.engine,
            hardware=self.hardware,
            robot=self.robot,
            residual_cap=self.residual_cap,
            horizon=self.horizon,
            num_samples=self.num_samples,
            cem_iters=self.cem_iters,
            seed=self.seed,
            device=self.device,
        )
        print(
            f"[microtest] CEM start anchor={self.anchor} "
            f"samples={self.num_samples} iters={self.cem_iters}",
            flush=True,
        )
        result = controller.plan(states[-1], command_state.previous_transmitted_q_ref, warm_start_shift_steps=0)
        if result.failure:
            raise RuntimeError(f"tracking-only CEM failed at anchor {self.anchor}: {result.failure_reason}")
        preview = build_preview_residual_candidates(
            self.reference,
            anchor=self.anchor,
            horizon=self.horizon,
            nominal=self.reference[self.anchor : self.anchor + self.horizon],
            residual_max=self.original_residual_max,
            preview_steps=(6,),
            nominal_preview_steps=0,
        )["preview:6"]
        preview_planner, _unused_controller = _make_tracking_planner(
            states=states,
            commands=commands,
            current_state=states[-1],
            command_state=command_state,
            reference=self.reference,
            dq_des=self.dq_des,
            anchor=self.anchor,
            bundle=self.bundle,
            spec=self.spec,
            engine=self.engine,
            hardware=self.hardware,
            robot=self.robot,
            residual_cap=self.original_residual_max,
            horizon=self.horizon,
            num_samples=self.num_samples,
            cem_iters=self.cem_iters,
            seed=self.seed,
            device=self.device,
        )
        preview_eval = preview_planner.evaluate_exact(
            torch.as_tensor(preview, dtype=torch.float32, device=self.device).unsqueeze(0)
        )
        preview_q_cost = float(preview_eval["cost_terms"]["q_tracking"][0].detach().cpu())
        selected_residual = np.asarray(result.selected_residual_sequence[0], dtype=np.float32)
        selected_q_ref = np.asarray(result.selected_q_ref_sequence[0], dtype=np.float32)
        expected_raw = None
        if result.selected_expected_raw_sequence.shape == result.selected_q_ref_sequence.shape:
            expected_raw = np.asarray(result.selected_expected_raw_sequence[0], dtype=np.int64)
        self.packet = PlannerCommand(
            residual=selected_residual,
            history_generation=int(history_generation),
            activation_tick=self.anchor,
            publication_tick=self.anchor,
            ood_valid=True,
            absolute_q_ref=selected_q_ref,
            expected_raw=expected_raw,
        )
        self.diagnostics = {
            "planner_ran": True,
            "planner_time_s": float(result.planning_time),
            "model_q_tracking_cem": float(result.selected_cost),
            "model_q_tracking_preview6": preview_q_cost,
            "model_cem_beats_preview6": bool(float(result.selected_cost) < preview_q_cost),
            "selection_mode": result.selection_mode,
            "selected_residual": selected_residual.astype(float).tolist(),
            "selected_q_ref": selected_q_ref.astype(float).tolist(),
            "selected_expected_raw": None if expected_raw is None else expected_raw.astype(int).tolist(),
            "candidate_cost_terms": {key: float(value) for key, value in result.cost_terms.items()},
        }
        print(
            f"[microtest] CEM done anchor={self.anchor} "
            f"model_q={float(result.selected_cost):.6g} "
            f"preview6_q={preview_q_cost:.6g} "
            f"planning_s={float(result.planning_time):.3f}",
            flush=True,
        )

    def latest(self) -> PlannerCommand | None:
        if self.consumed:
            return None
        self.consumed = True
        return self.packet

    def clear(self, history_generation: int) -> None:
        if self.packet is None:
            self.consumed = False


def _real_q_cost(records: list[Any], anchor: int, reference: np.ndarray, probe_horizon: int) -> tuple[float, list[float]]:
    states = []
    for tick in range(anchor, anchor + probe_horizon + 1):
        if tick >= len(records):
            break
        if int(records[tick].state.tick_index) == tick:
            states.append(np.asarray(records[tick].state.q_ctrl, dtype=np.float64))
    if len(states) < probe_horizon + 1:
        return float("nan"), []
    errors = [
        float(np.mean(np.square(states[offset] - reference[anchor + offset])))
        for offset in range(1, probe_horizon + 1)
    ]
    return float(np.mean(errors)), errors


def _trial_row(
    *,
    anchor: int,
    candidate: str,
    records: list[Any],
    reference: np.ndarray,
    probe_horizon: int,
    planner_diagnostics: dict[str, Any],
    stop_reason: str | None,
) -> dict[str, Any]:
    if len(records) > anchor:
        anchor_state = np.asarray(records[anchor].state.q_ctrl, dtype=np.float64)
        transmitted = np.asarray(records[anchor].command.transmitted_q_ref, dtype=np.float64)
        requested = np.asarray(records[anchor].command.requested_q_ref, dtype=np.float64)
        projection_flags = list(records[anchor].command.projection_flags)
    else:
        anchor_state = np.full(5, np.nan)
        transmitted = np.full(5, np.nan)
        requested = np.full(5, np.nan)
        projection_flags = []
    real_cost, real_by_step = _real_q_cost(records, anchor, reference, probe_horizon)
    return {
        "anchor": int(anchor),
        "candidate": candidate,
        "real_q_tracking_h3": real_cost,
        "real_q_tracking_by_step": real_by_step,
        "anchor_q_ctrl": anchor_state.tolist(),
        "requested_q_ref": requested.tolist(),
        "transmitted_q_ref": transmitted.tolist(),
        "projection_flags": projection_flags,
        "record_count": len(records),
        "runner_stop_reason": stop_reason,
        **planner_diagnostics,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# SO101 local counterfactual micro-test",
        "",
        f"level: `{report['protocol']['level']}`; residual cap: `{report['protocol']['residual_cap_deg']} deg`; "
        f"intervention ticks: `{report['protocol']['intervention_ticks']}`; probe horizon: `{report['protocol']['probe_horizon']}`",
        f"anchors: `{report['protocol']['anchors']}`",
        "",
        "Real cost is mean squared joint-position error against the original q_des over the future probe horizon.",
        "",
        "## Trial results",
        "",
        "| anchor | candidate | real Cq | model CEM q | model preview6 q | CEM model better | projection flags |",
        "|---:|---|---:|---:|---:|---|---|",
    ]
    for row in report["trials"]:
        lines.append(
            f"| {row['anchor']} | {row['candidate']} | {row['real_q_tracking_h3']:.6g} | "
            f"{row.get('model_q_tracking_cem', float('nan')):.6g} | "
            f"{row.get('model_q_tracking_preview6', float('nan')):.6g} | "
            f"{row.get('model_cem_beats_preview6', '')} | {';'.join(row['projection_flags'])} |"
        )
    lines += [
        "",
        "## Pairwise accuracy",
        "",
        "| comparison | model says CEM better | real says CEM better | accuracy |",
        "|---|---:|---:|---:|",
    ]
    for key, item in report["pairwise_accuracy"].items():
        lines.append(
            f"| {key} | {item['model_better_count']} | {item['real_better_count']} | {item['accuracy']:.6g} |"
        )
    lines += [
        "",
        "The CEM command was generated with tracking-only weights and the staged trust-region cap; "
        "all real commands passed the hardware executable projector and RealTimeRunner safety gates.",
    ]
    return "\n".join(lines) + "\n"


def _pairwise_accuracy(trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_anchor = {}
    for row in trials:
        by_anchor.setdefault(int(row["anchor"]), {})[row["candidate"]] = row
    result: dict[str, Any] = {}
    for alternative in ("preview_6", "Direct"):
        model_flags = []
        real_flags = []
        for candidates in by_anchor.values():
            cem = candidates.get("CEM_selected")
            other = candidates.get(alternative)
            if cem is None or other is None:
                continue
            model_cem = cem.get("model_q_tracking_cem")
            model_other = cem.get("model_q_tracking_preview6") if alternative == "preview_6" else None
            real_cem = cem.get("real_q_tracking_h3")
            real_other = other.get("real_q_tracking_h3")
            if not all(np.isfinite(value) for value in (real_cem, real_other)):
                continue
            if model_cem is not None and model_other is not None and np.isfinite(model_cem) and np.isfinite(model_other):
                model_flags.append(bool(model_cem < model_other))
                real_flags.append(bool(real_cem < real_other))
            elif alternative == "Direct":
                real_flags.append(bool(real_cem < real_other))
        if model_flags:
            correct = [model == real for model, real in zip(model_flags, real_flags)]
            result[f"CEM_vs_{alternative}"] = {
                "model_better_count": int(sum(model_flags)),
                "real_better_count": int(sum(real_flags)),
                "accuracy": float(np.mean(correct)),
                "n": len(correct),
            }
        else:
            result[f"CEM_vs_{alternative}"] = {
                "model_better_count": None,
                "real_better_count": int(sum(real_flags)),
                "accuracy": float("nan"),
                "n": len(real_flags),
            }
    return result


def _write_report_files(output_dir: Path, report: dict[str, Any], stem: str) -> None:
    """Write JSON/CSV/Markdown together so completed trials are recoverable."""
    (output_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    trials = report.get("trials", [])
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({
            key for row in trials for key in row
            if not isinstance(row.get(key), (list, dict))
        })
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in trials:
            writer.writerow({key: row.get(key) for key in fieldnames})
    (output_dir / f"{stem}.md").write_text(_markdown(report), encoding="utf-8")


def _make_report(
    protocol: dict[str, Any],
    anchor_metadata: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    *,
    dry_run: bool,
    partial: bool,
    error: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "protocol": protocol,
        "anchor_metadata": anchor_metadata,
        "trials": trials,
        "pairwise_accuracy": _pairwise_accuracy(trials),
        "dry_run": bool(dry_run),
        "partial": bool(partial),
    }
    if error is not None:
        report["error"] = error
    return report


def _load_partial_trials(output_dir: Path, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    path = output_dir / "microtest_partial.json"
    if not path.is_file():
        raise FileNotFoundError(f"resume requested but partial report is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    saved_protocol = payload.get("protocol", {})
    if saved_protocol.get("anchors") != protocol.get("anchors"):
        raise ValueError("partial report anchors do not match the current --anchors selection")
    trials = payload.get("trials")
    if not isinstance(trials, list):
        raise ValueError(f"partial report has invalid trials: {path}")
    seen: set[tuple[int, str]] = set()
    for row in trials:
        key = (int(row["anchor"]), str(row["candidate"]))
        if key in seen:
            raise ValueError(f"partial report contains duplicate trial: {key}")
        seen.add(key)
    return trials


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hardware-config", default="configs/hardware/so101_follower.local.yaml", type=Path)
    parser.add_argument("--robot-config", default="configs/robots/so101.yaml", type=Path)
    parser.add_argument("--reference-file", required=True, type=Path)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--normalizer", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--anchors", default=None, help="Comma-separated representative ticks")
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="Resume missing trials from output-dir/microtest_partial.json",
    )
    parser.add_argument("--residual-cap-deg", type=float, default=0.5)
    parser.add_argument("--probe-horizon", type=int, default=3)
    parser.add_argument("--intervention-ticks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--cem-iters", type=int, default=2)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--return-duration", type=float, default=2.0)
    parser.add_argument(
        "--home-tolerance-deg",
        type=float,
        default=1.0,
        help="Home convergence tolerance; must remain below the +/-3 deg experiment envelope",
    )
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--operator-supported-shutdown", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.residual_cap_deg <= 0.0 or args.probe_horizon != 3 or args.intervention_ticks != 1:
        raise SystemExit("Level 1 requires --residual-cap-deg > 0, --probe-horizon 3, and --intervention-ticks 1")
    if args.horizon <= 0 or args.num_samples < 3 or args.cem_iters <= 0:
        raise SystemExit("horizon/cem-iters must be positive and num-samples must be at least 3")
    if args.home_tolerance_deg <= 0.0 or args.home_tolerance_deg >= 3.0:
        raise SystemExit("--home-tolerance-deg must be > 0 and < 3 degrees")
    if not args.dry_run and (not args.enable_motion or not args.operator_supported_shutdown):
        raise SystemExit("refusing hardware motion without --enable-motion and --operator-supported-shutdown")

    hardware_path = _resolve_path(args.hardware_config)
    robot_path = _resolve_path(args.robot_config)
    reference_path = _resolve_path(args.reference_file)
    manifest_path = _resolve_path(args.reference_manifest)
    checkpoint = _resolve_path(args.checkpoint)
    normalizer = _resolve_path(args.normalizer)
    hardware = load_hardware_config(hardware_path)
    robot = load_robot_spec(robot_path, validate_model=True)
    if robot.n_joints != 5:
        raise SystemExit("micro-test currently targets the five controlled SO101 joints")
    q_des, dq_des, ddq_des, execution_steps = _load_reference(reference_path)
    find_reference_manifest_entry(manifest_path, reference_path)
    verify_real_artifact_identity(checkpoint, normalizer, hardware.plant_identity())
    anchors = _parse_anchors(args.anchors)
    _validate_reference_and_anchors(
        q_des,
        execution_steps,
        anchors,
        horizon=args.horizon,
        preview_steps=6,
        probe_horizon=args.probe_horizon,
    )
    anchor_metadata = _anchor_metadata(anchors, dq_des)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "level": "L1_tracking_only_cem_0p5deg_one_tick",
        "residual_cap_deg": float(args.residual_cap_deg),
        "intervention_ticks": int(args.intervention_ticks),
        "probe_horizon": int(args.probe_horizon),
        "horizon": int(args.horizon),
        "home_tolerance_deg": float(args.home_tolerance_deg),
        "anchors": anchors,
        "joint_names": list(JOINT_NAMES),
        "reference": str(reference_path),
        "reference_manifest": str(manifest_path),
        "checkpoint": str(checkpoint),
        "device": str(args.device),
        "weights": {
            "w_q": 1.0,
            "w_dq": 0.0,
            "w_residual": 0.0,
            "w_servo": 0.0,
            "w_residual_velocity": 0.0,
            "w_residual_acceleration": 0.0,
            "w_first": 0.0,
            "w_joint_limit": 0.0,
            "w_dq_limit": 0.0,
            "w_terminal": 0.0,
        },
        "physical_safety": {
            "canonical_projector": True,
            "command_velocity_limit": np.asarray(hardware.command_velocity_limit).tolist(),
            "command_acceleration_limit": np.asarray(hardware.command_acceleration_limit).tolist(),
            "hardware_joint_bounds": True,
            "real_time_runner_safety": True,
        },
    }
    trials: list[dict[str, Any]] = []
    if args.resume_partial:
        trials = _load_partial_trials(output_dir, protocol)
        print(f"[microtest] resuming {len(trials)} completed trials", flush=True)
    plan_report = {
        "protocol": protocol,
        "anchor_metadata": anchor_metadata,
        "execution_steps": execution_steps,
        "dry_run": bool(args.dry_run),
        "resume_partial": bool(args.resume_partial),
    }
    (output_dir / "microtest_plan.json").write_text(json.dumps(plan_report, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan_report, indent=2))
        return

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise SystemExit("refusing Level 1 hardware test: requested CUDA device is unavailable")
    if not Path(hardware.port).exists():
        raise SystemExit(f"refusing Level 1 hardware test: serial device is missing: {hardware.port}")

    device = torch.device(args.device)
    bundle = load_dynamics_bundle(
        checkpoint,
        normalizer,
        "gru",
        5,
        device,
        history_len=16,
        expected_robot_spec=robot,
    )
    if abs(bundle.control_dt - hardware.control_dt) > 1e-9:
        raise SystemExit("checkpoint and hardware control_dt differ")
    original_residual_max = np.asarray(robot.residual_max, dtype=np.float32)
    residual_cap = np.full(5, np.deg2rad(args.residual_cap_deg), dtype=np.float32)
    home_tolerance_rad = float(np.deg2rad(args.home_tolerance_deg))
    backend = make_so101_backend(hardware_path)
    total_trials = len(anchors) * len(CANDIDATES)
    completed_keys = {(int(row["anchor"]), str(row["candidate"])) for row in trials}
    try:
        backend.connect()
        print(
            f"[microtest] connected; initial home convergence "
            f"(total trials={total_trials})",
            flush=True,
        )
        backend.startup_to_home(home_tolerance_rad=home_tolerance_rad)
        print("[microtest] initial home settled", flush=True)
        spec = backend.executable_command_spec("hardware")
        engine = ExecutableRolloutEngine(
            model=bundle.model,
            normalizer=bundle.normalizer,
            model_type=bundle.model_type,
            state_dim=bundle.state_dim,
            target_mode=bundle.target_mode,
            control_dt=bundle.control_dt,
            spec=spec,
            backend="cuda_graph",
        )
        player = JointFilePlayer(q_des, config=hardware)
        for anchor in anchors:
            for candidate in CANDIDATES:
                trial_key = (int(anchor), candidate)
                if trial_key in completed_keys:
                    print(
                        f"[microtest] skip completed anchor={anchor} candidate={candidate}",
                        flush=True,
                    )
                    continue
                trial_number = len(trials) + 1
                steps = anchor + args.probe_horizon + 1
                print(
                    f"[microtest] trial {trial_number}/{total_trials} "
                    f"anchor={anchor} candidate={candidate} steps={steps}",
                    flush=True,
                )
                if trials:
                    # ``return_duration`` is the commanded trajectory time,
                    # not a guarantee that the compliant servo has settled.
                    # Use the backend's bounded hold-and-converge procedure
                    # instead of a single immediate readback.
                    print("[microtest] returning home before trial", flush=True)
                    backend.startup_to_home(
                        duration_s=args.return_duration,
                        home_tolerance_rad=home_tolerance_rad,
                    )
                    print("[microtest] home settled", flush=True)
                    backend.reset_estimator_and_history()
                backend.prepare_hardware_motion()
                planner_adapter = None
                if candidate == "CEM_selected":
                    planner_adapter = _OneShotCEMPlanner(
                        anchor=anchor,
                        reference=q_des,
                        dq_des=dq_des,
                        bundle=bundle,
                        spec=spec,
                        engine=engine,
                        hardware=hardware,
                        robot=robot,
                        residual_cap=residual_cap,
                        horizon=args.horizon,
                        num_samples=args.num_samples,
                        cem_iters=args.cem_iters,
                        seed=args.seed,
                        device=device,
                        original_residual_max=original_residual_max,
                    )
                    nominal = lambda tick, state: q_des[min(int(tick), len(q_des) - 1)]
                    mode = RealControlMode.ACTIVE_MPC
                    runner = RealTimeRunner(
                        backend,
                        nominal,
                        mode=mode,
                        planner=planner_adapter,
                        history_len=bundle.history_len,
                        residual_limit_rad=float(np.max(residual_cap)),
                        command_envelope="hardware",
                        mpc_reference_prevalidated=True,
                        active_start_tick=anchor,
                    )
                else:
                    planner_adapter = None
                    def nominal(tick: int, state: Any, candidate_name: str = candidate) -> np.ndarray:
                        if int(tick) == anchor and candidate_name == "preview_6":
                            return q_des[anchor + 6].copy()
                        return q_des[min(int(tick), len(q_des) - 1)].copy()
                    runner = RealTimeRunner(
                        backend,
                        nominal,
                        mode=RealControlMode.DIRECT,
                        planner=None,
                        history_len=bundle.history_len,
                        residual_limit_rad=float(np.max(residual_cap)),
                        command_envelope="hardware",
                    )
                print(f"[microtest] replaying reference to anchor={anchor}", flush=True)
                records = runner.run(steps)
                diagnostics = {} if planner_adapter is None else dict(planner_adapter.diagnostics)
                row = _trial_row(
                    anchor=anchor,
                    candidate=candidate,
                    records=records,
                    reference=q_des,
                    probe_horizon=args.probe_horizon,
                    planner_diagnostics=diagnostics,
                    stop_reason=runner.last_stop_reason,
                )
                trials.append(row)
                completed_keys.add(trial_key)
                _write_report_files(
                    output_dir,
                    _make_report(
                        protocol,
                        anchor_metadata,
                        trials,
                        dry_run=False,
                        partial=True,
                    ),
                    "microtest_partial",
                )
                print(
                    f"[microtest] done trial {trial_number}/{total_trials} "
                    f"anchor={anchor} candidate={candidate} "
                    f"real_Cq_h3={row['real_q_tracking_h3']:.6g}",
                    flush=True,
                )
                if runner.last_stop_reason is not None:
                    raise RuntimeError(f"RealTimeRunner stopped during {candidate} at anchor {anchor}: {runner.last_stop_reason}")
        # Keep the final pose inside the same verified home convergence
        # contract as startup.  A compliant servo can still be settling when
        # the commanded return duration ends, so one immediate read is not a
        # valid settle test.
        print("[microtest] all trials complete; final return home", flush=True)
        backend.startup_to_home(
            duration_s=args.return_duration,
            home_tolerance_rad=home_tolerance_rad,
        )
        print("[microtest] final home settled", flush=True)
    except Exception as exc:
        _write_report_files(
            output_dir,
            _make_report(
                protocol,
                anchor_metadata,
                trials,
                dry_run=False,
                partial=True,
                error=str(exc),
            ),
            "microtest_partial",
        )
        print(
            f"[microtest] partial report saved: {output_dir / 'microtest_partial.json'}",
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        # If a trial or a convergence check raises, make a bounded best-effort
        # return to home before disabling torque.  Preserve the original
        # exception if this recovery also fails.
        try:
            if getattr(backend, "_connected", False):
                print("[microtest] cleanup return home", flush=True)
                backend.startup_to_home(
                    duration_s=args.return_duration,
                    home_tolerance_rad=home_tolerance_rad,
                )
        except Exception as cleanup_exc:
            print(f"WARNING: final return-to-home failed: {cleanup_exc}", file=sys.stderr)
        finally:
            backend.close()

    report = _make_report(
        protocol,
        anchor_metadata,
        trials,
        dry_run=False,
        partial=False,
    )
    _write_report_files(output_dir, report, "microtest")
    print(_markdown(report))


if __name__ == "__main__":
    main()
