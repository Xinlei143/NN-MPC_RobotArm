from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import torch

from learned_dynamics.mujoco_env import MuJoCoArmEnv
from learned_dynamics.normalization import StandardNormalizer
from learned_dynamics.paths import DEFAULT_MODEL_XML, resolve_project_path
from learned_dynamics.train_utils import build_model, load_checkpoint, set_seed


def load_eval_dynamics_module() -> Any:
    spec = importlib.util.spec_from_file_location("local_eval_dynamics", ROOT / "scripts" / "eval_dynamics.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load scripts/eval_dynamics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVAL_DYNAMICS = load_eval_dynamics_module()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_run = "outputs/checkpoints_transformer/transformer_20260604_212044"
    parser = argparse.ArgumentParser(
        description="Diagnose whether MuJoCo rollout data quality causes learned-dynamics error spikes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_xml", default=DEFAULT_MODEL_XML, type=str)
    parser.add_argument("--checkpoint", default=f"{default_run}/best_model.pt", type=str)
    parser.add_argument("--normalizer", default=f"{default_run}/normalizer.pt", type=str)
    parser.add_argument("--model_type", choices=["mlp", "gru", "transformer"], default="transformer")
    parser.add_argument("--n_joints", default=6, type=int)
    parser.add_argument("--history_len", default=16, type=int)
    parser.add_argument("--rollout_len", default=200, type=int)
    parser.add_argument("--num_rollouts", default=10, type=int)
    parser.add_argument("--warmup_steps", default=50, type=int)
    parser.add_argument("--action_std", default="0.3", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save_dir", default="outputs/diagnostics/transformer_20260604_212044", type=str)
    parser.add_argument("--dataset_paths", nargs="*", default=["outputs/datasets/*.npz"], type=str)
    parser.add_argument("--max_dataset_samples", default=None, type=int)
    parser.add_argument("--top_k_spikes", default=5, type=int)
    parser.add_argument("--limit_margin", default=0.05, type=float)
    parser.add_argument("--near_limit_q1", default=1.85, type=float)
    parser.add_argument("--action_jump_percentile", default=95.0, type=float)
    parser.add_argument("--high_speed_percentile", default=95.0, type=float)
    parser.add_argument("--singularity_cond_threshold", default=200.0, type=float)
    parser.add_argument("--singularity_sigma_threshold", default=1e-2, type=float)
    return parser.parse_args(argv)


def state_labels(n_joints: int) -> list[str]:
    return [f"q{idx}" for idx in range(n_joints)] + [f"dq{idx}" for idx in range(n_joints)]


def expand_dataset_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern) if not Path(pattern).is_absolute() else Path("/").glob(pattern[1:]))
        if matches:
            paths.extend(matches)
        else:
            paths.append(resolve_project_path(pattern, ROOT))
    return sorted(dict.fromkeys(paths))


def as_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: as_builtin(row.get(key, "")) for key in fieldnames})


def controlled_joint_ranges(env: MuJoCoArmEnv, n_joints: int) -> np.ndarray:
    joint_ids: list[int] = []
    if env.model.actuator_trnid.shape[0] >= n_joints:
        joint_ids = [int(item) for item in env.model.actuator_trnid[:n_joints, 0]]
    if len(joint_ids) != n_joints or any(joint_id < 0 for joint_id in joint_ids):
        joint_ids = list(range(n_joints))
    ranges = np.asarray(env.model.jnt_range[joint_ids], dtype=np.float32)
    if ranges.shape != (n_joints, 2):
        raise ValueError(f"Expected joint ranges with shape ({n_joints}, 2), got {ranges.shape}")
    return ranges


def summarize_joint_limits(
    states: np.ndarray,
    joint_ranges: np.ndarray,
    n_joints: int,
    limit_margin: float,
    near_limit_overrides: dict[int, float],
) -> dict[str, float]:
    q = np.asarray(states[:, :n_joints], dtype=np.float64)
    lower = np.asarray(joint_ranges[:n_joints, 0], dtype=np.float64)
    upper = np.asarray(joint_ranges[:n_joints, 1], dtype=np.float64)
    lower_distance = q - lower
    upper_distance = upper - q
    limit_distance = np.minimum(lower_distance, upper_distance)
    summary: dict[str, float] = {
        "min_limit_distance": float(np.min(limit_distance)),
        "near_any_limit_rate": float(np.mean(np.any(limit_distance <= limit_margin, axis=1))),
        "any_limit_violation_rate": float(np.mean(np.any((q < lower) | (q > upper), axis=1))),
    }
    for idx in range(n_joints):
        near_upper = q[:, idx] >= upper[idx] - limit_margin
        if idx in near_limit_overrides:
            near_upper = q[:, idx] >= near_limit_overrides[idx]
        near_lower = q[:, idx] <= lower[idx] + limit_margin
        violated = (q[:, idx] < lower[idx]) | (q[:, idx] > upper[idx])
        summary.update(
            {
                f"q{idx}_min": float(np.min(q[:, idx])),
                f"q{idx}_mean": float(np.mean(q[:, idx])),
                f"q{idx}_max": float(np.max(q[:, idx])),
                f"q{idx}_near_upper_rate": float(np.mean(near_upper)),
                f"q{idx}_near_lower_rate": float(np.mean(near_lower)),
                f"q{idx}_violation_rate": float(np.mean(violated)),
                f"q{idx}_min_limit_distance": float(np.min(limit_distance[:, idx])),
            }
        )
    return summary


def summarize_dataset_file(
    path: Path,
    n_joints: int,
    joint_ranges: np.ndarray,
    limit_margin: float,
    near_limit_overrides: dict[int, float],
    max_dataset_samples: int | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "status": "ok"}
    try:
        with np.load(path) as data:
            states = np.asarray(data["states"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32) if "actions" in data.files else None
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        row.update({"status": "bad_npz", "error": str(exc)})
        return row

    if states.ndim != 2 or states.shape[1] < 2 * n_joints:
        row.update({"status": "bad_shape", "states_shape": str(states.shape)})
        return row
    if max_dataset_samples is not None and states.shape[0] > max_dataset_samples:
        indices = np.linspace(0, states.shape[0] - 1, max_dataset_samples, dtype=np.int64)
        states = states[indices]
        if actions is not None:
            actions = actions[indices]
        row["sampled"] = True
    else:
        row["sampled"] = False
    row["num_samples"] = int(states.shape[0])
    row.update(summarize_joint_limits(states, joint_ranges, n_joints, limit_margin, near_limit_overrides))
    dq = states[:, n_joints : 2 * n_joints]
    for idx in range(n_joints):
        abs_dq = np.abs(dq[:, idx])
        row[f"dq{idx}_abs_p95"] = float(np.percentile(abs_dq, 95))
        row[f"dq{idx}_abs_max"] = float(np.max(abs_dq))
    if actions is not None and actions.ndim == 2 and actions.shape[1] >= n_joints:
        for idx in range(n_joints):
            row[f"action{idx}_min"] = float(np.min(actions[:, idx]))
            row[f"action{idx}_max"] = float(np.max(actions[:, idx]))
    return row


def l2_distribution(l2: np.ndarray) -> dict[str, float]:
    sorted_l2 = np.sort(l2)
    return {
        "l2_mean": float(np.mean(l2)),
        "l2_median": float(np.median(l2)),
        "l2_p90": float(np.percentile(l2, 90)),
        "l2_p95": float(np.percentile(l2, 95)),
        "l2_p99": float(np.percentile(l2, 99)),
        "l2_mean_without_top1": float(np.mean(sorted_l2[:-1])) if len(sorted_l2) > 1 else 0.0,
        "l2_mean_without_top5": float(np.mean(sorted_l2[:-5])) if len(sorted_l2) > 5 else 0.0,
    }


def per_step_limit_distance(states: np.ndarray, joint_ranges: np.ndarray, n_joints: int) -> np.ndarray:
    q = np.asarray(states[:, :n_joints], dtype=np.float64)
    lower = np.asarray(joint_ranges[:n_joints, 0], dtype=np.float64)
    upper = np.asarray(joint_ranges[:n_joints, 1], dtype=np.float64)
    return np.minimum(q - lower, upper - q)


def dominant_errors(error: np.ndarray, labels: list[str], top_k: int) -> dict[str, Any]:
    order = np.argsort(np.abs(error))[::-1][:top_k]
    row: dict[str, Any] = {}
    for rank, idx in enumerate(order):
        row[f"dominant_error_{rank}"] = labels[int(idx)]
        row[f"dominant_error_{rank}_value"] = float(error[int(idx)])
    return row


def summarize_rollout_spike(
    rollout_idx: int,
    truth: np.ndarray,
    pred: np.ndarray,
    actions: np.ndarray,
    joint_ranges: np.ndarray,
    labels: list[str],
    jacobian: dict[str, np.ndarray],
    teacher_pred: np.ndarray | None,
    limit_margin: float,
    action_jump_percentile: float,
    high_speed_percentile: float,
    singularity_cond_threshold: float,
    singularity_sigma_threshold: float,
    top_k_errors: int = 5,
) -> dict[str, Any]:
    errors = truth - pred
    l2 = np.linalg.norm(errors, axis=1)
    peak_step = int(np.argmax(l2))
    n_joints = truth.shape[1] // 2
    action_delta = np.zeros(len(actions), dtype=np.float64)
    if len(actions) > 1:
        action_delta[1:] = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    dq_abs = np.max(np.abs(truth[:, n_joints : 2 * n_joints]), axis=1)
    limit_distance = per_step_limit_distance(truth, joint_ranges, n_joints)
    min_limit_distance = np.min(limit_distance, axis=1)
    teacher_l2 = None
    teacher_forcing_bad = False
    if teacher_pred is not None:
        teacher_l2 = np.linalg.norm(truth - teacher_pred, axis=1)
        teacher_forcing_bad = bool(teacher_l2[peak_step] >= np.percentile(teacher_l2, 95))
    near_joint_limit = bool(min_limit_distance[peak_step] <= limit_margin)
    high_speed = bool(dq_abs[peak_step] >= np.percentile(dq_abs, high_speed_percentile))
    action_jump = bool(action_delta[peak_step] >= np.percentile(action_delta, action_jump_percentile))
    sigma_min = float(jacobian["sigma_min"][peak_step])
    condition = float(jacobian["condition"][peak_step])
    possible_singularity = bool(condition >= singularity_cond_threshold or sigma_min <= singularity_sigma_threshold)
    row: dict[str, Any] = {
        "rollout": rollout_idx,
        "peak_step": peak_step,
        "max_l2": float(l2[peak_step]),
        "peak_action_delta": float(action_delta[peak_step]),
        "rollout_action_delta_p95": float(np.percentile(action_delta, 95)),
        "peak_max_abs_dq": float(dq_abs[peak_step]),
        "rollout_max_abs_dq_p95": float(np.percentile(dq_abs, 95)),
        "peak_min_limit_distance": float(min_limit_distance[peak_step]),
        "peak_jacobian_sigma_min": sigma_min,
        "peak_jacobian_condition": condition,
        "near_joint_limit": near_joint_limit,
        "high_speed": high_speed,
        "action_jump": action_jump,
        "possible_singularity": possible_singularity,
        "teacher_forcing_bad": teacher_forcing_bad,
        "cause_labels": ",".join(
            label
            for label, enabled in [
                ("near_joint_limit", near_joint_limit),
                ("high_speed", high_speed),
                ("action_jump", action_jump),
                ("possible_singularity", possible_singularity),
                ("teacher_forcing_bad", teacher_forcing_bad),
            ]
            if enabled
        ),
    }
    if teacher_l2 is not None:
        row["peak_teacher_l2"] = float(teacher_l2[peak_step])
    row.update(l2_distribution(l2))
    row.update(dominant_errors(errors[peak_step], labels, top_k=min(top_k_errors, len(labels))))
    for idx in range(n_joints):
        row[f"peak_q{idx}"] = float(truth[peak_step, idx])
        row[f"peak_dq{idx}"] = float(truth[peak_step, n_joints + idx])
        row[f"peak_q{idx}_limit_distance"] = float(limit_distance[peak_step, idx])
    return row


def find_end_effector_ids(env: MuJoCoArmEnv) -> tuple[int, int]:
    for name in ("ee_site", "tool0", "flange", "end_effector"):
        site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return site_id, -1
    for name in ("ee_site", "tool0", "flange", "end_effector"):
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return -1, body_id
    return -1, int(env.model.nbody - 1)


def compute_jacobian_stats(env: MuJoCoArmEnv, states: np.ndarray, n_joints: int) -> dict[str, np.ndarray]:
    site_id, body_id = find_end_effector_ids(env)
    sigma_min: list[float] = []
    condition: list[float] = []
    for state in states:
        env.data.qpos[:n_joints] = state[:n_joints]
        env.data.qvel[:n_joints] = state[n_joints : 2 * n_joints]
        mujoco.mj_forward(env.model, env.data)
        jacp = np.zeros((3, env.model.nv))
        jacr = np.zeros((3, env.model.nv))
        if site_id >= 0:
            mujoco.mj_jacSite(env.model, env.data, jacp, jacr, site_id)
        else:
            mujoco.mj_jacBody(env.model, env.data, jacp, jacr, body_id)
        jacobian = np.vstack([jacp[:, :n_joints], jacr[:, :n_joints]])
        svals = np.linalg.svd(jacobian, compute_uv=False)
        sigma = float(svals[-1])
        sigma_min.append(sigma)
        condition.append(float(svals[0] / max(sigma, 1e-12)))
    return {"sigma_min": np.asarray(sigma_min), "condition": np.asarray(condition)}


def load_model_and_normalizer(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, StandardNormalizer, dict[str, Any], int]:
    checkpoint = load_checkpoint(Path(args.checkpoint), map_location=device)
    config = checkpoint.get("config", {})
    state_dim = 2 * args.n_joints
    output_dim = int(config.get("output_dim", state_dim)) if isinstance(config, dict) else state_dim
    history_len = EVAL_DYNAMICS.resolve_history_len(args.model_type, args.history_len, config if isinstance(config, dict) else {})
    model = build_model(args.model_type, state_dim, args.n_joints, history_len, output_dim=output_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = StandardNormalizer.load(Path(args.normalizer), map_location=device)
    return model, normalizer, config, history_len


def analyze_eval_rollouts(args: argparse.Namespace, env: MuJoCoArmEnv, joint_ranges: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, config, history_len = load_model_and_normalizer(args, device)
    target_mode = str(config.get("target_mode", "delta_state")) if isinstance(config, dict) else "delta_state"
    control_dt = float(config.get("control_dt", env.control_dt)) if isinstance(config, dict) else env.control_dt
    action_std = EVAL_DYNAMICS.parse_action_std(args.action_std, args.n_joints)
    labels = state_labels(args.n_joints)
    spike_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    total_steps = args.warmup_steps + args.rollout_len
    for rollout_idx in range(args.num_rollouts):
        true_all, actions_all, _ = EVAL_DYNAMICS.collect_truth_rollout(env, rng, args.n_joints, total_steps, action_std)
        truth = true_all[args.warmup_steps : args.warmup_steps + args.rollout_len]
        actions = actions_all[args.warmup_steps : args.warmup_steps + args.rollout_len]
        pred = EVAL_DYNAMICS.predict_open_loop_segment(
            model,
            normalizer,
            args.model_type,
            true_all,
            actions_all,
            args.warmup_steps,
            args.rollout_len,
            history_len,
            2 * args.n_joints,
            device,
            target_mode,
            control_dt,
        )
        teacher_all = EVAL_DYNAMICS.predict_teacher_forcing(
            model,
            normalizer,
            args.model_type,
            true_all,
            actions_all,
            history_len,
            2 * args.n_joints,
            device,
            target_mode,
            control_dt,
        )
        teacher = teacher_all[args.warmup_steps : args.warmup_steps + args.rollout_len]
        jacobian = compute_jacobian_stats(env, truth, args.n_joints)
        spike_rows.append(
            summarize_rollout_spike(
                rollout_idx,
                truth,
                pred,
                actions,
                joint_ranges,
                labels,
                jacobian,
                teacher,
                args.limit_margin,
                args.action_jump_percentile,
                args.high_speed_percentile,
                args.singularity_cond_threshold,
                args.singularity_sigma_threshold,
                args.top_k_spikes,
            )
        )
        timeline_rows.extend(build_timeline_rows(rollout_idx, truth, pred, actions, joint_ranges, jacobian, teacher))
    return spike_rows, timeline_rows


def build_timeline_rows(
    rollout_idx: int,
    truth: np.ndarray,
    pred: np.ndarray,
    actions: np.ndarray,
    joint_ranges: np.ndarray,
    jacobian: dict[str, np.ndarray],
    teacher: np.ndarray,
) -> list[dict[str, Any]]:
    n_joints = truth.shape[1] // 2
    errors = truth - pred
    l2 = np.linalg.norm(errors, axis=1)
    teacher_l2 = np.linalg.norm(truth - teacher, axis=1)
    action_delta = np.zeros(len(actions), dtype=np.float64)
    if len(actions) > 1:
        action_delta[1:] = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    limit_distance = per_step_limit_distance(truth, joint_ranges, n_joints)
    rows: list[dict[str, Any]] = []
    for step in range(truth.shape[0]):
        row: dict[str, Any] = {
            "rollout": rollout_idx,
            "step": step,
            "l2": float(l2[step]),
            "teacher_l2": float(teacher_l2[step]),
            "action_norm": float(np.linalg.norm(actions[step])),
            "action_delta": float(action_delta[step]),
            "min_limit_distance": float(np.min(limit_distance[step])),
            "jacobian_sigma_min": float(jacobian["sigma_min"][step]),
            "jacobian_condition": float(jacobian["condition"][step]),
        }
        for idx in range(n_joints):
            row[f"q{idx}"] = float(truth[step, idx])
            row[f"dq{idx}"] = float(truth[step, n_joints + idx])
            row[f"q{idx}_limit_distance"] = float(limit_distance[step, idx])
        rows.append(row)
    return rows


def print_console_summary(dataset_rows: list[dict[str, Any]], spike_rows: list[dict[str, Any]]) -> None:
    ok_datasets = [row for row in dataset_rows if row.get("status") == "ok"]
    if ok_datasets:
        worst_q1 = max(ok_datasets, key=lambda row: float(row.get("q1_near_upper_rate", 0.0)))
        print(
            "dataset joint2 near-upper worst="
            f"{Path(str(worst_q1['path'])).name} rate={float(worst_q1.get('q1_near_upper_rate', 0.0)):.3f} "
            f"violation={float(worst_q1.get('q1_violation_rate', 0.0)):.3f}"
        )
    for row in sorted(spike_rows, key=lambda item: float(item["max_l2"]), reverse=True)[:5]:
        print(
            f"rollout={row['rollout']} peak_step={row['peak_step']} max_l2={float(row['max_l2']):.6f} "
            f"dominant={row.get('dominant_error_0')} causes={row.get('cause_labels')}"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    env = MuJoCoArmEnv(str(resolve_project_path(args.model_xml, ROOT)), n_joints=args.n_joints, seed=args.seed)
    try:
        joint_ranges = controlled_joint_ranges(env, args.n_joints)
        near_limit_overrides = {1: args.near_limit_q1} if args.n_joints > 1 else {}
        dataset_rows = [
            summarize_dataset_file(
                path,
                args.n_joints,
                joint_ranges,
                args.limit_margin,
                near_limit_overrides,
                args.max_dataset_samples,
            )
            for path in expand_dataset_paths(args.dataset_paths)
        ]
        spike_rows, timeline_rows = analyze_eval_rollouts(args, env, joint_ranges)
    finally:
        env.close()
    write_rows(save_dir / "dataset_quality.csv", dataset_rows)
    write_rows(save_dir / "rollout_spikes.csv", spike_rows)
    write_rows(save_dir / "rollout_timeline.csv", timeline_rows)
    print_console_summary(dataset_rows, spike_rows)
    print(f"saved diagnostics to {save_dir}")


if __name__ == "__main__":
    main()
