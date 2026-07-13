from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from neural_dynamics.mujoco_env import MuJoCoArmEnv
from neural_dynamics.parallel_collector import MOTION_MODE_NAMES, TERMINATION_REASON_CODES, parse_action_std, sample_smooth_action
from neural_dynamics.paths import DEFAULT_MODEL_XML, resolve_project_path


@dataclass(frozen=True)
class QaccAlignmentRecord:
    control_delta: np.ndarray
    summed_substep_delta: np.ndarray
    post_step_qacc_delta: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose MuJoCo learned-dynamics datasets and actuator-step alignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--model_xml", default=DEFAULT_MODEL_XML, type=str)
    parser.add_argument("--n_joints", default=6, type=int)
    parser.add_argument("--save_csv", default=None, type=str)
    parser.add_argument("--lag_csv", default=None, type=str)
    parser.add_argument("--coverage_dir", default=None, type=str, help="Directory for q/dq/q_ref/tau coverage CSVs and histograms")
    parser.add_argument("--qacc_rollout_steps", default=0, type=int)
    parser.add_argument("--action_std", default="0.3", type=str)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def state_labels(n_joints: int) -> list[str]:
    return [f"q{idx}" for idx in range(n_joints)] + [f"dq{idx}" for idx in range(n_joints)]


def summarize_columns(values: np.ndarray, labels: list[str]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    abs_values = np.abs(values)
    for idx, label in enumerate(labels):
        column = values[:, idx]
        abs_column = abs_values[:, idx]
        rows.append(
            {
                "label": label,
                "mean": float(np.mean(column)),
                "std": float(np.std(column)),
                "p50_abs": float(np.percentile(abs_column, 50)),
                "p90_abs": float(np.percentile(abs_column, 90)),
                "p99_abs": float(np.percentile(abs_column, 99)),
                "max_abs": float(np.max(abs_column)),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def joint_limit_arrays(n_joints: int) -> tuple[np.ndarray, np.ndarray]:
    default_low = np.full(n_joints, -1.0, dtype=np.float64)
    default_high = np.full(n_joints, 1.0, dtype=np.float64)
    irb_low = np.array([-3.1416, -1.7453, -1.0472, -3.49, -2.0944, -6.9813], dtype=np.float64)
    irb_high = np.array([3.1416, 1.9199, 1.1345, 3.49, 2.0944, 6.9813], dtype=np.float64)
    if n_joints <= len(irb_low):
        return irb_low[:n_joints], irb_high[:n_joints]
    default_low[: len(irb_low)] = irb_low
    default_high[: len(irb_high)] = irb_high
    return default_low, default_high


def resolve_joint_limits(n_joints: int, model_xml: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if model_xml is None:
        return joint_limit_arrays(n_joints)
    env = MuJoCoArmEnv(model_xml=model_xml, n_joints=n_joints)
    try:
        return env.joint_low.astype(np.float64), env.joint_high.astype(np.float64)
    finally:
        env.close()


def normalize_joint_values(values: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    center = (high + low) / 2.0
    half_range = (high - low) / 2.0
    return (values - center) / half_range


def coverage_rows(values: np.ndarray, labels: list[str], field: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for idx, label in enumerate(labels):
        column = values[:, idx]
        rows.append(
            {
                "field": field,
                "label": label,
                "min": float(np.min(column)),
                "max": float(np.max(column)),
                "mean": float(np.mean(column)),
                "std": float(np.std(column)),
                "p01": float(np.percentile(column, 1)),
                "p50": float(np.percentile(column, 50)),
                "p99": float(np.percentile(column, 99)),
                "max_abs": float(np.max(np.abs(column))),
            }
        )
    return rows


def joint_limit_margin_rows(
    q_norm: np.ndarray,
    labels: list[str],
    q_ref_norm: np.ndarray | None = None,
    near_limit_threshold: float = 0.9,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for idx, label in enumerate(labels):
        q_abs = np.abs(q_norm[:, idx])
        row: dict[str, float | int | str] = {
            "label": label,
            "q_max_abs_norm": float(np.max(q_abs)),
            "q_margin_to_limit": float(1.0 - np.max(q_abs)),
            "q_near_limit_fraction": float(np.mean(q_abs >= near_limit_threshold)),
            "q_outside_limit_count": int(np.sum(q_abs > 1.0)),
        }
        if q_ref_norm is not None:
            q_ref_abs = np.abs(q_ref_norm[:, idx])
            row.update(
                {
                    "q_ref_max_abs_norm": float(np.max(q_ref_abs)),
                    "q_ref_margin_to_limit": float(1.0 - np.max(q_ref_abs)),
                    "q_ref_near_limit_fraction": float(np.mean(q_ref_abs >= near_limit_threshold)),
                    "q_ref_outside_limit_count": int(np.sum(q_ref_abs > 1.0)),
                }
            )
        rows.append(row)
    return rows


def plot_histograms(values: np.ndarray, labels: list[str], title: str, save_path: Path, xlim: tuple[float, float] | None = None) -> None:
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(labels) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, max(4, rows * 3)))
    axes_array = np.asarray(axes).reshape(-1)
    for idx, label in enumerate(labels):
        ax = axes_array[idx]
        ax.hist(values[:, idx], bins=80, alpha=0.85)
        if xlim is not None:
            ax.set_xlim(*xlim)
            ax.axvline(xlim[0] * 0.9, color="tab:red", linestyle="--", linewidth=0.8)
            ax.axvline(xlim[1] * 0.9, color="tab:red", linestyle="--", linewidth=0.8)
        ax.set_title(label)
        ax.grid(True, alpha=0.2)
    for ax in axes_array[len(labels) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def selected_episode_indices(data: np.lib.npyio.NpzFile, sample_count: int, max_episodes: int = 8) -> list[tuple[int, np.ndarray]]:
    if "episode_ids" not in data.files:
        return [(0, np.arange(sample_count, dtype=np.int64))]
    episode_ids = data["episode_ids"]
    selected: list[tuple[int, np.ndarray]] = []
    for episode_id in np.unique(episode_ids)[:max_episodes]:
        indices = np.where(episode_ids == episode_id)[0]
        if len(indices):
            selected.append((int(episode_id), indices))
    return selected


def plot_normalized_trajectories(
    values_norm: np.ndarray,
    labels: list[str],
    episodes: list[tuple[int, np.ndarray]],
    title: str,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(labels) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, max(4, rows * 3)), sharex=False)
    axes_array = np.asarray(axes).reshape(-1)
    for joint_idx, label in enumerate(labels):
        ax = axes_array[joint_idx]
        for episode_id, indices in episodes:
            ax.plot(np.arange(len(indices)), values_norm[indices, joint_idx], linewidth=1.0, label=f"ep {episode_id}")
        ax.axhline(-1.0, color="tab:red", linestyle="--", linewidth=0.8)
        ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=0.8)
        ax.set_title(label)
        ax.set_ylabel("normalized q")
        ax.grid(True, alpha=0.2)
    for ax in axes_array[len(labels) :]:
        ax.axis("off")
    if episodes:
        axes_array[0].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_q_ref_tracking(
    q_norm: np.ndarray,
    q_ref_norm: np.ndarray,
    labels: list[str],
    episodes: list[tuple[int, np.ndarray]],
    title: str,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(labels) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, max(4, rows * 3)), sharex=False)
    axes_array = np.asarray(axes).reshape(-1)
    for joint_idx, label in enumerate(labels):
        ax = axes_array[joint_idx]
        for episode_id, indices in episodes:
            x = np.arange(len(indices))
            ax.plot(x, q_norm[indices, joint_idx], linewidth=1.0, label=f"q ep {episode_id}")
            ax.plot(x, q_ref_norm[indices, joint_idx], linewidth=0.9, linestyle="--", label=f"q_ref ep {episode_id}")
        ax.axhline(-1.0, color="tab:red", linestyle="--", linewidth=0.8)
        ax.axhline(1.0, color="tab:red", linestyle="--", linewidth=0.8)
        ax.set_title(label)
        ax.set_ylabel("normalized")
        ax.grid(True, alpha=0.2)
    for ax in axes_array[len(labels) :]:
        ax.axis("off")
    if episodes:
        axes_array[0].legend(loc="best", fontsize=7, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def write_count_summary(path: Path, name: str, values: np.ndarray, value_labels: dict[int, str] | None = None) -> None:
    unique, counts = np.unique(values, return_counts=True)
    rows = []
    for value, count in zip(unique, counts):
        row = {name: int(value), "samples": int(count)}
        if value_labels is not None:
            row["label"] = value_labels.get(int(value), "unknown")
        rows.append(row)
    write_rows(path, rows)


def write_coverage_diagnostics(
    data_path: Path,
    n_joints: int | None,
    save_dir: Path,
    model_xml: str | None = None,
) -> list[dict[str, float | str]]:
    data = np.load(data_path)
    states = data["states"]
    if n_joints is None:
        n_joints = states.shape[1] // 2
    save_dir = Path(save_dir)
    labels = [f"joint_{idx + 1}" for idx in range(n_joints)]
    joint_low, joint_high = resolve_joint_limits(n_joints, model_xml)

    q = states[:, :n_joints]
    dq = states[:, n_joints : 2 * n_joints]
    q_norm = normalize_joint_values(q, joint_low, joint_high)
    q_ref_norm = None
    fields: list[tuple[str, np.ndarray, tuple[float, float] | None]] = [
        ("q", q, None),
        ("q_norm", q_norm, (-1.0, 1.0)),
        ("dq", dq, None),
        ("action", data["actions"], None),
    ]
    if "q_ref" in data.files:
        q_ref = data["q_ref"]
        q_ref_norm = normalize_joint_values(q_ref, joint_low, joint_high)
        fields.append(("q_ref", q_ref, None))
        fields.append(("q_ref_norm", q_ref_norm, (-1.0, 1.0)))
    for key in ("delta_q_ref", "tau_actuator", "tau_gravity", "tau_total", "action_std_normalized"):
        if key in data.files:
            fields.append((key, data[key], None))

    rows: list[dict[str, float | str]] = []
    for field, values, xlim in fields:
        rows.extend(coverage_rows(values, labels, field))
        plot_histograms(values, labels, field, save_dir / f"{field}_hist.png", xlim=xlim)
    write_rows(save_dir / "coverage_summary.csv", rows)
    write_rows(save_dir / "joint_limit_margin_summary.csv", joint_limit_margin_rows(q_norm, labels, q_ref_norm))

    if "termination_reasons" in data.files:
        termination_labels = {code: name for name, code in TERMINATION_REASON_CODES.items()}
        write_count_summary(
            save_dir / "termination_summary.csv",
            "termination_reason",
            data["termination_reasons"],
            value_labels=termination_labels,
        )
    if "motion_mode_ids" in data.files:
        motion_mode_labels = {idx: name for idx, name in enumerate(MOTION_MODE_NAMES)}
        write_count_summary(
            save_dir / "motion_mode_summary.csv",
            "motion_mode_id",
            data["motion_mode_ids"],
            value_labels=motion_mode_labels,
        )
    if "settle_steps" in data.files:
        write_count_summary(save_dir / "settle_steps_summary.csv", "settle_steps", data["settle_steps"])
    episodes = selected_episode_indices(data, len(states), max_episodes=8)
    if episodes:
        plot_normalized_trajectories(
            q_norm,
            labels,
            episodes,
            "q_norm trajectories",
            save_dir / "q_norm_trajectories_first8.png",
        )
        if "q_ref" in data.files:
            plot_q_ref_tracking(
                q_norm,
                q_ref_norm,
                labels,
                episodes[:2],
                "q_norm vs q_ref_norm tracking",
                save_dir / "q_ref_tracking_first2.png",
            )
    return rows


def print_table(title: str, rows: list[dict[str, float | str]]) -> None:
    print(f"\n{title}")
    print("label      mean        std    p50_abs    p90_abs    p99_abs    max_abs")
    for row in rows:
        print(
            f"{row['label']:<6}"
            f" {row['mean']:>9.5f}"
            f" {row['std']:>10.5f}"
            f" {row['p50_abs']:>10.5f}"
            f" {row['p90_abs']:>10.5f}"
            f" {row['p99_abs']:>10.5f}"
            f" {row['max_abs']:>10.5f}"
        )


def lag_correlation_rows(
    actions: np.ndarray,
    delta_dq: np.ndarray,
    episode_ids: np.ndarray | None,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    n_joints = actions.shape[1]
    indices = np.arange(len(actions))
    for lag in (-1, 0, 1):
        action_indices = indices + lag
        valid = (action_indices >= 0) & (action_indices < len(actions))
        if episode_ids is not None:
            valid &= episode_ids[action_indices.clip(0, len(actions) - 1)] == episode_ids
        if not np.any(valid):
            continue
        shifted_actions = actions[action_indices[valid]]
        valid_delta = delta_dq[valid]
        for idx in range(n_joints):
            x = shifted_actions[:, idx]
            y = valid_delta[:, idx]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(x, y)[0, 1])
            rows.append({"lag": lag, "joint": idx, "corr_action_delta_dq": corr, "samples": int(np.sum(valid))})
    return rows


def step_with_alignment_records(env: MuJoCoArmEnv, action: np.ndarray, n_joints: int) -> QaccAlignmentRecord:
    action_array = np.asarray(action, dtype=np.float64)
    if action_array.shape != (n_joints,):
        raise ValueError(f"Action must have shape ({n_joints},), got {action_array.shape}")

    env.data.ctrl[:n_joints] = np.clip(action_array, env.action_low, env.action_high)
    qvel_before = env.data.qvel[:n_joints].copy()
    summed_substep_delta = np.zeros(n_joints, dtype=np.float64)
    post_step_qacc_delta = np.zeros(n_joints, dtype=np.float64)
    for _substep in range(env.frame_skip):
        substep_qvel_before = env.data.qvel[:n_joints].copy()
        if env.gravity_compensation:
            env.data.qfrc_applied[:n_joints] = env._gravity_compensation_force()
        mujoco.mj_step(env.model, env.data)
        substep_qvel_after = env.data.qvel[:n_joints].copy()
        summed_substep_delta += substep_qvel_after - substep_qvel_before
        post_step_qacc_delta += env.data.qacc[:n_joints] * env.model.opt.timestep
    qvel_after = env.data.qvel[:n_joints].copy()
    return QaccAlignmentRecord(
        control_delta=qvel_after - qvel_before,
        summed_substep_delta=summed_substep_delta,
        post_step_qacc_delta=post_step_qacc_delta,
    )


def run_qacc_alignment_check(model_xml: str, n_joints: int, rollout_steps: int, action_std: str, seed: int) -> None:
    if rollout_steps <= 0:
        return
    env = MuJoCoArmEnv(model_xml=model_xml, n_joints=n_joints, seed=seed)
    rng = np.random.default_rng(seed)
    parsed_action_std = parse_action_std(action_std, n_joints)
    state = env.reset_random()
    action = np.zeros(n_joints, dtype=np.float32)
    try:
        exact_errors = []
        euler_qacc_errors = []
        for _ in range(rollout_steps):
            action = sample_smooth_action(
                rng,
                action,
                parsed_action_std,
                n_joints,
                action_low=env.action_low,
                action_high=env.action_high,
            )
            record = step_with_alignment_records(env, action, n_joints)
            exact_errors.append(record.control_delta - record.summed_substep_delta)
            euler_qacc_errors.append(record.control_delta - record.post_step_qacc_delta)
            state = env.get_state()
        exact_abs_error = np.abs(np.asarray(exact_errors))
        euler_abs_error = np.abs(np.asarray(euler_qacc_errors))
        print("\nqacc substep integration check")
        print(f"control_dt={env.control_dt:.6f}, rollout_steps={rollout_steps}, final_state_norm={np.linalg.norm(state):.6f}")
        print("exact_substep_qvel_max_abs_error_by_dq=", np.round(np.max(exact_abs_error, axis=0), 8).tolist())
        print("exact_substep_qvel_mean_abs_error_by_dq=", np.round(np.mean(exact_abs_error, axis=0), 8).tolist())
        print("post_step_qacc_euler_max_abs_error_by_dq=", np.round(np.max(euler_abs_error, axis=0), 8).tolist())
        print("post_step_qacc_euler_mean_abs_error_by_dq=", np.round(np.mean(euler_abs_error, axis=0), 8).tolist())
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    if args.data_path is not None:
        data = np.load(args.data_path)
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]
        episode_ids = data["episode_ids"] if "episode_ids" in data.files else None
        n_joints = states.shape[1] // 2
        deltas = next_states - states
        delta_rows = summarize_columns(deltas, state_labels(n_joints))
        action_rows = summarize_columns(actions, [f"u{idx}" for idx in range(actions.shape[1])])
        print_table("delta_state distribution", delta_rows)
        print_table("action distribution", action_rows)
        if args.save_csv is not None:
            write_rows(Path(args.save_csv), delta_rows + action_rows)
        if args.coverage_dir is not None:
            rows = write_coverage_diagnostics(
                Path(args.data_path),
                n_joints,
                Path(args.coverage_dir),
                model_xml=str(resolve_project_path(args.model_xml, ROOT)),
            )
            print(f"\nwrote coverage diagnostics to {args.coverage_dir} with rows={len(rows)}")
        lag_rows = lag_correlation_rows(actions, deltas[:, n_joints:], episode_ids)
        if lag_rows:
            print("\naction lag correlation with delta_dq")
            for row in lag_rows:
                print(
                    f"lag={row['lag']:>2} joint={row['joint']} "
                    f"corr={row['corr_action_delta_dq']:.5f} samples={row['samples']}"
                )
            if args.lag_csv is not None:
                write_rows(Path(args.lag_csv), lag_rows)
    run_qacc_alignment_check(
        str(resolve_project_path(args.model_xml, ROOT)),
        args.n_joints,
        args.qacc_rollout_steps,
        args.action_std,
        args.seed,
    )


if __name__ == "__main__":
    main()
