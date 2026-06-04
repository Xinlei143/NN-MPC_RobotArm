from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np

from learned_dynamics.mujoco_env import MuJoCoArmEnv
from learned_dynamics.parallel_collector import parse_action_std, sample_smooth_action
from learned_dynamics.paths import DEFAULT_MODEL_XML, resolve_project_path


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
            qvel_before = env.data.qvel[:n_joints].copy()
            summed_substep_delta = np.zeros(n_joints, dtype=np.float64)
            post_step_qacc_delta = np.zeros(n_joints, dtype=np.float64)
            env.data.ctrl[:n_joints] = action
            for _substep in range(env.frame_skip):
                substep_qvel_before = env.data.qvel[:n_joints].copy()
                mujoco.mj_step(env.model, env.data)
                substep_qvel_after = env.data.qvel[:n_joints].copy()
                summed_substep_delta += substep_qvel_after - substep_qvel_before
                post_step_qacc_delta += env.data.qacc[:n_joints] * env.model.opt.timestep
            qvel_after = env.data.qvel[:n_joints].copy()
            control_delta = qvel_after - qvel_before
            exact_errors.append(control_delta - summed_substep_delta)
            euler_qacc_errors.append(control_delta - post_step_qacc_delta)
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
