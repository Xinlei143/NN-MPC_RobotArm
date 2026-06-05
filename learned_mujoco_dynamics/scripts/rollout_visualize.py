from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mujoco.viewer
import numpy as np

from learned_dynamics.mujoco_env import MuJoCoArmEnv
from learned_dynamics.parallel_collector import (
    MOTION_MODE_NAMES,
    generate_q_ref_sequence,
    parse_action_std,
    reset_safe_workspace,
    sample_smooth_action,
)
from learned_dynamics.paths import DEFAULT_MODEL_XML, resolve_project_path


def sample_visualization_action(
    rng: np.random.Generator,
    previous_action: np.ndarray,
    action_std: float | np.ndarray,
    n_joints: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    return sample_smooth_action(
        rng,
        previous_action,
        action_std,
        n_joints,
        action_low=action_low,
        action_high=action_high,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize structured closed-loop MuJoCo arm q_ref rollouts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_xml", default=DEFAULT_MODEL_XML, type=str, help="MuJoCo XML/MJCF model path")
    parser.add_argument("--n_joints", default=6, type=int)
    parser.add_argument("--episode_len", default=1000, type=int)
    parser.add_argument("--action_std", default="0.5", type=str)
    parser.add_argument("--settle_steps", default=50, type=int, help="Steps to hold q_ref=q after reset before playback")
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def build_visualization_q_refs(
    rng: np.random.Generator,
    start_q_ref: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    episode_len: int,
    action_std: np.ndarray,
) -> np.ndarray:
    segments: list[np.ndarray] = []
    remaining = episode_len
    current_q_ref = np.asarray(start_q_ref, dtype=np.float32)
    for mode_id in range(len(MOTION_MODE_NAMES)):
        segment_len = remaining // (len(MOTION_MODE_NAMES) - mode_id)
        if segment_len <= 0:
            continue
        segment = generate_q_ref_sequence(
            rng,
            current_q_ref,
            action_low,
            action_high,
            segment_len,
            action_std,
            mode_id,
        )
        segments.append(segment)
        current_q_ref = segment[-1]
        remaining -= segment_len
    if not segments:
        return np.empty((0, len(start_q_ref)), dtype=np.float32)
    return np.concatenate(segments, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = MuJoCoArmEnv(str(resolve_project_path(args.model_xml, ROOT)), n_joints=args.n_joints, seed=args.seed)
    state = reset_safe_workspace(env, rng, args.n_joints)
    q_ref = np.asarray(state[: args.n_joints], dtype=np.float32).copy()
    for _ in range(args.settle_steps):
        state = env.step(q_ref)
    parsed_action_std = parse_action_std(args.action_std, args.n_joints)
    action_std = (
        np.full(args.n_joints, float(parsed_action_std), dtype=np.float32)
        if np.isscalar(parsed_action_std)
        else np.asarray(parsed_action_std, dtype=np.float32)
    )
    q_ref_sequence = build_visualization_q_refs(
        rng,
        q_ref,
        env.action_low,
        env.action_high,
        args.episode_len,
        action_std,
    )
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for q_ref in q_ref_sequence:
                if not viewer.is_running():
                    break
                env.step(q_ref)
                viewer.sync()
                time.sleep(env.model.opt.timestep * env.frame_skip)
    finally:
        env.close()


if __name__ == "__main__":
    main()
