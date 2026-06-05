from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learned_dynamics.parallel_collector import (
    collect_parallel_detailed,
    collect_rollouts_detailed,
    save_dataset,
    validate_append_dataset,
)
from learned_dynamics.paths import DEFAULT_MODEL_XML, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect episode-aware MuJoCo arm dynamics data for Transformer training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_xml", default=DEFAULT_MODEL_XML, type=str, help="MuJoCo XML/MJCF model path")
    parser.add_argument("--n_joints", default=6, type=int)
    parser.add_argument("--num_episodes", default=15000, type=int)
    parser.add_argument("--episode_len", default=600, type=int)
    parser.add_argument("--save_path", default="outputs/datasets/irb2400_parallel_data_transformer.npz", type=str)
    parser.add_argument("--action_std", default="0.3", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_envs", default=32, type=int)
    parser.add_argument("--history_len", default=16, type=int, help="Recommended Transformer history length")
    parser.add_argument("--settle_steps", default=50, type=int, help="Steps to hold q_ref=q after reset before recording")
    parser.add_argument("--append", action="store_true", help="Append new samples to save_path if it already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError(f"num_envs must be at least 1, got {args.num_envs}")
    if args.history_len <= 0:
        raise ValueError(f"history_len must be positive, got {args.history_len}")
    if args.episode_len < args.history_len:
        raise ValueError(
            f"episode_len={args.episode_len} must be at least history_len={args.history_len} "
            "so each episode contributes sequence windows."
        )

    save_path = Path(args.save_path)
    if args.append:
        validate_append_dataset(save_path, require_episode_ids=True)

    model_xml = str(resolve_project_path(args.model_xml, ROOT))
    if args.num_envs == 1:
        data = collect_rollouts_detailed(
            model_xml=model_xml,
            n_joints=args.n_joints,
            num_episodes=args.num_episodes,
            episode_len=args.episode_len,
            action_std=args.action_std,
            seed=args.seed,
            worker_id=0,
            episode_id_offset=0,
            settle_steps=args.settle_steps,
        )
    else:
        data = collect_parallel_detailed(
            model_xml=model_xml,
            n_joints=args.n_joints,
            num_episodes=args.num_episodes,
            episode_len=args.episode_len,
            action_std=args.action_std,
            seed=args.seed,
            num_envs=args.num_envs,
            settle_steps=args.settle_steps,
        )

    extra_arrays = {key: value for key, value in data.items() if key not in {"states", "actions", "next_states", "episode_ids"}}
    states, actions, next_states, episode_ids = save_dataset(
        save_path,
        data["states"],
        data["actions"],
        data["next_states"],
        append=args.append,
        episode_ids=data["episode_ids"],
        extra_arrays=extra_arrays,
    )
    action = "Appended Transformer dataset to" if args.append else "Saved Transformer dataset to"
    print(
        f"{action} {save_path} with states={states.shape}, actions={actions.shape}, "
        f"next_states={next_states.shape}, episode_ids={episode_ids.shape}, "
        f"recommended_history_len={args.history_len}"
    )


if __name__ == "__main__":
    main()
