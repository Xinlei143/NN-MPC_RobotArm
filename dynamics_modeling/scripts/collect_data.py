from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DYNAMICS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DYNAMICS_ROOT.parent
for root in (PROJECT_ROOT, DYNAMICS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import numpy as np
import mujoco
from neural_dynamics.parallel_collector import collect_parallel_detailed, collect_rollouts_detailed, save_dataset, validate_append_dataset
from mpc.robot_config import file_sha256, load_robot_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MuJoCo arm dynamics data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot_config", default="configs/robots/abb_irb2400.yaml")
    parser.add_argument("--model_xml", default=None, type=str, help="Advanced RobotSpec XML override")
    parser.add_argument("--n_joints", default=None, type=int, help="Compatibility check; must match RobotSpec")
    parser.add_argument("--num_episodes", default=20, type=int)
    parser.add_argument("--episode_len", default=200, type=int)
    parser.add_argument("--save_path", default="outputs/datasets/arm_data.npz", type=str)
    parser.add_argument("--action_std", default="0.5", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--settle_steps", default=50, type=int, help="Steps to hold q_ref=q after reset before recording")
    parser.add_argument("--append", action="store_true", help="Append new samples to save_path if it already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError(f"num_envs must be at least 1, got {args.num_envs}")

    robot = load_robot_spec(args.robot_config, validate_model=True)
    robot = robot.with_runtime_overrides(model_xml=args.model_xml)
    if args.n_joints is not None and args.n_joints != robot.n_joints:
        raise ValueError("--n_joints must match RobotSpec")
    args.n_joints = robot.n_joints
    save_path = Path(args.save_path)
    manifest_path = save_path.with_suffix(".manifest.json")
    existing_manifest = None
    if args.append:
        validate_append_dataset(save_path)
        if not manifest_path.is_file():
            raise FileNotFoundError("Strict append requires the existing dataset manifest")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("robot_identity") != robot.artifact_identity():
            raise ValueError("Cannot append data collected with a different robot identity")

    model_xml = str(robot.model_xml)
    if args.num_envs == 1:
        data = collect_rollouts_detailed(
            model_xml=model_xml,
            n_joints=args.n_joints,
            num_episodes=args.num_episodes,
            episode_len=args.episode_len,
            action_std=args.action_std,
            seed=args.seed,
            worker_id=0,
            settle_steps=args.settle_steps,
            robot_spec=robot,
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
            robot_spec=robot,
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
    action = "Appended dataset to" if args.append else "Saved dataset to"
    compiled_model = mujoco.MjModel.from_xml_path(model_xml)
    env_bounds = robot.collection_bounds(
        np.asarray(compiled_model.jnt_range[: robot.n_joints, 0], dtype=np.float32),
        np.asarray(compiled_model.jnt_range[: robot.n_joints, 1], dtype=np.float32),
    )
    with np.load(save_path, allow_pickle=False) as archive:
        mode_ids = np.asarray(archive["motion_mode_ids"], dtype=np.int64)
    unique_modes, mode_counts = np.unique(mode_ids, return_counts=True)
    collection_runs = [] if existing_manifest is None else list(
        existing_manifest.get("collection_runs", [])
    )
    collection_runs.append({
        "num_episodes_requested": int(args.num_episodes),
        "episode_len": int(args.episode_len),
        "action_std_normalized": args.action_std,
        "seed": int(args.seed),
        "num_envs": int(args.num_envs),
        "settle_steps": int(args.settle_steps),
    })
    manifest = {
        "schema_version": 1,
        "kind": "robot_dynamics_dataset",
        "robot_identity": robot.artifact_identity(),
        "dataset": {
            "path": str(save_path),
            "sha256": file_sha256(save_path),
            "samples": int(len(states)),
            "episodes": int(len(np.unique(episode_ids))),
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
        },
        "collection": {
            "control_dt": robot.expected_control_dt,
            "workspace": {key: value.tolist() for key, value in env_bounds.items()},
            "motion_mode_sample_counts": {
                str(int(key)): int(value) for key, value in zip(unique_modes, mode_counts, strict=True)
            },
        },
        "collection_runs": collection_runs,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"{action} {save_path} with states={states.shape}, actions={actions.shape}, "
        f"next_states={next_states.shape}, episode_ids={episode_ids.shape}; manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
