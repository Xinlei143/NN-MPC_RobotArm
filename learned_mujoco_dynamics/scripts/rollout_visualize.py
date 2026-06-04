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
from learned_dynamics.parallel_collector import sample_smooth_action
from learned_dynamics.paths import DEFAULT_MODEL_XML, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize random MuJoCo arm rollouts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_xml", default=DEFAULT_MODEL_XML, type=str, help="MuJoCo XML/MJCF model path")
    parser.add_argument("--n_joints", default=6, type=int)
    parser.add_argument("--episode_len", default=1000, type=int)
    parser.add_argument("--action_std", default=0.5, type=float)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = MuJoCoArmEnv(str(resolve_project_path(args.model_xml, ROOT)), n_joints=args.n_joints, seed=args.seed)
    action = np.zeros(args.n_joints, dtype=np.float32)
    env.reset_random()
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for _ in range(args.episode_len):
                if not viewer.is_running():
                    break
                action = sample_smooth_action(rng, action, args.action_std, args.n_joints)
                env.step(action)
                viewer.sync()
                time.sleep(env.model.opt.timestep * env.frame_skip)
    finally:
        env.close()


if __name__ == "__main__":
    main()
