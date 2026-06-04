from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot q and dq channels from a learned-dynamics .npz dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--save_path", default="outputs/figures/dataset_q_dq.png", type=str)
    parser.add_argument("--max_samples", default=2000, type=int)
    parser.add_argument(
        "--episode_index",
        default=None,
        type=int,
        help="Ordinal episode to plot when episode_ids exists; defaults to the first samples.",
    )
    return parser.parse_args()


def select_states(data: np.lib.npyio.NpzFile, max_samples: int, episode_index: int | None) -> tuple[np.ndarray, str]:
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive, got {max_samples}")
    states = data["states"]
    if episode_index is None or "episode_ids" not in data.files:
        count = min(max_samples, len(states))
        return states[:count], f"first {count} samples"

    episode_ids = data["episode_ids"]
    unique_ids = np.unique(episode_ids)
    if episode_index < 0 or episode_index >= len(unique_ids):
        raise ValueError(f"episode_index must be in [0, {len(unique_ids) - 1}], got {episode_index}")
    episode_id = unique_ids[episode_index]
    indices = np.flatnonzero(episode_ids == episode_id)
    indices = indices[:max_samples]
    return states[indices], f"episode_index={episode_index}, episode_id={episode_id}, samples={len(indices)}"


def plot_q_dq(states: np.ndarray, title_suffix: str, save_path: Path) -> None:
    if states.ndim != 2 or states.shape[1] % 2 != 0:
        raise ValueError(f"states must have shape [N, 2 * n_joints], got {states.shape}")
    n_joints = states.shape[1] // 2
    q = states[:, :n_joints]
    dq = states[:, n_joints:]
    steps = np.arange(len(states))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for idx in range(n_joints):
        axes[0].plot(steps, q[:, idx], linewidth=1.0, label=f"q{idx}")
        axes[1].plot(steps, dq[:, idx], linewidth=1.0, label=f"dq{idx}")
    axes[0].set_ylabel("q [rad]")
    axes[1].set_ylabel("dq [rad/s]")
    axes[1].set_xlabel("sample")
    axes[0].set_title(f"Dataset q channels ({title_suffix})")
    axes[1].set_title(f"Dataset dq channels ({title_suffix})")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=6, fontsize=9)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {data_path}")
    with np.load(data_path) as data:
        missing = {"states"}.difference(data.files)
        if missing:
            raise KeyError(f"Dataset file {data_path} is missing arrays: {sorted(missing)}")
        states, title_suffix = select_states(data, args.max_samples, args.episode_index)
        plot_q_dq(states, title_suffix, Path(args.save_path))
    print(f"saved q/dq plot: {args.save_path}")


if __name__ == "__main__":
    main()
