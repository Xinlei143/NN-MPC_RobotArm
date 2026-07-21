"""Build a fixed, episode-contiguous Model-A replay subset from a large NPZ."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stratified complete-episode Model-A replay NPZ.")
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--output_path", default="outputs/model_c/model_a_replay_1m.npz")
    parser.add_argument("--staging_dir", default="outputs/model_c/staging/model_a_replay")
    parser.add_argument("--target_transitions", default=1_000_000, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--keep_staging", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source_path).expanduser().resolve()
    output = (ROOT / args.output_path).resolve() if not Path(args.output_path).is_absolute() else Path(args.output_path)
    staging = (ROOT / args.staging_dir).resolve() if not Path(args.staging_dir).is_absolute() else Path(args.staging_dir)
    if args.target_transitions <= 0:
        raise ValueError("target_transitions must be positive")
    staging.mkdir(parents=True, exist_ok=True)
    members = ("states.npy", "actions.npy", "next_states.npy", "episode_ids.npy", "motion_mode_ids.npy")
    with zipfile.ZipFile(source) as archive:
        missing = set(members).difference(archive.namelist())
        if missing:
            raise KeyError(f"Source archive is missing {sorted(missing)}")
        for member in members:
            destination = staging / member
            if not destination.exists():
                with archive.open(member) as reader, destination.open("wb") as writer:
                    shutil.copyfileobj(reader, writer, length=1 << 20)
    arrays = {name[:-4]: np.load(staging / name, mmap_mode="r") for name in members}
    episode_ids = arrays["episode_ids"]
    boundaries = np.flatnonzero(np.diff(episode_ids) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_ids)]))
    modes = arrays["motion_mode_ids"][starts]
    rng = np.random.default_rng(args.seed)
    selected: list[tuple[int, int]] = []
    for mode in np.unique(modes):
        quota = int(round(args.target_transitions * np.mean(modes == mode)))
        candidates = np.flatnonzero(modes == mode)
        rng.shuffle(candidates)
        total = 0
        for index in candidates:
            start, end = int(starts[index]), int(ends[index])
            selected.append((start, end))
            total += end - start
            if total >= quota:
                break
    selected.sort()
    total = sum(end - start for start, end in selected)
    result = {name: np.empty((total, *arrays[name].shape[1:]), dtype=arrays[name].dtype) for name in ("states", "actions", "next_states")}
    result["episode_ids"] = np.empty(total, dtype=np.int64)
    result["split_group_ids"] = np.empty(total, dtype=np.int64)
    result["source_ids"] = np.zeros(total, dtype=np.int16)
    result["valid_target"] = np.ones(total, dtype=np.int8)
    result["context_only"] = np.zeros(total, dtype=np.int8)
    result["motion_mode_ids"] = np.empty(total, dtype=np.int16)
    cursor = 0
    for new_id, (start, end) in enumerate(selected):
        count = end - start
        for name in ("states", "actions", "next_states"):
            result[name][cursor : cursor + count] = arrays[name][start:end]
        result["episode_ids"][cursor : cursor + count] = new_id
        result["split_group_ids"][cursor : cursor + count] = new_id
        result["motion_mode_ids"][cursor : cursor + count] = arrays["motion_mode_ids"][start:end]
        cursor += count
    result["q_ref"] = result["actions"].copy()
    result["delta_q_ref"] = np.zeros_like(result["actions"])
    for start, end in zip(np.flatnonzero(np.r_[True, np.diff(result["episode_ids"]) != 0]), np.r_[np.flatnonzero(np.diff(result["episode_ids"]) != 0) + 1, len(result["episode_ids"])]):
        result["delta_q_ref"][start:end] = np.diff(result["actions"][start:end], axis=0, prepend=result["actions"][start:start + 1])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    manifest = {"source_path": str(source), "source_size": source.stat().st_size, "source_sha256": digest(source), "target_transitions": args.target_transitions, "actual_transitions": total, "episodes": len(selected), "seed": args.seed}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.keep_staging:
        shutil.rmtree(staging)


if __name__ == "__main__":
    main()
