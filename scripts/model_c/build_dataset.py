"""Merge immutable Model-C shards into a compact training NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = ("states", "actions", "next_states", "episode_id", "split_group_id", "source_id", "valid_target", "context_only")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact C1/C2 training dataset from replay and shards.")
    parser.add_argument("--input", action="append", required=True, help="Replay NPZ or a directory containing transitions_*.npz; repeatable.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--selection_manifest", default=None, help="Optional Round-2 selection JSON; filters only branch rows in shard inputs.")
    return parser.parse_args()


def paths(values: list[str]) -> list[tuple[Path, int]]:
    result: list[tuple[Path, int]] = []
    for source_index, value in enumerate(values):
        path = Path(value)
        path = ROOT / path if not path.is_absolute() else path
        result.extend((item, source_index) for item in (sorted(path.glob("transitions_*.npz")) if path.is_dir() else [path]))
    return result


def main() -> None:
    args = parse_args()
    chunks = paths(args.input)
    if not chunks:
        raise ValueError("No input datasets found")
    selected_branch_ids: set[int] | None = None
    selected_input_dir: Path | None = None
    selected_branch_kind_id: int | None = None
    if args.selection_manifest:
        with _resolve(args.selection_manifest).open(encoding="utf-8") as file:
            selection = json.load(file)
        selected_branch_ids = {int(value) for value in selection["selected_branch_ids"]}
        selected_input_dir = Path(selection["input_dir"]).resolve()
        selected_branch_kind_id = selection.get("branch_kind_id")
        if not selected_branch_ids:
            raise ValueError("selection manifest has no selected branches")
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ("states", "actions", "next_states", "episode_ids", "split_group_ids", "source_ids", "valid_target", "context_only")}
    episode_offset = 0
    group_offset = 0
    input_source: dict[str, int] = {}
    for path, source_offset in chunks:
        with np.load(path) as data:
            state_keys = {"states", "actions", "next_states"}
            if not state_keys.issubset(data.files):
                raise KeyError(f"{path} is missing required state arrays")
            episode = data["episode_ids"] if "episode_ids" in data.files else data["episode_id"]
            group = data["split_group_ids"] if "split_group_ids" in data.files else (data["split_group_id"] if "split_group_id" in data.files else episode)
            mask = np.ones(len(episode), dtype=bool)
            if selected_branch_ids is not None and "branch_id" in data.files and path.parent.resolve() == selected_input_dir:
                mask = np.isin(data["branch_id"], np.fromiter(selected_branch_ids, dtype=np.int64))
                if selected_branch_kind_id is not None:
                    if "branch_kind_id" not in data.files:
                        raise KeyError(f"{path} lacks branch_kind_id required by selection manifest")
                    mask &= np.asarray(data["branch_kind_id"], dtype=np.int64) == int(selected_branch_kind_id)
                # Shards may contain the main trajectory plus the selected
                # branch.  Never keep main rows in a Round-2 selected build.
                if not np.any(mask):
                    continue
            if not np.any(mask):
                continue
            source = np.full(np.count_nonzero(mask), source_offset, dtype=np.int16)
            valid = (data["valid_target"] if "valid_target" in data.files else np.ones(len(episode), dtype=np.int8))[mask]
            context = (data["context_only"] if "context_only" in data.files else np.zeros(len(episode), dtype=np.int8))[mask]
            for key in state_keys:
                parts[key].append(data[key][mask])
            episode = episode[mask]; group = group[mask]
            _, episode_inverse = np.unique(episode, return_inverse=True)
            _, group_inverse = np.unique(group, return_inverse=True)
            parts["episode_ids"].append(episode_inverse.astype(np.int64) + episode_offset)
            parts["split_group_ids"].append(group_inverse.astype(np.int64) + group_offset)
            parts["source_ids"].append(source)
            parts["valid_target"].append(valid.astype(np.int8)); parts["context_only"].append(context.astype(np.int8))
            episode_offset += int(len(np.unique(episode)))
            group_offset += int(len(np.unique(group)))
            input_source[str(path)] = source_offset
    if not parts["states"]:
        raise RuntimeError("No rows selected from inputs")
    merged = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    merged["q_ref"] = merged["actions"].copy()
    merged["delta_q_ref"] = np.zeros_like(merged["actions"])
    for episode in np.unique(merged["episode_ids"]):
        indices = np.flatnonzero(merged["episode_ids"] == episode)
        merged["delta_q_ref"][indices] = np.diff(merged["actions"][indices], axis=0, prepend=merged["actions"][indices[:1]])
    output = Path(args.output_path); output = ROOT / output if not output.is_absolute() else output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    output.with_suffix(".manifest.json").write_text(json.dumps({"inputs": [str(path) for path, _ in chunks], "source_ids": input_source, "samples": int(len(merged["states"])), "episodes": int(len(np.unique(merged["episode_ids"]))), "selection_manifest": args.selection_manifest}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
