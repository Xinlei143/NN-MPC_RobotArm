#!/usr/bin/env python3
"""Assemble shadow-run evidence for delay and OOD calibration.

Inputs: two or more shadow rollout NPZs (run_real_cem_mpc.py save_mpc_run
outputs, one per played reference) plus the frozen Model-A collection.

Outputs (into --output-dir):
  merged_shadow.npz  - concatenated planner_end_to_end_latency_s /
                       planner_late_drop / packet_expired and the OOD token
                       arrays (executed/selected_action/predicted_state),
                       consumed by scripts/calibrate_real_delay.py (>=2000
                       samples -> p99.5) and by this script's OOD side.
  OOD_TOKENS.npz     - the five arrays scripts/calibrate_real_ood.py requires:
                       training_tokens / validation_tokens rebuilt from the
                       collection with the gate model's group splits
                       (gru_20260807_130024/config.yaml: train groups 0..37
                       stride 2, validation groups 38..42 stride 1, test
                       groups 43..47 held out; valid_target rows only), plus
                       the three runtime arrays from merged_shadow.npz.

Token semantics (15-dim rows of [state, command]):
  - collection: [measured states[t] (q+dq), actions[t] (== q_ref, position
    control, train_dynamics.py:241)]
  - runtime executed: [measured state at t, previous transmitted q_ref]
  - runtime future (selected_action == predicted_state, the array the
    adapter OOD-checks): [predicted_state; selected q_ref] rows, flattened
    from (packets, 6, 15) at H=6.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

COLLECTION_DEFAULT = ROOT / "outputs/hardware/so101_pre_mpc/20260804_e_stage/model_a_workspace_48x15min.npz"
TRAIN_GROUPS = range(0, 38)   # 0..37, matches gate config train split
VAL_GROUPS = range(38, 43)    # 38..42, matches config validation_group_ids
TEST_GROUPS = range(43, 48)   # 43..47 held out entirely (config test_group_ids)
TRAIN_STRIDE = 2              # config train_sample_stride
VAL_STRIDE = 1                # config val_sample_stride


def merge_runs(paths: list[Path]) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            for key in ("planner_end_to_end_latency_s", "planner_late_drop", "packet_expired",
                        "executed_tokens", "selected_action_tokens", "predicted_state_tokens"):
                if key not in archive.files:
                    raise KeyError(f"{path} is missing {key!r} (is it a patched shadow rollout?)")
                arrays.setdefault(key, []).append(np.asarray(archive[key]))
    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    for key in ("planner_late_drop", "packet_expired"):
        merged[key] = merged[key].astype(bool)
    return merged


def collection_tokens(collection_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(collection_path, mmap_mode="r") as data:
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        groups = np.asarray(data["split_group_ids"])
        valid = np.asarray(data["valid_target"], dtype=bool)
    finite = np.isfinite(states).all(axis=1) & np.isfinite(actions).all(axis=1)
    keep = valid & finite

    def rows(mask: np.ndarray, stride: int) -> np.ndarray:
        indices = np.flatnonzero(keep & mask)[::stride]
        tokens = np.concatenate((states[indices], actions[indices]), axis=1).astype(np.float32)
        return tokens

    train_mask = np.isin(groups, list(TRAIN_GROUPS))
    val_mask = np.isin(groups, list(VAL_GROUPS))
    if not np.any(train_mask) or not np.any(val_mask):
        raise RuntimeError("collection group masks are empty; check split_group_ids")
    return rows(train_mask, TRAIN_STRIDE), rows(val_mask, VAL_STRIDE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", nargs="+", type=Path,
                        help="Shadow rollout NPZs (at least two; >=2000 latency samples required).")
    parser.add_argument("--collection", type=Path, default=COLLECTION_DEFAULT,
                        help="Frozen Model-A collection NPZ.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.rollouts) < 2:
        parser.error("at least two shadow rollouts are needed to merge calibration evidence")
    if not args.collection.exists():
        parser.error(f"collection not found: {args.collection}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_runs(args.rollouts)
    latency = merged["planner_end_to_end_latency_s"]
    latency = latency[np.isfinite(latency)]
    print(f"merged shadow: {len(args.rollouts)} runs, {latency.size} latency samples "
          f"(need >=2000 for the p99.5 gate), late_drop={merged['planner_late_drop'].mean():.4%}, "
          f"expiry={merged['packet_expired'].mean():.4%}")
    if latency.size < 2000:
        print("WARNING: <2000 latency samples; calibrate_real_delay.py will fall back to "
              "p99+guard and the active-MPC delay gate will refuse. Re-run shadow on another reference.")

    merged_path = args.output_dir / "merged_shadow.npz"
    np.savez_compressed(merged_path, **merged)
    print(f"merged shadow: {merged_path}")

    training_tokens, validation_tokens = collection_tokens(args.collection)
    for name, tokens in (("training", training_tokens), ("validation", validation_tokens)):
        if tokens.size == 0 or tokens.shape[1] != 15:
            raise RuntimeError(f"{name}_tokens have unexpected shape {tokens.shape}")
    ood_path = args.output_dir / "OOD_TOKENS.npz"
    np.savez_compressed(ood_path,
                        training_tokens=training_tokens,
                        validation_tokens=validation_tokens,
                        executed_tokens=merged["executed_tokens"],
                        selected_action_tokens=merged["selected_action_tokens"],
                        predicted_state_tokens=merged["predicted_state_tokens"])
    print(f"OOD tokens: {ood_path}")
    print(f"  training={training_tokens.shape} validation={validation_tokens.shape} "
          f"executed={merged['executed_tokens'].shape} "
          f"selected/predicted={merged['selected_action_tokens'].shape}")


if __name__ == "__main__":
    main()
