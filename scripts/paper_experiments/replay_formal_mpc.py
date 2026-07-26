"""Replay the frozen GRU on commands actually executed in nominal formal MPC runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_dynamics.rollout import load_dynamics_bundle
from mpc.replay_diagnostics import replay_executed_commands


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mpc-root", required=True)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--normalizer", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--device", default="cuda")
    value.add_argument("--horizons", default="1,5,10,20")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _nominal_runs(root: Path) -> list[dict[str, Any]]:
    output = []
    for path in root.glob("**/run_fingerprint.json"):
        fingerprint = json.loads(path.read_text(encoding="utf-8"))
        payload = fingerprint.get("payload", {})
        if payload.get("kind") != "delay_aware_mpc_robustness":
            continue
        if payload.get("condition", {}).get("name") != "nominal":
            continue
        output.append({
            "run_dir": path.parent,
            "method": payload["method"],
            "trajectory": payload["reference_type"],
            "seed": int(payload["run_config"]["seed"]),
            "fingerprint": fingerprint["sha256"],
        })
    return sorted(output, key=lambda row: (row["method"], row["trajectory"], row["seed"]))


def main() -> None:
    args = parser().parse_args()
    horizons = sorted({int(value) for value in args.horizons.split(",") if value})
    maximum = max(horizons)
    device = torch.device(args.device)
    bundle = load_dynamics_bundle(args.checkpoint, args.normalizer, "gru", 6, device, history_len=16)
    rows: list[dict[str, Any]] = []
    for index, run in enumerate(_nominal_runs(Path(args.mpc_root).resolve()), start=1):
        with np.load(run["run_dir"] / "rollout.npz", allow_pickle=False) as archive:
            actual = np.asarray(archive["actual_states"], dtype=np.float32)
            next_states = np.asarray(archive["next_states"], dtype=np.float32)
            q_ref = np.asarray(archive["actuator_q_ref"], dtype=np.float32)
            velocity = np.asarray(archive["command_velocity"], dtype=np.float32)
            acceleration = np.asarray(archive["command_acceleration"], dtype=np.float32)
        states_history = [*actual, next_states[-1]]
        q_ref_history = [*q_ref, q_ref[-1]]
        replay = replay_executed_commands(
            model=bundle.model,
            normalizer=bundle.normalizer,
            model_type=bundle.model_type,
            state_dim=bundle.state_dim,
            target_mode=bundle.target_mode,
            control_dt=bundle.control_dt,
            history_len=bundle.history_len,
            states_history=states_history,
            q_ref_history=q_ref_history,
            executed_q_ref=list(q_ref),
            horizon=maximum,
            device=device,
            rollout_batch_size=128,
            command_velocity=list(velocity),
            command_acceleration=list(acceleration),
        )
        for horizon in horizons:
            q_values = replay["replay_q_error_norm"][:, horizon - 1]
            dq_values = replay["replay_dq_error_norm"][:, horizon - 1]
            q_values = q_values[np.isfinite(q_values)]
            dq_values = dq_values[np.isfinite(dq_values)]
            rows.append({
                **{key: run[key] for key in ("method", "trajectory", "seed", "fingerprint")},
                "horizon": horizon, "n_anchors": len(q_values),
                "q_error_norm_rmse": float(np.sqrt(np.mean(np.square(q_values)))),
                "q_error_norm_p95": float(np.percentile(q_values, 95)),
                "dq_error_norm_rmse": float(np.sqrt(np.mean(np.square(dq_values)))),
                "dq_error_norm_p95": float(np.percentile(dq_values, 95)),
            })
        print(f"[{index}/80] replayed {run['method']}/{run['trajectory']}/seed_{run['seed']}")
    output = Path(args.output_dir).resolve()
    _write_csv(output / "formal_mpc_replay.csv", rows)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["horizon"])].append(row)
    aggregate = []
    for key, members in sorted(grouped.items()):
        item: dict[str, Any] = {"method": key[0], "horizon": key[1], "n_cases": len(members)}
        for metric in ("q_error_norm_rmse", "q_error_norm_p95", "dq_error_norm_rmse", "dq_error_norm_p95"):
            values = np.asarray([row[metric] for row in members])
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1))
            item[f"{metric}_p95"] = float(np.percentile(values, 95))
            item[f"{metric}_worst"] = float(np.max(values))
        aggregate.append(item)
    _write_csv(output / "formal_mpc_replay_aggregate.csv", aggregate)


if __name__ == "__main__":
    main()
