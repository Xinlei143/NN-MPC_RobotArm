"""Select continuous counterfactual branches for Round-2 collection.

Labels are intentionally non-exclusive.  Selection assigns a branch to at
most one quota bucket, so that a branch is never duplicated merely because it
is both difficult to predict and close to a residual bound.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SELECTED, BASELINE, ALTERNATIVE = 1, 2, 4
ACTIVATION_PROJECTED = 1
DEFAULT_WEIGHTS = {
    "high_model_error": 0.35,
    "high_tracking_error": 0.25,
    "ranking_flip": 0.20,
    "near_residual_bound": 0.10,
    "recovery_or_fallback": 0.10,
}


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _parse_weights(value: str | None) -> dict[str, float]:
    if value is None:
        return DEFAULT_WEIGHTS.copy()
    parsed: dict[str, float] = {}
    for item in value.split(","):
        key, sep, number = item.partition(":")
        if not sep or key not in DEFAULT_WEIGHTS:
            raise ValueError("--weights must use high_model_error:0.35,...")
        parsed[key] = float(number)
    if set(parsed) != set(DEFAULT_WEIGHTS) or any(weight < 0 for weight in parsed.values()):
        raise ValueError("--weights must specify all five non-negative label weights")
    if not np.isclose(sum(parsed.values()), 1.0):
        raise ValueError("--weights must sum to 1")
    return parsed


def _main_telemetry(root: Path) -> dict[tuple[int, int], dict[str, float]]:
    result: dict[tuple[int, int], dict[str, float]] = {}
    required = {"episode_id", "control_step", "tracking_error", "residual_fraction", "recovery_active", "packet_fallback"}
    for path in sorted(root.glob("transitions_*.npz")):
        with np.load(path) as data:
            if not required.issubset(data.files):
                raise KeyError(f"{path} lacks Round-2 telemetry fields; recollect with collect_model_c_data.py")
            main = np.asarray(data["branch_id"]) < 0 if "branch_id" in data.files else np.ones(len(data["episode_id"]), dtype=bool)
            for index in np.flatnonzero(main):
                step = int(data["control_step"][index])
                if step >= 0:
                    result[(int(data["episode_id"][index]), step)] = {
                        "tracking_error": float(data["tracking_error"][index]),
                        "residual_fraction": float(data["residual_fraction"][index]),
                        "recovery_active": float(data["recovery_active"][index]),
                        "packet_fallback": float(data["packet_fallback"][index]),
                    }
    return result


def _nearest_telemetry(table: dict[tuple[int, int], dict[str, float]], parent: int, step: int) -> dict[str, float]:
    direct = table.get((parent, step))
    if direct is not None:
        return direct
    candidates = [(abs(candidate_step - step), value) for (episode, candidate_step), value in table.items() if episode == parent]
    return min(candidates, key=lambda item: item[0])[1] if candidates else {"tracking_error": np.nan, "residual_fraction": np.nan, "recovery_active": 0.0, "packet_fallback": 0.0}


def _branch_rows(root: Path, telemetry: dict[tuple[int, int], dict[str, float]], branch_kind_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("branches_*.npz")):
        with np.load(path) as data:
            required = {"branch_id", "parent_main_episode_id", "activation_step", "role_mask", "activation_infeasible", "predicted_state_sequence", "realized_state_sequence", "predicted_cost", "realized_cost"}
            if not required.issubset(data.files):
                raise KeyError(f"{path} missing branch diagnostics")
            for index in range(len(data["branch_id"])):
                # Schema v3 explicitly separates strict diagnostics from
                # activation-projected training branches.  Legacy shards are
                # strict-only and must not silently enter C2.
                if "branch_kind_id" not in data.files or int(data["branch_kind_id"][index]) != branch_kind_id:
                    continue
                if int(data["activation_infeasible"][index]):
                    continue
                predicted, realized = data["predicted_state_sequence"][index], data["realized_state_sequence"][index]
                predicted_cost, realized_cost = float(data["predicted_cost"][index]), float(data["realized_cost"][index])
                if not (np.all(np.isfinite(predicted)) and np.all(np.isfinite(realized)) and np.isfinite(predicted_cost) and np.isfinite(realized_cost)):
                    continue
                parent, activation = int(data["parent_main_episode_id"][index]), int(data["activation_step"][index])
                row = {
                    "branch_id": int(data["branch_id"][index]), "parent_main_episode_id": parent,
                    "activation_step": activation, "role_mask": int(data["role_mask"][index]),
                    "candidate_group_id": int(data["candidate_group_id"][index]) if "candidate_group_id" in data.files else parent * 10_000 + activation,
                    "model_error": float(np.mean(np.square(predicted - realized))),
                    "predicted_cost": predicted_cost, "realized_cost": realized_cost,
                }
                row.update(_nearest_telemetry(telemetry, parent, activation))
                rows.append(row)
    return rows


def _label_rows(rows: list[dict[str, Any]], weights: dict[str, float] | None = None) -> None:
    weights = DEFAULT_WEIGHTS if weights is None else weights
    errors = np.asarray([row["model_error"] for row in rows], dtype=float)
    tracking = np.asarray([row["tracking_error"] for row in rows], dtype=float)
    # The high-error labels are sampling strata, not a fixed 80th-percentile
    # anomaly detector.  Their threshold must expose at least their requested
    # quota before overlap with other labels is resolved by _select().
    error_threshold = float(np.quantile(errors, 1.0 - float(weights["high_model_error"])))
    tracking_threshold = (
        float(np.quantile(tracking[np.isfinite(tracking)], 1.0 - float(weights["high_tracking_error"])))
        if np.any(np.isfinite(tracking)) else float("inf")
    )
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = int(row.get("candidate_group_id", int(row["parent_main_episode_id"]) * 10_000 + int(row["activation_step"])))
        groups[group_id].append(row)
    for group in groups.values():
        predicted_best = min(group, key=lambda row: row["predicted_cost"])
        realized_best = min(group, key=lambda row: row["realized_cost"])
        ranking_flip = predicted_best["branch_id"] != realized_best["branch_id"]
        for row in group:
            row["high_model_error"] = bool(row["model_error"] >= error_threshold)
            row["high_tracking_error"] = bool(np.isfinite(row["tracking_error"]) and row["tracking_error"] >= tracking_threshold)
            row["ranking_flip"] = ranking_flip
            row["near_residual_bound"] = bool(np.isfinite(row["residual_fraction"]) and row["residual_fraction"] >= 0.90)
            row["recovery_or_fallback"] = bool(row["recovery_active"] or row["packet_fallback"])


def _select(rows: list[dict[str, Any]], target: int, weights: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    # Ranking is deterministic and lets a rare label fall back to the next
    # category without selecting a discrete transition or duplicate branch.
    scores = {
        "high_model_error": lambda row: (row["model_error"],),
        "high_tracking_error": lambda row: (row["tracking_error"] if np.isfinite(row["tracking_error"]) else -np.inf, row["model_error"]),
        "ranking_flip": lambda row: (abs(row["predicted_cost"] - row["realized_cost"]), row["model_error"]),
        "near_residual_bound": lambda row: (row["residual_fraction"] if np.isfinite(row["residual_fraction"]) else -np.inf, row["model_error"]),
        "recovery_or_fallback": lambda row: (row["tracking_error"] if np.isfinite(row["tracking_error"]) else -np.inf, row["model_error"]),
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    counts: dict[str, int] = {label: 0 for label in weights}
    for label, weight in weights.items():
        quota = int(round(target * weight))
        # Model/tracking strata are quota-ranked rather than capped by a
        # Boolean percentile label.  Otherwise a 35% target can never be met
        # by a top-20% label, and overlap can starve tracking after the model
        # bucket has claimed its candidates.  The rare event strata remain
        # Boolean and fall back explicitly when their events are scarce.
        eligible = (
            (row for row in rows if row["branch_id"] not in selected_ids)
            if label in {"high_model_error", "high_tracking_error"}
            else (row for row in rows if row[label] and row["branch_id"] not in selected_ids)
        )
        candidates = sorted(eligible, key=scores[label], reverse=True)
        for row in candidates[:quota]:
            row = dict(row); row["selection_bucket"] = label
            selected.append(row); selected_ids.add(row["branch_id"]); counts[label] += 1
    if len(selected) < target:
        remaining = sorted((row for row in rows if row["branch_id"] not in selected_ids), key=lambda row: row["model_error"], reverse=True)
        for row in remaining[: target - len(selected)]:
            row = dict(row); row["selection_bucket"] = "fallback_high_model_error"
            selected.append(row); selected_ids.add(row["branch_id"])
        counts["fallback_high_model_error"] = len(selected) - sum(counts.values())
    return selected, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Label counterfactual branches and select Round-2 continuous units.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--target_branches", type=int, required=True)
    parser.add_argument("--branch_kind", choices=("activation_projected",), default="activation_projected")
    parser.add_argument("--weights", default=None, help="Optional five-label target mix; must sum to one.")
    args = parser.parse_args()
    if args.target_branches <= 0:
        raise ValueError("--target_branches must be positive")
    root = _resolve(args.input_dir)
    weights = _parse_weights(args.weights)
    rows = _branch_rows(root, _main_telemetry(root), ACTIVATION_PROJECTED)
    if not rows:
        raise RuntimeError("No feasible finite branches found")
    if args.target_branches > len(rows):
        raise ValueError(f"Requested {args.target_branches} branches, but only {len(rows)} feasible branches are available")
    _label_rows(rows, weights)
    selected, counts = _select(rows, args.target_branches, weights)
    output = _resolve(args.output_path); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": 3, "input_dir": str(root.resolve()), "branch_kind": args.branch_kind, "branch_kind_id": ACTIVATION_PROJECTED, "weights": weights,
        "target_branches": args.target_branches, "available_branches": len(rows),
        "selected_branches": selected, "selected_branch_ids": [row["branch_id"] for row in selected],
        "selection_counts": counts, "all_branch_labels": rows,
        "note": "late-drop is logged as packet_fallback but is not a primary sampling bucket.",
    }, indent=2, allow_nan=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
