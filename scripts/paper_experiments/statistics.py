"""Pre-registered paper statistics beyond the generic per-suite summaries."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main_endpoint(rows: list[dict[str, Any]], output: Path, samples: int) -> None:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["trajectory"]), int(row["seed"]))][str(row["label"])] = row
    paired: list[dict[str, Any]] = []
    for (trajectory, seed), methods in sorted(grouped.items()):
        if {"NaiveDelayed", "FullVirtual"}.issubset(methods):
            naive = methods["NaiveDelayed"]
            full = methods["FullVirtual"]
            paired.append({
                "trajectory": trajectory, "seed": seed,
                "naive_tcp_rmse_m": naive["tcp_rmse_m"],
                "full_tcp_rmse_m": full["tcp_rmse_m"],
                "delta_full_minus_naive_m": full["tcp_rmse_m"] - naive["tcp_rmse_m"],
                "improvement_fraction": (
                    (naive["tcp_rmse_m"] - full["tcp_rmse_m"]) / naive["tcp_rmse_m"]
                    if naive["tcp_rmse_m"] > 0 else float("nan")
                ),
            })
    by_trajectory: list[dict[str, Any]] = []
    for index, trajectory in enumerate(sorted({row["trajectory"] for row in paired})):
        members = [row for row in paired if row["trajectory"] == trajectory]
        values = np.asarray([row["delta_full_minus_naive_m"] for row in members])
        low, high = _bootstrap(values, samples, 20260722 + index)
        by_trajectory.append({
            "trajectory": trajectory, "n": len(values),
            "mean_delta_full_minus_naive_m": float(np.mean(values)),
            "ci95_low_m": low, "ci95_high_m": high,
            "mean_improvement_fraction": float(np.mean([row["improvement_fraction"] for row in members])),
        })
    all_values = np.asarray([row["delta_full_minus_naive_m"] for row in paired])
    low, high = _bootstrap(all_values, samples, 20260730)
    pooled = {
        "n": len(all_values),
        "mean_delta_full_minus_naive_m": float(np.mean(all_values)),
        "ci95": [low, high],
        "mean_improvement_fraction": float(np.mean([row["improvement_fraction"] for row in paired])),
    }
    worst: list[dict[str, Any]] = []
    for trajectory in sorted({row["trajectory"] for row in paired}):
        members = [row for row in paired if row["trajectory"] == trajectory]
        for label, key in (("best", "full_tcp_rmse_m"), ("worst", "full_tcp_rmse_m")):
            selected = (min if label == "best" else max)(members, key=lambda row: row[key])
            worst.append({"trajectory": trajectory, "selection": label, **selected})
        ordered = sorted(members, key=lambda row: row["full_tcp_rmse_m"])
        worst.append({"trajectory": trajectory, "selection": "median", **ordered[len(ordered) // 2]})
    write_csv(output / "primary_endpoint_by_trajectory.csv", by_trajectory)
    write_csv(output / "primary_endpoint_worst_seed.csv", worst)
    (output / "primary_endpoint_pooled.json").write_text(json.dumps(pooled, indent=2, sort_keys=True) + "\n")


def delay_sweep_report(rows: list[dict[str, Any]], output: Path, samples: int) -> None:
    slopes: list[dict[str, Any]] = []
    monotonicity: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["delay_protocol"]), str(row["trajectory"]), int(row["seed"]))].append(row)
    for (protocol, trajectory, seed), members in sorted(grouped.items()):
        members.sort(key=lambda row: int(row["delay_steps"]))
        delays = np.asarray([int(row["delay_steps"]) for row in members], dtype=np.float64)
        errors = np.asarray([float(row["tcp_rmse_m"]) for row in members])
        slope = float(np.polyfit(delays, errors, 1)[0]) if len(delays) >= 2 else float("nan")
        slopes.append({"delay_protocol": protocol, "trajectory": trajectory, "seed": seed, "tcp_rmse_slope_m_per_step": slope})
        diffs = np.diff(errors)
        monotonicity.append({
            "delay_protocol": protocol, "trajectory": trajectory, "seed": seed,
            "nondecreasing_adjacent_fraction": float(np.mean(diffs >= 0)) if diffs.size else float("nan"),
            "worst_tcp_rmse_m": float(np.max(errors)),
            "worst_delay_steps": int(delays[int(np.argmax(errors))]),
        })
    aggregate: list[dict[str, Any]] = []
    for index, key in enumerate(sorted({(row["delay_protocol"], row["trajectory"]) for row in slopes})):
        values = np.asarray([row["tcp_rmse_slope_m_per_step"] for row in slopes if (row["delay_protocol"], row["trajectory"]) == key])
        low, high = _bootstrap(values, samples, 20260800 + index)
        aggregate.append({
            "delay_protocol": key[0], "trajectory": key[1], "n": len(values),
            "slope_mean_m_per_step": float(np.mean(values)), "ci95_low": low, "ci95_high": high,
        })
    write_csv(output / "delay_sweep_component_slopes.csv", slopes)
    write_csv(output / "delay_sweep_component_slope_aggregate.csv", aggregate)
    write_csv(output / "delay_sweep_monotonicity.csv", monotonicity)


def projection_report(rows: list[dict[str, Any]], output: Path, samples: int, margin: float) -> None:
    comparisons: dict[str, Any] = {"noninferiority_margin_fraction": margin, "sets": {}}
    for set_index, evaluation_set in enumerate(("common_d", "deployed")):
        members = [row for row in rows if row.get("evaluation_set") == evaluation_set]
        grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in members:
            grouped[(str(row["trajectory"]), int(row["seed"]))][str(row["variant"])] = row
        relative = []
        absolute = []
        for methods in grouped.values():
            if {"FullCompiled", "TwoStageCompiled"}.issubset(methods):
                full = float(methods["FullCompiled"]["tcp_rmse_m"])
                two_stage = float(methods["TwoStageCompiled"]["tcp_rmse_m"])
                absolute.append(two_stage - full)
                relative.append((two_stage - full) / full)
        values = np.asarray(relative, dtype=np.float64)
        rng = np.random.default_rng(20260900 + set_index)
        draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
        upper = float(np.quantile(draws, 0.95))
        comparisons["sets"][evaluation_set] = {
            "n": len(values),
            "mean_relative_delta": float(np.mean(values)),
            "mean_absolute_delta_m": float(np.mean(absolute)),
            "one_sided_95_upper_relative_delta": upper,
            "noninferior": upper <= margin,
        }
    (output / "projection_noninferiority.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
