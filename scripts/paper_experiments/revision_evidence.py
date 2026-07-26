"""Assemble the paper-revision evidence without rerunning the frozen benchmarks."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.paper_experiments.evaluation import latency_recovery, summarize_arrays
from scripts.paper_experiments.merge_mpc_ik_results import perturbation_family
from scripts.experiment_utils.bootstrap import paired_bootstrap_rows


TIMING_ARRAYS = {
    "solve": ("planning_time", "mpc_replanned"),
    "e2e": ("planner_end_to_end_latency_s", None),
    "control_compute": ("control_step_wall_time", None),
    "control_period": ("actual_control_period_s", None),
    "wakeup_lateness": ("control_wakeup_lateness_s", None),
    "start_jitter": ("control_start_jitter_s", None),
    "packet_age": ("packet_age", "nonnegative"),
    "safety_projection_offset": ("safety_projection_offset", "absolute"),
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_fullvirtual_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    from scripts.paper_experiments.workflow import suite_cases

    cases = [case for case in suite_cases(manifest, "ablation") if case["label"] == "FullVirtual"]
    keys = {(case["trajectory"], int(case["seed"])) for case in cases}
    if len(cases) != 20 or len(keys) != 20:
        raise ValueError(f"Expected 20 unique FullVirtual cases, found {len(cases)}/{len(keys)}")
    return cases


def replace_fullvirtual(
    original_entries: list[dict[str, Any]], fresh_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    fresh = {(row["trajectory"], int(row["seed"])): row for row in fresh_entries}
    if len(fresh) != 20:
        raise ValueError(f"Fresh FullVirtual matrix must contain 20 cases, found {len(fresh)}")
    output = []
    for entry in original_entries:
        if entry["label"] == "FullVirtual":
            output.append({**fresh[(entry["trajectory"], int(entry["seed"]))], "suite": entry["suite"]})
        else:
            output.append(entry)
    keys = [(row["label"], row["trajectory"], int(row["seed"])) for row in output]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate cases after FullVirtual replacement")
    return output


def pooled_timing(entries: list[dict[str, Any]], label: str) -> dict[str, Any]:
    selected = [entry for entry in entries if entry["label"] == label]
    pools: dict[str, list[np.ndarray]] = defaultdict(list)
    per_case: list[dict[str, Any]] = []
    total_steps = total_solves = total_late = total_expired = total_deadline = 0
    total_fallback = total_projection = 0
    planner_hz: list[float] = []
    for entry in selected:
        with np.load(Path(entry["run_dir"]) / "rollout.npz", allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        events_path = Path(entry["run_dir"]) / "planner_events.jsonl"
        events = [
            json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
        ] if events_path.is_file() else []
        summary = summarize_arrays(label, arrays, events)
        summary.update(trajectory=entry["trajectory"], seed=int(entry["seed"]))
        per_case.append(summary)
        steps = len(np.asarray(arrays.get("actuator_q_ref", [])))
        solves = int(np.asarray(arrays.get("planner_solve_count", 0)).reshape(-1)[0])
        total_steps += steps
        total_solves += solves
        total_late += int(np.asarray(arrays.get("planner_late_drop_count", 0)).reshape(-1)[0])
        total_expired += int(np.asarray(arrays.get("packet_expiration_count", 0)).reshape(-1)[0])
        total_deadline += int(np.sum(np.asarray(arrays.get("control_deadline_miss", [])) != 0))
        total_fallback += int(np.sum(np.asarray(arrays.get("fallback_active", [])) != 0))
        total_projection += int(np.sum(np.asarray(arrays.get("projection_active", [])) != 0))
        planner_hz.append(float(np.asarray(arrays.get("planner_actual_update_rate_hz", np.nan)).reshape(-1)[0]))
        for name, (key, selector) in TIMING_ARRAYS.items():
            values = np.asarray(arrays.get(key, []), dtype=np.float64).reshape(-1)
            if selector == "nonnegative":
                values = values[values >= 0]
            elif selector == "absolute":
                values = np.abs(values)
            elif selector:
                mask = np.asarray(arrays.get(selector, []), dtype=bool).reshape(-1)
                values = values[mask] if len(mask) == len(values) else values
            pools[name].append(values[np.isfinite(values)])
    result: dict[str, Any] = {
        "label": label, "n_cases": len(selected), "total_steps": total_steps,
        "planner_hz_mean": float(np.nanmean(planner_hz)),
        "planner_hz_std": float(np.nanstd(planner_hz, ddof=1)),
        "late_packet_count": total_late,
        "late_packet_rate": total_late / total_solves,
        "packet_expiration_count": total_expired,
        "fallback_duty_cycle": total_fallback / total_steps,
        "control_deadline_miss_count": total_deadline,
        "control_deadline_miss_rate": total_deadline / total_steps,
        "projection_activation_rate": total_projection / total_steps,
    }
    for name, chunks in pools.items():
        values = np.concatenate(chunks) if chunks else np.empty(0)
        for percentile in (50, 95, 99):
            result[f"{name}_p{percentile}"] = float(np.percentile(values, percentile))
        result[f"{name}_max"] = float(np.max(values))
    worst = max(per_case, key=lambda row: float(row["tcp_rmse_m"]))
    result["worst_tracking_case"] = {
        "trajectory": worst["trajectory"], "seed": worst["seed"],
        "tcp_rmse_m": worst["tcp_rmse_m"],
    }
    return result


def _summary_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        with np.load(Path(entry["run_dir"]) / "rollout.npz", allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        row = summarize_arrays(entry["label"], arrays)
        row.update(label=entry["label"], trajectory=entry["trajectory"], seed=int(entry["seed"]))
        row["case_id"] = f"{entry['trajectory']}:{entry['seed']}"
        rows.append(row)
    return rows


def latency_recovery_with_ci(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    """Case-bootstrap the pre-registered recovery ratio without resampling ticks."""
    grouped: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row["trajectory"]), int(row["seed"]))][str(row["label"])] = float(row["tcp_rmse_m"])
    by_trajectory: dict[str, list[float]] = defaultdict(list)
    for (trajectory, _), methods in grouped.items():
        if {"NaiveDelayed", "FullVirtual", "IdealZeroDelay"}.issubset(methods):
            denominator = methods["NaiveDelayed"] - methods["IdealZeroDelay"]
            if denominator > 1e-6:
                by_trajectory[trajectory].append(
                    (methods["NaiveDelayed"] - methods["FullVirtual"]) / denominator
                )
    rng = np.random.default_rng(seed)

    def report(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
        return {
            "n": int(len(array)),
            "mean": float(array.mean()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        }

    flattened = [value for values in by_trajectory.values() for value in values]
    return {
        "definition": "(NaiveDelayed - FullVirtual) / (NaiveDelayed - IdealZeroDelay)",
        "epsilon_m": 1e-6,
        "pooled": report(flattened),
        "by_trajectory": {
            trajectory: report(values)
            for trajectory, values in sorted(by_trajectory.items())
        },
        "legacy_point_estimate": latency_recovery(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", required=True)
    parser.add_argument("--fresh-full-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    paper = Path(args.paper_root).resolve()
    fresh = Path(args.fresh_full_root).resolve()
    output = Path(args.output_dir).resolve()
    fresh_entries = _load(fresh / "runs/indexes/fullvirtual_frozen.json")["entries"]
    assembled: dict[str, list[dict[str, Any]]] = {}
    for suite in ("main", "ablation"):
        original = _load(paper / f"runs/indexes/{suite}.json")["entries"]
        assembled[suite] = replace_fullvirtual(original, fresh_entries)
        _write_json(output / f"indexes/{suite}.json", {"suite": suite, "entries": assembled[suite]})
        _write_csv(output / f"summaries/{suite}.csv", _summary_rows(assembled[suite]))
    main_rows = _summary_rows(assembled["main"])
    comparisons = {
        "ThreadedASAP_minus_FullVirtual": paired_bootstrap_rows(
            main_rows, left="FullVirtual", right="ThreadedASAP",
            metrics=("tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad"),
            samples=args.bootstrap_samples, seed=20260724
        )
    }
    _write_json(output / "statistics/threaded_vs_fullvirtual.json", comparisons)
    _write_json(
        output / "statistics/latency_recovery_with_ci.json",
        latency_recovery_with_ci(main_rows, args.bootstrap_samples, 20261001),
    )
    _write_json(output / "statistics/threaded_realtime_nominal.json", pooled_timing(assembled["main"], "ThreadedASAP"))
    ablation_rows = _summary_rows(assembled["ablation"])
    ablations = {}
    for offset, variant in enumerate(("NoFutureAlignment", "NoReanchor", "NoFeedback")):
        ablations[f"{variant}_minus_FullVirtual"] = paired_bootstrap_rows(
            ablation_rows, left="FullVirtual", right=variant,
            metrics=("tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad"),
            samples=args.bootstrap_samples, seed=20260730 + offset
        )
    _write_json(output / "statistics/ablation_frozen_matrix.json", ablations)
    print(f"Saved revised main/ablation evidence to {output}")


if __name__ == "__main__":
    main()
