"""Merge the frozen 720 MPC and 540 Direct-IK results with cluster bootstrap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("tcp_rmse_m", "tcp_p95_m", "orientation_rmse_rad", "joint_rmse_rad")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mpc-root", required=True)
    value.add_argument("--ik-root", required=True)
    value.add_argument("--output-dir", required=True)
    value.add_argument("--bootstrap-samples", type=int, default=10000)
    value.add_argument("--bootstrap-seed", type=int, default=20260722)
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fingerprints(root: Path, kind: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in root.glob("**/run_fingerprint.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        payload = data.get("payload", {})
        if payload.get("kind") != kind:
            continue
        method = str(payload.get("method", payload.get("projection", "")))
        condition = str(payload.get("condition", {}).get("name", ""))
        case_id = str(payload.get("case_id", ""))
        output[(method, condition, case_id)] = {
            "reference_sha256": payload.get("reference", {}).get("sha256", ""),
            "run_dir": str(path.parent),
        }
    return output


def _content_hash(run_dir: Path) -> str:
    digest = hashlib.sha256()
    with np.load(run_dir / "rollout.npz", allow_pickle=False) as archive:
        for key in ("actual_states", "ee_position_errors", "ee_orientation_errors"):
            value = np.ascontiguousarray(archive[key])
            digest.update(key.encode())
            digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _normalize(
    rows: list[dict[str, str]],
    fingerprints: dict[tuple[str, str, str], dict[str, Any]],
    cohort: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        method = row.get("method", row.get("projection", ""))
        lookup_method = method if cohort == "mpc" else row.get("projection", "")
        metadata = fingerprints[(lookup_method, row["condition"], row["case_id"])]
        seed = int(row["case_id"].rsplit("_", 1)[-1])
        item: dict[str, Any] = {
            **row,
            "cohort": cohort,
            "method": method,
            "seed": seed,
            "trajectory": row["reference_type"],
            "reference_sha256": metadata["reference_sha256"],
            "run_dir": metadata["run_dir"],
        }
        for metric in METRICS:
            item[metric] = float(row[metric])
        output.append(item)
    return output


def _deduplicate_ik(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(
            row["method"], row["trajectory"], row["condition"],
            row["perturbation"], row["level"], row["reference_sha256"],
        )].append(row)
    unique: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for member in members:
            by_hash[_content_hash(Path(member["run_dir"]))].append(member)
        deterministic = len(by_hash) == 1
        selected = [min(members, key=lambda row: int(row["seed"]))] if deterministic else members
        unique.extend(selected)
        report.append({
            "method": key[0], "trajectory": key[1], "condition": key[2],
            "level": key[4], "input_rows": len(members), "unique_content": len(by_hash),
            "output_rows": len(selected), "deterministic": deterministic,
        })
        for member in selected:
            member["deterministic_baseline"] = deterministic
    return unique, report


def _pairs(mpc: list[dict[str, Any]], ik: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for baseline in ik:
        candidates = [
            row for row in mpc
            if row["trajectory"] == baseline["trajectory"]
            and row["condition"] == baseline["condition"]
            and row["perturbation"] == baseline["perturbation"]
            and row["level"] == baseline["level"]
            and row["reference_sha256"] == baseline["reference_sha256"]
            and (baseline["deterministic_baseline"] or row["seed"] == baseline["seed"])
        ]
        for row in candidates:
            item = {
                "trajectory": row["trajectory"], "condition": row["condition"],
                "perturbation": row["perturbation"], "level": row["level"],
                "seed": row["seed"], "reference_sha256": row["reference_sha256"],
                "mpc_method": row["method"], "ik_method": baseline["method"],
                "deterministic_ik": baseline["deterministic_baseline"],
            }
            for metric in METRICS:
                item[f"mpc_{metric}"] = row[metric]
                item[f"ik_{metric}"] = baseline[metric]
                item[f"delta_{metric}"] = row[metric] - baseline[metric]
            output.append(item)
    return output


def _cluster_bootstrap(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {"samples": samples, "seed": seed, "comparisons": {}}
    comparisons = sorted({(row["mpc_method"], row["ik_method"]) for row in rows})
    for mpc_method, ik_method in comparisons:
        members = [row for row in rows if row["mpc_method"] == mpc_method and row["ik_method"] == ik_method]
        clusters: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in members:
            clusters[(row["trajectory"], row["condition"])].append(row)
        keys = list(clusters)
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            observed = np.asarray([float(row[f"delta_{metric}"]) for row in members])
            draws = np.empty(samples, dtype=np.float64)
            for draw in range(samples):
                chosen = [keys[index] for index in rng.integers(0, len(keys), len(keys))]
                cluster_means = []
                for key in chosen:
                    values = np.asarray([float(row[f"delta_{metric}"]) for row in clusters[key]])
                    cluster_means.append(float(np.mean(values[rng.integers(0, len(values), len(values))])))
                draws[draw] = float(np.mean(cluster_means))
            metrics[metric] = {
                "n_pairs": len(observed), "n_clusters": len(keys),
                "mean_delta_mpc_minus_ik": float(np.mean(observed)),
                "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            }
        report["comparisons"][f"{mpc_method}_minus_{ik_method}"] = metrics
    return report


def main() -> None:
    args = parser().parse_args()
    mpc_root = Path(args.mpc_root).resolve()
    ik_root = Path(args.ik_root).resolve()
    output = Path(args.output_dir).resolve()
    mpc_rows = _normalize(
        _read_csv(mpc_root / "delay_aware_mpc_robustness_summary.csv"),
        _fingerprints(mpc_root, "delay_aware_mpc_robustness"),
        "mpc",
    )
    ik_rows = _normalize(
        _read_csv(ik_root / "direct_ik_robustness_summary.csv"),
        _fingerprints(ik_root, "direct_ik_robustness"),
        "ik",
    )
    unique_ik, deduplication = _deduplicate_ik(ik_rows)
    pairs = _pairs(mpc_rows, unique_ik)
    _write_csv(output / "mpc_vs_projected_ik_pairs.csv", pairs)
    aggregate: list[dict[str, Any]] = []
    for key in sorted({(row["mpc_method"], row["ik_method"], row["condition"]) for row in pairs}):
        members = [row for row in pairs if (row["mpc_method"], row["ik_method"], row["condition"]) == key]
        item = dict(zip(("mpc_method", "ik_method", "condition"), key))
        item["n"] = len(members)
        for metric in METRICS:
            values = np.asarray([row[f"delta_{metric}"] for row in members])
            item[f"delta_{metric}_mean"] = float(np.mean(values))
            item[f"delta_{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(item)
    _write_csv(output / "mpc_vs_projected_ik_by_condition.csv", aggregate)
    (output / "mpc_vs_projected_ik_bootstrap.json").write_text(
        json.dumps(_cluster_bootstrap(pairs, args.bootstrap_samples, args.bootstrap_seed), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "baseline_deduplication_report.json").write_text(
        json.dumps({"groups": deduplication}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(pairs)} MPC/IK pairs to {output}")


if __name__ == "__main__":
    main()
