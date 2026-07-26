"""Audit historical ROBIO MPC and IK evidence without modifying source runs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiment_utils import file_identity, load_json


MPC_KIND = "delay_aware_mpc_robustness"
IK_KIND = "direct_ik_robustness"
PRIMARY_ARRAYS = ("actual_states", "q_des", "ee_position_errors")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mpc-roots", nargs="+", required=True)
    value.add_argument("--ik-roots", nargs="+", required=True)
    value.add_argument("--target-manifest", default=None)
    value.add_argument("--output-dir", required=True)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _scalar(archive: Any, key: str, default: Any = "") -> Any:
    if key not in archive:
        return default
    return np.asarray(archive[key]).reshape(-1)[0].item()


def _kind(fingerprint: dict[str, Any]) -> str:
    return str(fingerprint.get("payload", {}).get("kind", ""))


def _discover(roots: list[Path], expected_kind: str) -> list[Path]:
    output: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for fingerprint in root.glob("**/run_fingerprint.json"):
            fingerprint = fingerprint.resolve()
            if fingerprint in seen:
                continue
            try:
                payload = json.loads(fingerprint.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                output.append(fingerprint)
                seen.add(fingerprint)
                continue
            if _kind(payload) == expected_kind:
                output.append(fingerprint)
                seen.add(fingerprint)
    return sorted(output)


def _identity(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return str(value.get("sha256", "")) if isinstance(value, dict) else ""


def audit_run(fingerprint_path: Path, cohort: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    run_dir = fingerprint_path.parent
    try:
        fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        payload = fingerprint["payload"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return {"cohort": cohort, "run_dir": str(run_dir), "status": "invalid_fingerprint"}, [{
            "cohort": cohort, "run_dir": str(run_dir), "issue": "invalid_fingerprint", "detail": str(exc),
        }]

    required = ("rollout.npz", "rollout.csv", "run_summary.json")
    for name in required:
        if not (run_dir / name).is_file():
            issues.append({"cohort": cohort, "run_dir": str(run_dir), "issue": "missing_file", "detail": name})
    if cohort == "mpc" and not (run_dir / "planner_events.jsonl").is_file():
        issues.append({"cohort": cohort, "run_dir": str(run_dir), "issue": "missing_file", "detail": "planner_events.jsonl"})

    run_config = payload.get("run_config", {})
    condition = payload.get("condition", {})
    method = str(payload.get("method", payload.get("projection", "")))
    case_id = str(payload.get("case_id", ""))
    row: dict[str, Any] = {
        "cohort": cohort,
        "run_dir": str(run_dir),
        "fingerprint_sha256": fingerprint.get("sha256", ""),
        "method": method,
        "case_id": case_id,
        "trajectory": payload.get("reference_type", ""),
        "condition": condition.get("name", ""),
        "perturbation": condition.get("perturbation", ""),
        "level": condition.get("level", ""),
        "seed": run_config.get("seed", ""),
        "reference_sha256": _identity(payload, "reference"),
        "checkpoint_sha256": _identity(payload, "checkpoint"),
        "normalizer_sha256": _identity(payload, "normalizer"),
        "model_xml_sha256": _identity(payload, "nominal_model"),
        "control_semantics_version": "",
        "projection_semantics_version": "",
        "residual_parameterization": run_config.get("residual_parameterization", "full_compat"),
        "stage_one_task_space_cost": run_config.get("stage_one_task_space_cost", "off_compat"),
        "stage_one_task_compile": run_config.get("stage_one_task_compile", "off_compat"),
        "mpc_preview_nominal_steps": run_config.get("mpc_preview_nominal_steps", 0),
        "exact_task_space_cost": run_config.get("exact_task_space_cost", "off" if cohort == "ik" else ""),
        "status": "ok",
        "provenance_class": "historical_compatible_unverified_clean_commit",
    }
    rollout = run_dir / "rollout.npz"
    if rollout.is_file():
        try:
            with np.load(rollout, allow_pickle=False) as archive:
                row["control_semantics_version"] = _scalar(archive, "control_semantics_version", 0)
                row["projection_semantics_version"] = _scalar(archive, "projection_semantics_version", 0)
                lengths = {}
                for key in PRIMARY_ARRAYS:
                    if key not in archive:
                        issues.append({"cohort": cohort, "run_dir": str(run_dir), "issue": "missing_array", "detail": key})
                        continue
                    value = np.asarray(archive[key])
                    lengths[key] = len(value)
                    if not np.all(np.isfinite(value)):
                        issues.append({"cohort": cohort, "run_dir": str(run_dir), "issue": "nonfinite_primary_array", "detail": key})
                row.update({f"{key}_length": value for key, value in lengths.items()})
                if len(set(lengths.values())) > 1:
                    issues.append({"cohort": cohort, "run_dir": str(run_dir), "issue": "length_mismatch", "detail": json.dumps(lengths, sort_keys=True)})
                steps = int(lengths.get("actual_states", 0))
                if steps <= 0:
                    issues.append({"cohort": cohort, "run_dir": str(run_dir), "issue": "empty_rollout", "detail": str(steps)})
        except (OSError, ValueError, KeyError) as exc:
            issues.append({"cohort": cohort, "run_dir": str(run_dir), "issue": "invalid_rollout", "detail": str(exc)})
    if issues:
        row["status"] = "issues"
    return row, issues


def _duplicates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["cohort"], row["method"], row["case_id"], row["condition"],
            row["perturbation"], row["level"],
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, members in groups.items():
        fingerprints = {member["fingerprint_sha256"] for member in members}
        if len(members) > 1 and len(fingerprints) > 1:
            output.append({
                "key": json.dumps(key), "count": len(members),
                "fingerprints": ";".join(sorted(fingerprints)), "conflicting": True,
            })
    return output


def _configuration_mismatches(rows: list[dict[str, Any]], target: dict[str, Any] | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    expected = {
        "checkpoint_sha256": target.get("checkpoint", {}).get("sha256", "") if target else "",
        "normalizer_sha256": target.get("normalizer", {}).get("sha256", "") if target else "",
        "model_xml_sha256": target.get("model_xml", {}).get("sha256", "") if target else "",
    }
    for row in rows:
        if row["cohort"] == "ik":
            relevant = ("model_xml_sha256",)
        else:
            relevant = tuple(expected)
        for key in relevant:
            if expected[key] and row[key] != expected[key]:
                output.append({
                    "run_dir": row["run_dir"], "field": key,
                    "actual": row[key], "expected": expected[key], "explained": False,
                })
    return output


def main() -> None:
    args = parser().parse_args()
    output = Path(args.output_dir).resolve()
    mpc_paths = _discover([Path(value).resolve() for value in args.mpc_roots], MPC_KIND)
    ik_paths = _discover([Path(value).resolve() for value in args.ik_roots], IK_KIND)
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for cohort, paths in (("mpc", mpc_paths), ("ik", ik_paths)):
        for path in paths:
            row, run_issues = audit_run(path, cohort)
            rows.append(row)
            issues.extend(run_issues)
    target = load_json(Path(args.target_manifest).resolve()) if args.target_manifest else None
    duplicates = _duplicates(rows)
    mismatches = _configuration_mismatches(rows, target)
    semantics = [
        {"cohort": cohort, "control_semantics_version": control, "projection_semantics_version": projection, "count": count}
        for (cohort, control, projection), count in Counter(
            (row["cohort"], row["control_semantics_version"], row["projection_semantics_version"])
            for row in rows
        ).items()
    ]
    expected = {"mpc": 720, "ik": 540}
    missing_cases = [
        {"cohort": cohort, "expected": count, "actual": sum(row["cohort"] == cohort for row in rows)}
        for cohort, count in expected.items()
        if sum(row["cohort"] == cohort for row in rows) != count
    ]
    summary = {
        "schema_version": 1,
        "source_roots": {
            "mpc": [file_identity(Path(value).resolve() / "experiment_manifest.json") for value in args.mpc_roots],
            "ik": [file_identity(Path(value).resolve() / "experiment_manifest.json") for value in args.ik_roots],
        },
        "counts": Counter(row["cohort"] for row in rows),
        "missing_formal_cases": len(missing_cases),
        "duplicate_conflicting_cases": len(duplicates),
        "run_issues": len(issues),
        "configuration_mismatches": len(mismatches),
        "mixed_control_semantics": len({row["control_semantics_version"] for row in rows}) != 1,
        "mixed_projection_semantics": len({row["projection_semantics_version"] for row in rows}) != 1,
        "compatibility_policy": "historical cohort; configuration-compatible but clean commit is unverified",
    }
    summary["passed"] = not any((
        summary["missing_formal_cases"],
        summary["duplicate_conflicting_cases"],
        summary["run_issues"],
        summary["configuration_mismatches"],
        summary["mixed_control_semantics"],
        summary["mixed_projection_semantics"],
    ))
    _write_csv(output / "evidence_inventory.csv", rows)
    _write_csv(output / "missing_cases.csv", missing_cases)
    _write_csv(output / "duplicate_cases.csv", duplicates)
    _write_csv(output / "configuration_mismatches.csv", mismatches)
    _write_csv(output / "semantics_versions.csv", semantics)
    _write_csv(output / "run_issues.csv", issues)
    (output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
