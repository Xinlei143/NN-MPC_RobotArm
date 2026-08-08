#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robot_runtime.config import file_sha256


def _same_length(data: np.lib.npyio.NpzFile, fields: set[str]) -> bool:
    sizes = {field: len(data[field]) for field in fields}
    return len(set(sizes.values())) == 1


def validate_dataset(dataset: str | Path, *, manifest: str | Path | None = None,
                     nominal_dt: float = 1 / 30, dt_tolerance: float = .10,
                     max_diagnostic_age_s: float = 3.0) -> dict[str, int | bool]:
    """Validate a current real-data archive and return the printed summary.

    Kept importable so the workspace collector validates its shard before a
    strict append.  No hardware access occurs here.
    """
    dataset_path = Path(dataset)
    data = np.load(dataset_path, allow_pickle=False)
    manifest_path = Path(manifest) if manifest else dataset_path.with_suffix(".manifest.json")
    parsed_manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                       if manifest_path.is_file() else None)
    required = {"states", "actions", "next_states", "q_ref", "transmitted_q_ref", "valid_target",
                "state_timestamp_ns", "next_state_timestamp_ns", "history_generation", "next_history_generation",
                "estimator_generation", "next_estimator_generation", "actual_dt", "episode_ids", "session_ids",
                "split_group_ids", "diagnostic_sample_age_s", "goal_position_readback_raw",
                "goal_readback_mismatch"}
    table_required = {"table_predicted_clearance_m", "table_effective_clearance_m", "table_nearest_component",
                      "table_safety_identity", "table_safety_violation"}
    table_profile = None if parsed_manifest is None else parsed_manifest.get("table_safety")
    table_profiles = ({} if parsed_manifest is None
                      else dict(parsed_manifest.get("table_safety_profiles", {})))
    if table_profile is not None:
        if table_profile.get("sha256"):
            table_profiles.setdefault(str(table_profile["sha256"]), table_profile)
        required |= table_required
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    valid_target = np.asarray(data["valid_target"], dtype=bool)
    if valid_target.ndim != 1 or len(valid_target) != len(data["states"]):
        raise ValueError("valid_target must be a rank-1 array with one value per transition")
    if not np.any(valid_target):
        raise ValueError("dataset contains no valid training transitions")
    aligned = np.array_equal(data["actions"], data["q_ref"]) and np.array_equal(data["actions"], data["transmitted_q_ref"])
    monotonic = bool(np.all(data["next_state_timestamp_ns"] > data["state_timestamp_ns"]))
    # The recorder deliberately retains invalid rows as audit evidence.  A
    # timing reset necessarily crosses a history/estimator generation and is
    # marked ``valid_target=false``; it must not invalidate a session whose
    # usable transitions are otherwise continuous and in-window.
    no_cross_generation = bool(np.all(data["history_generation"][valid_target]
                                      == data["next_history_generation"][valid_target])
                               and np.all(data["estimator_generation"][valid_target]
                                          == data["next_estimator_generation"][valid_target]))
    shapes_valid = (data["states"].ndim == 2 and data["states"].shape[1] == 10
                    and data["next_states"].shape == data["states"].shape
                    and data["actions"].shape == (len(data["states"]), 5)
                    and data["goal_position_readback_raw"].shape == (len(data["states"]), 6))
    same_length = _same_length(data, required)
    finite = bool(all(np.all(np.isfinite(data[field])) for field in ("states", "actions", "next_states", "actual_dt")))
    dt_in_window = bool(np.all(np.abs(data["actual_dt"][valid_target] - nominal_dt)
                               <= dt_tolerance * nominal_dt))
    diagnostic_fresh = bool(np.all(np.isfinite(data["diagnostic_sample_age_s"]))
                            and np.all(data["diagnostic_sample_age_s"] <= max_diagnostic_age_s))
    session_ids = np.asarray(data["session_ids"]).astype(str)
    session_valid = bool(np.all(session_ids != "") and np.all(np.asarray(data["episode_ids"]) >= 0)
                         and np.all(np.asarray(data["split_group_ids"]) >= 0))
    manifest_valid = False
    if parsed_manifest is not None:
        manifest_valid = parsed_manifest.get("dataset", {}).get("sha256") == file_sha256(dataset_path)
    table_clearance_valid = True
    if table_profile is not None:
        identities = np.asarray(data["table_safety_identity"]).astype(str)
        thresholds = np.asarray([
            float(table_profiles.get(identity, {}).get("minimum_clearance_m", np.inf))
            for identity in identities
        ], dtype=np.float64)
        table_clearance_valid = bool(
            np.all(np.isfinite(data["table_effective_clearance_m"]))
            and np.all(data["table_effective_clearance_m"] >= thresholds)
            and np.all(np.isin(identities, list(table_profiles)))
            and np.all(np.asarray(data["table_safety_violation"]).astype(str) == "")
        )
    summary = {"samples": len(data["states"]), "valid_samples": int(np.sum(valid_target)),
               "invalid_audit_samples": int(np.sum(~valid_target)),
               "actions_aligned": aligned, "timestamps_monotonic": monotonic,
               "no_cross_generation": no_cross_generation, "shapes_valid": shapes_valid,
               "all_fields_same_length": same_length, "finite": finite, "dt_in_nominal_window": dt_in_window,
               "diagnostic_fresh": diagnostic_fresh, "session_fields_valid": session_valid,
               "manifest_hash_valid": manifest_valid, "table_clearance_valid": table_clearance_valid}
    # Gate only on the boolean verdicts.  The integer counts (samples,
    # valid_samples, invalid_audit_samples) are informational: in particular a
    # clean session has invalid_audit_samples == 0, which is falsy and must not
    # trip the all() gate.
    if not all(value for key, value in summary.items() if isinstance(value, bool)):
        raise ValueError(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--nominal-dt", type=float, default=1 / 30)
    parser.add_argument("--dt-tolerance", type=float, default=.10)
    parser.add_argument("--max-diagnostic-age-s", type=float, default=3.0)
    args = parser.parse_args()
    try:
        summary = validate_dataset(args.dataset, manifest=args.manifest, nominal_dt=args.nominal_dt,
                                   dt_tolerance=args.dt_tolerance, max_diagnostic_age_s=args.max_diagnostic_age_s)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
