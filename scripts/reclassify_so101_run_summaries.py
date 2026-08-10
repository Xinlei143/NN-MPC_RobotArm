#!/usr/bin/env python3
"""Reclassify legacy discrete-command acceleration diagnostics offline.

Older SO101 evidence files stored encoder-quantisation acceleration steps in
``command_acceleration_violation_flags``.  This migration updates only the
derived ``run_summary.json`` files; raw ``rollout.npz`` evidence is preserved.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/hardware/so101_paper_final/20260810_tracking1deg_v1"


def main() -> int:
    updated = 0
    for rollout_path in sorted(RUN_ROOT.glob("*/rollout.npz")):
        summary_path = rollout_path.with_name("run_summary.json")
        if not summary_path.exists():
            continue
        with np.load(rollout_path, allow_pickle=False) as archive:
            names = set(archive.files)
            if "command_acceleration_quantization_exceedance_flags" in names:
                quantization = np.asarray(
                    archive["command_acceleration_quantization_exceedance_flags"], dtype=bool
                )
                runtime = np.asarray(
                    archive.get("command_acceleration_violation_flags", np.zeros_like(quantization)),
                    dtype=bool,
                )
            else:
                legacy = np.asarray(
                    archive.get("command_acceleration_violation_flags", np.zeros(0)), dtype=bool
                )
                quantization = legacy
                runtime = np.zeros_like(legacy)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        safety = summary.setdefault("safety", {})
        safety["command_acceleration_violation_count"] = int(np.sum(runtime))
        safety["command_acceleration_quantization_exceedance_count"] = int(np.sum(quantization))
        safety["command_acceleration_flag_semantics"] = (
            "runtime_fault_only; quantization_exceedance_separate"
        )
        planning = summary.setdefault("planning", {})
        planning["command_acceleration_violation_count"] = int(np.sum(runtime))
        planning["command_acceleration_quantization_exceedance_count"] = int(np.sum(quantization))
        planning["command_acceleration_quantization_exceedance_rate"] = (
            float(np.mean(quantization)) if quantization.size else 0.0
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        updated += 1
    print(json.dumps({"updated_run_summaries": updated, "raw_rollouts_modified": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
