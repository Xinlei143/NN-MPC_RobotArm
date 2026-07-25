"""Build immutable robustness benchmarks from generated task-space references."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.robustness._evaluation import sha256
from scripts.robustness._runtime import ROOT, load_runner


TYPES = ("multi_joint_sine", "waypoint", "chirp", "circle", "figure8", "ellipse", "square")
TASK_TYPES = {"circle", "figure8", "ellipse", "square"}
PAPER_IK_TYPES = ("circle", "figure8", "fast_ellipse", "rounded_square")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def runner_defaults() -> dict[str, Any]:
    runner = load_runner("robustness_manifest_runner")
    defaults = vars(runner.build_arg_parser().parse_args([]))
    excluded = {"checkpoint", "normalizer", "model_type", "history_len", "device", "seed", "save_dir", "reference_mode", "reference_file", "episode_len", "horizon", "anticipation_delay_steps", "multirate_mode", "payload_level", "actuator_gain_level", "force_pulse_level", "observation_noise_level"}
    return {key: value for key, value in defaults.items() if key not in excluded}


def paper_ik_cases(reference_manifest: Path, locked: dict[str, Any], seed: int) -> list[dict[str, object]]:
    """Freeze the four paper task-space references for DirectIK robustness tests."""
    payload = json.loads(reference_manifest.read_text(encoding="utf-8"))
    records = payload.get("references")
    if not isinstance(records, list):
        raise ValueError("Paper reference manifest has no references list")
    by_label = {
        str(record.get("label")): record
        for record in records
        if isinstance(record, dict)
    }
    missing = set(PAPER_IK_TYPES).difference(by_label)
    if missing:
        raise ValueError(f"Paper reference manifest is missing: {sorted(missing)}")
    if "preview_calibration" not in by_label:
        raise ValueError("Paper reference manifest must contain preview_calibration for provenance")
    cases: list[dict[str, object]] = []
    for index, label in enumerate(PAPER_IK_TYPES):
        record = by_label[label]
        identity = record.get("file")
        if not isinstance(identity, dict):
            raise ValueError(f"Paper reference record has no file identity: {label}")
        reference = Path(str(identity.get("path", "")))
        if not reference.is_file():
            raise FileNotFoundError(f"Missing paper reference for {label}: {reference}")
        expected = str(identity.get("sha256", ""))
        actual = sha256(reference)
        if not expected or actual != expected:
            raise ValueError(f"Paper reference hash mismatch for {label}")
        run_args = dict(locked)
        run_args.update(
            seed=seed + index,
            reference_file=str(reference),
            reference_mode="task",
            episode_len=None,
        )
        cases.append({
            "id": label,
            "reference_type": label,
            "reference_sha256": actual,
            "run_args": run_args,
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable robustness benchmark manifest.")
    parser.add_argument("--output_path", default="outputs/robustness/benchmark.json")
    parser.add_argument("--reference_dir", default="outputs/robustness/references")
    parser.add_argument("--cases_per_type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--delay", type=int)
    parser.add_argument(
        "--paper_reference_manifest",
        default=None,
        help="Build the four-trajectory DirectIK benchmark from a paper reference manifest.",
    )
    args = parser.parse_args()
    if args.cases_per_type <= 0 or args.horizon <= 0:
        raise ValueError("--cases_per_type and --horizon must be positive")
    if not args.paper_reference_manifest and (args.delay is None or args.delay <= 0):
        raise ValueError("--delay must be positive unless --paper_reference_manifest is used")
    output, references = resolve(args.output_path), resolve(args.reference_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable robustness manifest: {output}")
    locked = runner_defaults()
    locked.update(n_joints=6)
    if args.paper_reference_manifest:
        source = resolve(args.paper_reference_manifest)
        cases = paper_ik_cases(source, locked, args.seed)
        kind = "paper_direct_ik_robustness"
        note = "Immutable paper DirectIK benchmark; all cases use frozen task-space paper references."
    else:
        assert args.delay is not None
        locked.update(controller_mode="mpc", mpc_policy="residual", multirate_mode="threaded_asap", num_samples=128, cem_iters=2, replan_interval_steps=5, rollout_batch_size=128)
        cases = []
        for type_index, reference_type in enumerate(TYPES):
            for ordinal in range(args.cases_per_type):
                reference = references / f"{reference_type}_{ordinal:02d}" / "reference.npz"
                if not reference.is_file():
                    raise FileNotFoundError(f"Missing benchmark reference: {reference}")
                run_args = dict(locked)
                run_args.update(seed=args.seed + type_index * 10_000 + ordinal, episode_len=500, horizon=args.horizon,
                                anticipation_delay_steps=args.delay, reference_file=str(reference),
                                reference_mode="task" if reference_type in TASK_TYPES else "joint_file")
                cases.append({"id": f"{reference_type}_{ordinal:02d}", "reference_type": reference_type,
                              "reference_sha256": sha256(reference), "run_args": run_args})
        kind = "model_a_robustness"
        note = "Immutable Model-A robustness benchmark; MPC runs threaded_asap and Direct IK runs synchronous."
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "kind": kind, "seed": args.seed, "cases": cases,
               "locked_controller_config": locked, "note": note}
    if args.paper_reference_manifest:
        payload["paper_reference_manifest"] = {"path": str(resolve(args.paper_reference_manifest)), "sha256": sha256(resolve(args.paper_reference_manifest))}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} immutable {kind} cases to {output}")


if __name__ == "__main__":
    main()
