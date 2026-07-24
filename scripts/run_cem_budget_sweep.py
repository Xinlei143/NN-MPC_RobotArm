"""Run and evaluate the prescribed CEM-budget ablation matrix.

Each run writes to an isolated directory below ``--output_dir``.  Existing
completed runs are reused unless ``--rerun`` is supplied, making a long sweep
safe to resume after an interruption.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CEMBudget:
    name: str
    num_samples: int
    cem_iters: int
    uniform_sample_ratio: float
    rollout_batch_size: int | None = None
    num_elites: int | None = None
    elite_ratio: float | None = None

    def cli_args(self) -> list[str]:
        args = [
            "--num_samples",
            str(self.num_samples),
            "--cem_iters",
            str(self.cem_iters),
            "--uniform_sample_ratio",
            str(self.uniform_sample_ratio),
            "--rollout_batch_size",
            str(self.rollout_batch_size if self.rollout_batch_size is not None else self.num_samples),
        ]
        if self.num_elites is not None:
            args.extend(["--num_elites", str(self.num_elites)])
        if self.elite_ratio is not None:
            args.extend(["--elite_ratio", str(self.elite_ratio)])
        return args


BUDGETS = {
    "baseline": CEMBudget("baseline", 128, 2, 0.15, elite_ratio=0.08),
    "a": CEMBudget("a", 64, 2, 0.15, elite_ratio=0.08),
    "b": CEMBudget("b", 32, 2, 0.15, num_elites=4),
    "c": CEMBudget("c", 24, 2, 0.20, num_elites=4),
    "d": CEMBudget("d", 32, 1, 0.20, num_elites=4),
}

TASK_REFERENCE_DIRS = {
    "circle": "circle_3laps",
    "ellipse": "ellipse_3laps",
    "figure8": "figure8_3laps",
    "square": "square_3laps",
}


def _csv_names(value: str, allowed: set[str], argument: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in names if item not in allowed]
    if unknown:
        raise ValueError(f"{argument} has unsupported values: {unknown}; choose from {sorted(allowed)}")
    if not names:
        raise ValueError(f"{argument} must select at least one value")
    return names


def _csv_ints(value: str, argument: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{argument} must be comma-separated integers") from exc
    if not parsed or any(item < 0 for item in parsed):
        raise ValueError(f"{argument} must contain non-negative integers")
    return parsed


def _csv_positive_ints(value: str, argument: str) -> list[int]:
    values = _csv_ints(value, argument)
    if any(item <= 0 for item in values):
        raise ValueError(f"{argument} must contain positive integers")
    return values


def _grid_budgets(args: argparse.Namespace) -> tuple[dict[str, CEMBudget], list[str], str]:
    grid_values = (args.num_samples_values, args.rollout_batch_size_values, args.cem_iters_values)
    if not any(value is not None for value in grid_values):
        configs = _csv_names(args.configs, set(BUDGETS), "--configs")
        return {name: BUDGETS[name] for name in configs}, configs, "baseline"
    if any(value is None for value in grid_values):
        raise ValueError(
            "--num_samples_values, --rollout_batch_size_values, and --cem_iters_values must be provided together"
        )

    num_samples_values = _csv_positive_ints(args.num_samples_values, "--num_samples_values")
    rollout_batch_size_values = _csv_positive_ints(args.rollout_batch_size_values, "--rollout_batch_size_values")
    cem_iters_values = _csv_positive_ints(args.cem_iters_values, "--cem_iters_values")
    if len(num_samples_values) != len(rollout_batch_size_values):
        raise ValueError(
            "--num_samples_values and --rollout_batch_size_values must have the same length for paired sweeps"
        )

    budgets: dict[str, CEMBudget] = {}
    for num_samples, rollout_batch_size in zip(num_samples_values, rollout_batch_size_values):
        for cem_iters in cem_iters_values:
            name = f"n{num_samples}_b{rollout_batch_size}_i{cem_iters}"
            budgets[name] = CEMBudget(
                name=name,
                num_samples=num_samples,
                cem_iters=cem_iters,
                uniform_sample_ratio=0.15,
                rollout_batch_size=rollout_batch_size,
                elite_ratio=0.08,
            )

    configs = list(budgets)
    baseline_config = args.baseline_config or "n128_b128_i2"
    if baseline_config not in budgets:
        raise ValueError(
            f"--baseline_config {baseline_config!r} is not in the generated budget grid; choose from {configs}"
        )
    return budgets, configs, baseline_config


def _resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable CEM sample/iteration ablation sweep.")
    parser.add_argument(
        "--checkpoint",
        default="dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/best_model.pt",
    )
    parser.add_argument(
        "--normalizer",
        default="dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/normalizer.pt",
    )
    parser.add_argument("--model_type", default="transformer", choices=["mlp", "gru", "transformer"])
    parser.add_argument("--reference_root", default="outputs/references")
    parser.add_argument("--output_dir", default="outputs/mpc/cem_budget_sweep")
    parser.add_argument("--tasks", default="circle,ellipse,figure8,square")
    parser.add_argument("--configs", default="baseline,a,b,c,d")
    parser.add_argument(
        "--num_samples_values",
        default=None,
        help="Comma-separated CEM sample counts for a paired runtime budget sweep.",
    )
    parser.add_argument(
        "--rollout_batch_size_values",
        default=None,
        help="Comma-separated rollout batch sizes paired by position with --num_samples_values.",
    )
    parser.add_argument(
        "--cem_iters_values",
        default=None,
        help="Comma-separated CEM iteration counts to combine with each paired sample/batch budget.",
    )
    parser.add_argument(
        "--baseline_config",
        default=None,
        help="Generated budget name used for relative RMSE and safety comparisons; defaults to n128_b128_i2.",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument(
        "--replan_interval_steps",
        type=int,
        default=5,
        help="Fixed replan interval for synchronous/virtual modes; ignored by threaded_asap.",
    )
    parser.add_argument(
        "--multirate_mode",
        choices=["synchronous", "virtual_asap", "virtual_smooth", "threaded_asap"],
        default="threaded_asap",
        help="Execution architecture passed to run_cem_mpc.py; threaded_asap is the default CUDA soft-real-time controller.",
    )
    parser.add_argument(
        "--anticipation_delay_steps",
        type=int,
        default=6,
        help="Expected planner-to-activation delay in 100 Hz command steps.",
    )
    parser.add_argument(
        "--mpc_warmup_plans",
        type=int,
        default=1,
        help="Discarded CEM plans before the first control command of each rollout.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latency_budget_ms", type=float, default=50.0)
    parser.add_argument("--target_latency_ms", type=float, default=20.0)
    parser.add_argument("--max_rmse_regression", type=float, default=0.05)
    parser.add_argument("--max_execution_steps", type=int, default=None)
    parser.add_argument("--rerun", action="store_true", help="Replace completed run directories.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands and manifest without launching runs.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def _finite_median(values: list[float]) -> float | None:
    finite = [value for value in values if value == value and value not in (float("inf"), float("-inf"))]
    return float(median(finite)) if finite else None


def _run_record(run_dir: Path, config: str, task: str, seed: int) -> dict[str, Any] | None:
    run = _read_json(run_dir / "run_summary.json")
    tracking = _read_json(run_dir / "task_tracking_summary.json")
    if run is None or tracking is None:
        return None
    timing = run.get("timing", {}).get("planning_time_s", {})
    safety = run.get("safety", {})
    position_rmse = tracking.get("overall", {}).get("position_rmse_m")
    orientation_rmse = tracking.get("overall", {}).get("orientation_rmse_rad")
    return {
        "config": config,
        "task": task,
        "seed": seed,
        "run_dir": str(run_dir),
        "planning_p95_s": timing.get("p95"),
        "planning_mean_s": timing.get("mean"),
        "tcp_position_rmse_m": position_rmse,
        "tcp_orientation_rmse_rad": orientation_rmse,
        "planner_failures": safety.get("controller_failure_count"),
        "joint_limit_violations": safety.get("joint_limit_violation_count"),
        "recovery_triggers": safety.get("recovery_trigger_count"),
    }


def build_summary(
    output_dir: Path,
    budgets: dict[str, CEMBudget],
    configs: list[str],
    baseline_config: str,
    tasks: list[str],
    seeds: list[int],
    latency_budget_s: float,
    target_latency_s: float,
    max_rmse_regression: float,
    partial_execution: bool = False,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for config in configs:
        for task in tasks:
            for seed in seeds:
                record = _run_record(output_dir / config / task / f"seed_{seed}", config, task, seed)
                if record is not None:
                    records.append(record)
    baseline = {(item["task"], item["seed"]): item for item in records if item["config"] == baseline_config}
    configurations: dict[str, Any] = {}
    for config in configs:
        config_records = [item for item in records if item["config"] == config]
        task_summary: dict[str, Any] = {}
        for task in tasks:
            task_records = [item for item in config_records if item["task"] == task]
            relative_rmse = []
            safety_ok = True
            for item in task_records:
                reference = baseline.get((task, item["seed"]))
                if config != baseline_config and reference is not None:
                    current = item.get("tcp_position_rmse_m")
                    baseline_value = reference.get("tcp_position_rmse_m")
                    if isinstance(current, (int, float)) and isinstance(baseline_value, (int, float)) and baseline_value > 0:
                        relative_rmse.append(float(current) / float(baseline_value))
                    for key in ("planner_failures", "joint_limit_violations"):
                        if isinstance(item.get(key), int) and isinstance(reference.get(key), int):
                            safety_ok = safety_ok and item[key] <= reference[key]
            p95_values = [float(item["planning_p95_s"]) for item in task_records if isinstance(item.get("planning_p95_s"), (int, float))]
            task_summary[task] = {
                "completed_seeds": sorted(item["seed"] for item in task_records),
                "planning_p95_median_s": _finite_median(p95_values),
                "tcp_position_rmse_median_m": _finite_median(
                    [float(item["tcp_position_rmse_m"]) for item in task_records if isinstance(item.get("tcp_position_rmse_m"), (int, float))]
                ),
                "tcp_position_rmse_ratio_to_baseline_max": max(relative_rmse) if relative_rmse else None,
                "safety_not_worse_than_baseline": safety_ok,
            }
        complete = not partial_execution and all(
            task_summary[task]["completed_seeds"] == sorted(seeds) for task in tasks
        )
        p95_values = [task_summary[task]["planning_p95_median_s"] for task in tasks]
        rmse_ratios = [task_summary[task]["tcp_position_rmse_ratio_to_baseline_max"] for task in tasks]
        configurations[config] = {
            "budget": asdict(budgets[config]),
            "complete": complete,
            "tasks": task_summary,
            "acceptance": {
                "meets_20hz": complete and all(isinstance(value, float) and value <= latency_budget_s for value in p95_values),
                "meets_50hz": complete and all(isinstance(value, float) and value <= target_latency_s for value in p95_values),
                "within_rmse_regression": config == baseline_config
                or (complete and all(isinstance(value, float) and value <= 1.0 + max_rmse_regression for value in rmse_ratios)),
                "safety_not_worse_than_baseline": complete and all(
                    task_summary[task]["safety_not_worse_than_baseline"] for task in tasks
                ),
            },
        }
    return {
        "schema_version": 1,
        "records": records,
        "configurations": configurations,
        "baseline_config": baseline_config,
        "criteria": {
            "latency_budget_s": latency_budget_s,
            "target_latency_s": target_latency_s,
            "max_rmse_regression": max_rmse_regression,
        },
        "partial_execution": partial_execution,
    }


def main() -> None:
    args = parse_args()
    budgets, configs, baseline_config = _grid_budgets(args)
    tasks = _csv_names(args.tasks, set(TASK_REFERENCE_DIRS), "--tasks")
    seeds = _csv_ints(args.seeds, "--seeds")
    if args.horizon <= 0 or args.replan_interval_steps <= 0 or args.latency_budget_ms <= 0 or args.target_latency_ms <= 0:
        raise ValueError("horizon and latency targets must be positive")
    if args.replan_interval_steps > args.horizon:
        raise ValueError("replan_interval_steps must not exceed horizon")
    if args.mpc_warmup_plans < 0:
        raise ValueError("mpc_warmup_plans must be non-negative")
    if args.max_rmse_regression < 0:
        raise ValueError("--max_rmse_regression must be non-negative")
    checkpoint = _resolve(args.checkpoint)
    normalizer = _resolve(args.normalizer)
    reference_root = _resolve(args.reference_root)
    output_dir = _resolve(args.output_dir)
    for required in (checkpoint, normalizer):
        if not required.exists():
            raise FileNotFoundError(required)
    references = {task: reference_root / TASK_REFERENCE_DIRS[task] / "reference.npz" for task in tasks}
    for reference in references.values():
        if not reference.exists():
            raise FileNotFoundError(reference)

    manifest = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "normalizer": str(normalizer),
        "model_type": args.model_type,
        "horizon": args.horizon,
        "replan_interval_steps": args.replan_interval_steps,
        "multirate_mode": args.multirate_mode,
        "anticipation_delay_steps": args.anticipation_delay_steps,
        "mpc_warmup_plans": args.mpc_warmup_plans,
        "device": args.device,
        "tasks": {task: str(reference) for task, reference in references.items()},
        "seeds": seeds,
        "configs": {name: asdict(budgets[name]) for name in configs},
        "baseline_config": baseline_config,
        "criteria": {
            "latency_budget_ms": args.latency_budget_ms,
            "target_latency_ms": args.target_latency_ms,
            "max_rmse_regression": args.max_rmse_regression,
        },
        "max_execution_steps": args.max_execution_steps,
    }
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    runner = ROOT / "scripts" / "run_cem_mpc.py"
    for config in configs:
        budget = budgets[config]
        for task in tasks:
            for seed in seeds:
                run_dir = output_dir / config / task / f"seed_{seed}"
                if (run_dir / "run_summary.json").exists() and not args.rerun:
                    print(f"SKIP completed {config}/{task}/seed_{seed}: {run_dir}", flush=True)
                    continue
                command = [
                    sys.executable,
                    str(runner),
                    "--checkpoint",
                    str(checkpoint),
                    "--normalizer",
                    str(normalizer),
                    "--model_type",
                    args.model_type,
                    "--reference_mode",
                    "task",
                    "--reference_file",
                    str(references[task]),
                    "--horizon",
                    str(args.horizon),
                    "--replan_interval_steps",
                    str(args.replan_interval_steps),
                    "--multirate_mode",
                    args.multirate_mode,
                    "--anticipation_delay_steps",
                    str(args.anticipation_delay_steps),
                    "--mpc_warmup_plans",
                    str(args.mpc_warmup_plans),
                    "--device",
                    args.device,
                    "--mpc_policy",
                    "residual",
                    "--cem_execute",
                    "lowest_cost",
                    "--seed",
                    str(seed),
                    "--save_dir",
                    str(run_dir),
                    *budget.cli_args(),
                ]
                if args.max_execution_steps is not None:
                    command.extend(["--max_execution_steps", str(args.max_execution_steps)])
                print("RUN "+" ".join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, check=True)
    if not args.dry_run:
        summary = build_summary(
            output_dir,
            budgets,
            configs,
            baseline_config,
            tasks,
            seeds,
            args.latency_budget_ms / 1e3,
            args.target_latency_ms / 1e3,
            args.max_rmse_regression,
            partial_execution=args.max_execution_steps is not None,
        )
        with (output_dir / "sweep_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"Saved sweep summary to {output_dir / 'sweep_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
