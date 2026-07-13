from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from mpc.logging import save_mpc_run


def load_run_cem_mpc_module():
    spec = importlib.util.spec_from_file_location("local_run_cem_mpc", ROOT / "scripts" / "run_cem_mpc.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load scripts/run_cem_mpc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_CEM_MPC = load_run_cem_mpc_module()


def parse_model_spec(value: str) -> dict[str, str]:
    parts = [item.strip() for item in value.split(",")]
    if len(parts) not in {4, 5}:
        raise ValueError(
            "--model_spec must be label,checkpoint,normalizer,model_type[,dataset_path], "
            f"got {value!r}"
        )
    return {
        "label": parts[0],
        "checkpoint": parts[1],
        "normalizer": parts[2],
        "model_type": parts[3],
        "dataset_path": "" if len(parts) == 4 else parts[4],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = RUN_CEM_MPC.build_arg_parser()
    parser.description = "Evaluate Model A/B/C checkpoints with common CEM-MPC settings."
    for action in parser._actions:
        if action.dest in {"checkpoint", "normalizer"}:
            action.required = False
    parser.add_argument(
        "--model_spec",
        action="append",
        required=True,
        help="Repeat as label,checkpoint,normalizer,model_type[,dataset_path].",
    )
    parser.set_defaults(save_dir="outputs/mpc/model_abc")
    return parser.parse_args(argv)


def summarize(label: str, arrays: dict[str, np.ndarray], dataset_path: str) -> dict[str, float | str | int]:
    tracking = arrays["realized_tracking_error"]
    failures = arrays["failure_flags"]
    planning_time = arrays["planning_time"]
    predicted_gap = arrays["predicted_real_error_gap"]
    return {
        "label": label,
        "dataset_path": dataset_path,
        "steps": int(len(tracking)),
        "tracking_error_mean": float(np.mean(tracking)) if len(tracking) else float("nan"),
        "tracking_error_final": float(tracking[-1]) if len(tracking) else float("nan"),
        "failure_rate": float(np.mean(failures)) if len(failures) else float("nan"),
        "planning_time_mean": float(np.mean(planning_time)) if len(planning_time) else float("nan"),
        "best_cost_mean": float(np.mean(arrays["best_cost"])) if len(arrays["best_cost"]) else float("nan"),
        "predicted_real_error_gap_mean": float(np.mean(predicted_gap)) if len(predicted_gap) else float("nan"),
    }


def write_summary(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    save_root = RUN_CEM_MPC.resolve_runtime_path(args.save_dir)
    rows: list[dict[str, float | str | int]] = []
    for spec_text in args.model_spec:
        spec = parse_model_spec(spec_text)
        run_args = deepcopy(args)
        run_args.checkpoint = spec["checkpoint"]
        run_args.normalizer = spec["normalizer"]
        run_args.model_type = spec["model_type"]
        run_args.save_dir = str(save_root / spec["label"])
        result = RUN_CEM_MPC.run_closed_loop_mpc(run_args)
        save_mpc_run(Path(run_args.save_dir), result["arrays"], result["rows"])
        rows.append(summarize(spec["label"], result["arrays"], spec["dataset_path"]))
    write_summary(save_root / "model_abc_summary.csv", rows)
    print(f"Saved Model A/B/C summary to {save_root / 'model_abc_summary.csv'}")


if __name__ == "__main__":
    main()
