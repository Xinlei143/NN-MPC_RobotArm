from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from neural_dynamics.parallel_collector import save_dataset, validate_append_dataset


def load_run_cem_mpc_module():
    spec = importlib.util.spec_from_file_location("local_run_cem_mpc", ROOT / "scripts" / "run_cem_mpc.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load scripts/run_cem_mpc.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_CEM_MPC = load_run_cem_mpc_module()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = RUN_CEM_MPC.build_arg_parser()
    parser.description = "Collect MPC-induced closed-loop data compatible with train_dynamics.py."
    parser.add_argument("--save_path", default="outputs/datasets/mpc_induced_data.npz", type=str)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--source_policy", default="cem_mpc", type=str)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    save_path = RUN_CEM_MPC.resolve_runtime_path(args.save_path)
    if args.append:
        validate_append_dataset(save_path, require_episode_ids=True)

    result = RUN_CEM_MPC.run_closed_loop_mpc(args)
    arrays = result["arrays"]
    sample_count = min(len(arrays["actual_states"]), len(arrays["next_states"]), len(arrays["actuator_q_ref"]))
    if sample_count <= 0:
        raise RuntimeError("MPC rollout produced no complete transition samples")

    states = arrays["actual_states"][:sample_count].astype(np.float32)
    actions = arrays["actuator_q_ref"][:sample_count].astype(np.float32)
    next_states = arrays["next_states"][:sample_count].astype(np.float32)
    episode_ids = np.zeros(sample_count, dtype=np.int64)
    motion_mode_ids = np.full(sample_count, 100, dtype=np.int64)
    reference_mode_ids = np.full(sample_count, 0, dtype=np.int64)
    source_policy = np.asarray([args.source_policy] * sample_count)
    extra_arrays = {
        "q_ref": actions.copy(),
        "delta_q_ref": arrays["delta_q_ref"][:sample_count].astype(np.float32),
        "tau_actuator": arrays["tau_actuator"][:sample_count].astype(np.float32),
        "tau_gravity": arrays["tau_gravity"][:sample_count].astype(np.float32),
        "tau_total": arrays["tau_total"][:sample_count].astype(np.float32),
        "motion_mode_ids": motion_mode_ids,
        "reference_mode_ids": reference_mode_ids,
        "mpc_cost": arrays["best_cost"][:sample_count].astype(np.float32),
        "planning_time": arrays["planning_time"][:sample_count].astype(np.float32),
        "failure_flags": arrays["failure_flags"][:sample_count].astype(np.int64),
        "source_policy": source_policy,
    }
    saved = save_dataset(
        save_path,
        states,
        actions,
        next_states,
        append=args.append,
        episode_ids=episode_ids,
        extra_arrays=extra_arrays,
    )
    saved_episode_ids = saved[3]
    print(
        f"Saved MPC-induced dataset to {save_path} with states={states.shape}, "
        f"actions={actions.shape}, next_states={next_states.shape}, episode_ids={saved_episode_ids.shape}"
    )


if __name__ == "__main__":
    main()
