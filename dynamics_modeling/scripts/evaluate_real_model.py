#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

import numpy as np
import torch

from neural_dynamics.rollout import load_dynamics_bundle, rollout_dynamics_batch
from mpc.robot_config import load_robot_spec


def metrics(error: np.ndarray, n_joints: int) -> dict[str, object]:
    absolute = np.abs(error[..., :n_joints])
    return {"q_rmse": float(np.sqrt(np.mean(absolute ** 2))), "q_p95": float(np.percentile(absolute, 95)),
            "q_max": float(np.max(absolute)),
            "q_rmse_per_joint": np.sqrt(np.mean(absolute ** 2, axis=0)).tolist()}


def fit_first_order(states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = actions.shape[1]
    coefficients = np.empty((n, 3))
    for joint in range(n):
        design = np.stack((np.ones(np.sum(mask)), states[mask, n + joint],
                           actions[mask, joint] - states[mask, joint]), axis=1)
        coefficients[joint] = np.linalg.lstsq(design, next_states[mask, n + joint], rcond=None)[0]
    return coefficients


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate SO101 Model A against persistence/CV/first-order baselines.")
    parser.add_argument("--dataset", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalizer", required=True); parser.add_argument("--test-group-ids", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    data = np.load(args.dataset)
    states, actions, next_states = data["states"], data["actions"], data["next_states"]
    groups, episodes = data["split_group_ids"], data["episode_ids"]
    valid = data["valid_target"].astype(bool)
    test_groups = np.asarray([int(value) for value in args.test_group_ids.split(",")])
    test_mask = np.isin(groups, test_groups)
    fit_mask = valid & ~test_mask
    coefficient = fit_first_order(states, actions, next_states, fit_mask)
    robot = load_robot_spec("configs/robots/so101.yaml")
    bundle = load_dynamics_bundle(args.checkpoint, args.normalizer, "gru", 5, "cuda" if torch.cuda.is_available() else "cpu",
                                  history_len=8, expected_robot_spec=robot)
    horizon, history_len = 10, bundle.history_len
    anchors = []
    for index in range(history_len - 1, len(states) - horizon):
        window = slice(index - history_len + 1, index + horizon)
        if (test_mask[index] and np.all(valid[index:index+horizon]) and
                np.all(episodes[window] == episodes[index]) and np.all(groups[window] == groups[index])):
            anchors.append(index)
    if not anchors: raise SystemExit("no valid 10-step test windows")
    histories = np.stack([np.concatenate((states[i-history_len+1:i+1], actions[i-history_len+1:i+1]), axis=1) for i in anchors])
    future_actions = np.stack([actions[i:i+horizon] for i in anchors])
    truth = np.stack([[next_states[i+k] for k in range(horizon)] for i in anchors])
    learned = rollout_dynamics_batch(bundle.model, bundle.normalizer, bundle.model_type,
                                     torch.as_tensor(histories, device=bundle.device),
                                     torch.as_tensor(future_actions, device=bundle.device), bundle.state_dim,
                                     bundle.target_mode, bundle.control_dt, rollout_batch_size=1024)[:, 1:].cpu().numpy()
    initial = states[np.asarray(anchors)]
    persistence = np.repeat(initial[:, None, :], horizon, axis=1)
    constant_velocity = np.empty_like(truth); first_order = np.empty_like(truth)
    cv_state, fo_state = initial.copy(), initial.copy()
    for step in range(horizon):
        cv_state = cv_state.copy(); cv_state[:, :5] += cv_state[:, 5:] * bundle.control_dt
        constant_velocity[:, step] = cv_state
        dq = np.stack([coefficient[j, 0] + coefficient[j, 1] * fo_state[:, 5+j] +
                       coefficient[j, 2] * (future_actions[:, step, j] - fo_state[:, j]) for j in range(5)], axis=1)
        fo_state = np.concatenate((fo_state[:, :5] + dq * bundle.control_dt, dq), axis=1)
        first_order[:, step] = fo_state
    reports = {}
    for step in (1, 3, 5, 10):
        reports[str(step)] = {name: metrics(pred[:, step-1] - truth[:, step-1], 5) for name, pred in
                              (("learned", learned), ("persistence", persistence),
                               ("constant_velocity", constant_velocity), ("first_order", first_order))}
    gate = all(reports[str(step)]["learned"]["q_rmse"] < min(reports[str(step)][name]["q_rmse"]
               for name in ("persistence", "constant_velocity", "first_order")) for step in (1, 3, 5, 10))
    payload = {"window_count": len(anchors), "fixed_control_dt": bundle.control_dt, "by_horizon": reports,
               "model_gate_passed": gate}
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output: Path(args.output).write_text(text + "\n", encoding="utf-8")
    if not gate: raise SystemExit(2)


if __name__ == "__main__": main()
