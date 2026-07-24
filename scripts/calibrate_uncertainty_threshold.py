"""Calibrate selected-trajectory disagreement thresholds from a nominal rollout."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "dynamics_modeling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from mpc.budgeted_uncertainty import DynamicsEnsemble, evaluate_replicas_with_primary_predictions
from mpc.history import history_tokens
from neural_dynamics.rollout import load_dynamics_bundle, rollout_dynamics_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True, help="Nominal Model-A rollout.npz with actual_states and actuator_q_ref.")
    parser.add_argument("--output", default=None, help="JSON output; defaults beside --rollout.")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--stride", type=int, default=5, help="Score every N control ticks.")
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="dynamics_modeling/outputs/checkpoints/gru_20260717_182930/best_model.pt")
    parser.add_argument("--normalizer", default="dynamics_modeling/outputs/checkpoints/gru_20260717_182930/normalizer.pt")
    parser.add_argument("--replica_checkpoints", nargs="+", required=True)
    parser.add_argument("--replica_normalizers", nargs="+", required=True)
    args = parser.parse_args()
    if args.horizon <= 0 or args.stride <= 0 or not 0.0 < args.quantile < 1.0:
        raise ValueError("horizon/stride must be positive and quantile must be in (0, 1)")
    if len(args.replica_checkpoints) < 2 or len(args.replica_checkpoints) != len(args.replica_normalizers):
        raise ValueError("provide at least two replica checkpoints and paired normalizers")
    return args


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    rollout_path = resolve(args.rollout)
    with np.load(rollout_path, allow_pickle=False) as archive:
        states = np.asarray(archive["actual_states"], dtype=np.float32)
        commands = np.asarray(archive["actuator_q_ref"], dtype=np.float32)
    if states.ndim != 2 or commands.ndim != 2 or len(states) != len(commands):
        raise ValueError("rollout actual_states and actuator_q_ref must be equal-length matrices")
    device = torch.device(args.device)
    primary = load_dynamics_bundle(resolve(args.checkpoint), resolve(args.normalizer), "gru", n_joints=6, device=device)
    ensemble = DynamicsEnsemble.from_replica_paths(
        primary,
        [resolve(path) for path in args.replica_checkpoints],
        [resolve(path) for path in args.replica_normalizers],
        device,
    )
    scores: list[float] = []
    steps: list[int] = []
    for step in range(0, len(states) - args.horizon + 1, args.stride):
        start = max(0, step - primary.history_len + 1)
        history = torch.as_tensor(
            history_tokens(states[start : step + 1], commands[start : step + 1], primary.history_len),
            dtype=torch.float32,
            device=device,
        )
        selected = commands[step : step + args.horizon]
        with torch.inference_mode():
            primary_prediction = rollout_dynamics_batch(
                model=primary.model, normalizer=primary.normalizer, model_type=primary.model_type,
                initial_history=history, future_q_ref=torch.as_tensor(selected, dtype=torch.float32, device=device).unsqueeze(0),
                state_dim=primary.state_dim, target_mode=primary.target_mode, control_dt=primary.control_dt,
            )[0].detach().cpu().numpy()
        report = evaluate_replicas_with_primary_predictions(
            ensemble, history, {"selected": selected}, {"selected": primary_prediction}, budget_ms=1e6,
        )
        if not report.timed_out and np.isfinite(report.selected_score):
            scores.append(report.selected_score)
            steps.append(step)
    if not scores:
        raise RuntimeError("No finite uncertainty scores were produced")
    values = np.asarray(scores, dtype=np.float64)
    payload = {
        "rollout": str(rollout_path.resolve()), "models": 1 + len(args.replica_checkpoints),
        "horizon": args.horizon, "stride": args.stride, "quantile": args.quantile,
        "threshold": float(np.quantile(values, args.quantile)), "samples": int(len(values)),
        "distribution": {key: float(np.quantile(values, q)) for key, q in (("min", 0.0), ("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99), ("max", 1.0))},
        "steps": steps, "scores": scores,
    }
    output = resolve(args.output) if args.output else rollout_path.with_name("uncertainty_calibration_h5.json")
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
