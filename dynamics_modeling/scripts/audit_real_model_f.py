#!/usr/bin/env python3
"""Stage-F review audit for SO101 Model-A real dynamics.

Implements the review items of `docs/hardware/so101-pre-mpc-experiment-checklist.md` §9
(stage F) and the learned-dynamics validation items of `Paper/PAPER_EXPERIMENT_CHECKLIST.md` §2:

  - per-joint NMSE / R^2 (q and dq channels)
  - amplitude ratio (predicted vs measured motion amplitude, commanded-motion subset)
  - divergence rate with explicit criteria (max-joint |q error| > theta at horizon H)
  - per-motion-mode RMSE breakdown
  - commanded-motion subset (hold excluded, to avoid dilution)
  - worst-window analysis (top windows by max-joint error at H=10)
  - test-session isolation and artifact identity audit
  - per-step error-growth curve, time axis in seconds (T_H = H * dt)

Outputs a timestamped JSON report and an error-growth CSV into the checkpoint directory.
Does not modify training/eval artifacts and does not gate anything; it produces review evidence.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_ROOT = ROOT / "dynamics_modeling"
for path in (ROOT, DYNAMICS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from neural_dynamics.rollout import load_dynamics_bundle, rollout_dynamics_batch
from mpc.robot_config import load_robot_spec

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MODE_NAMES = {0: "hold", 1: "step", 2: "smooth_random", 3: "sine", 4: "delta_ref_random"}
REPORT_HORIZONS = [1, 3, 5, 6, 10, 20]
DIVERGENCE_THRESHOLDS_DEG = [1.0, 2.0, 5.0]
DIVERGENCE_HORIZONS = [5, 10, 20]
HEADLINE_DIVERGENCE = (10, 2.0)  # horizon, threshold deg
AMPLITUDE_HORIZON = 10
AMPLITUDE_FLOOR_RAD = np.deg2rad(0.1)  # keep windows whose true motion std exceeds 0.1 deg
TOP_WINDOWS = 20


def fit_first_order(states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = actions.shape[1]
    coefficients = np.empty((n, 3))
    for joint in range(n):
        design = np.stack((np.ones(np.sum(mask)), states[mask, n + joint],
                           actions[mask, joint] - states[mask, joint]), axis=1)
        coefficients[joint] = np.linalg.lstsq(design, next_states[mask, n + joint], rcond=None)[0]
    return coefficients


def rmse_stats(err: np.ndarray) -> dict[str, object]:
    """err: [W, 10] q|dq error slice. Returns q and dq aggregate statistics."""
    q_err = np.abs(err[:, :5])
    dq_err = np.abs(err[:, 5:])
    return {
        "q_rmse": float(np.sqrt(np.mean(q_err ** 2))),
        "q_p95": float(np.percentile(q_err, 95)),
        "q_max": float(np.max(q_err)),
        "dq_rmse": float(np.sqrt(np.mean(dq_err ** 2))),
        "dq_p95": float(np.percentile(dq_err, 95)),
        "dq_max": float(np.max(dq_err)),
    }


def per_joint_nmse_r2(err: np.ndarray, truth: np.ndarray) -> dict[str, object]:
    """err/truth: [W, 10] slices at one horizon. Per-joint NMSE = MSE/var(truth), R^2 = 1 - NMSE."""
    out = {}
    for j in range(5):
        var_q = float(np.var(truth[:, j]))
        mse_q = float(np.mean(err[:, j] ** 2))
        var_dq = float(np.var(truth[:, 5 + j]))
        mse_dq = float(np.mean(err[:, 5 + j] ** 2))
        out[JOINTS[j]] = {
            "q_nmse": mse_q / var_q if var_q > 0 else None,
            "q_r2": 1.0 - mse_q / var_q if var_q > 0 else None,
            "dq_nmse": mse_dq / var_dq if var_dq > 0 else None,
            "dq_r2": 1.0 - mse_dq / var_dq if var_dq > 0 else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-F review audit for SO101 Model-A real dynamics.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normalizer", required=True)
    parser.add_argument("--test-group-ids", default="43,44,45,46,47")
    parser.add_argument("--output-dir", default=None, help="defaults to the checkpoint's parent directory")
    parser.add_argument("--max-horizon", type=int, default=30)
    args = parser.parse_args()

    test_groups = np.asarray([int(v) for v in args.test_group_ids.split(",")])
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"audit_f_{stamp}.json"
    curve_path = out_dir / f"error_growth_curve_{stamp}.csv"

    # ---- data and model ----
    data = np.load(args.dataset)
    states, actions, next_states = data["states"], data["actions"], data["next_states"]
    groups, episodes = data["split_group_ids"], data["episode_ids"]
    mode_ids, variants = data["motion_mode_ids"], data["motion_variant"]
    session_ids, valid = data["session_ids"], data["valid_target"].astype(bool)
    test_mask = np.isin(groups, test_groups)
    fit_mask = valid & ~test_mask
    coefficient = fit_first_order(states, actions, next_states, fit_mask)
    robot = load_robot_spec("configs/robots/so101.yaml")
    bundle = load_dynamics_bundle(args.checkpoint, args.normalizer, "gru", 5,
                                  "cuda" if torch.cuda.is_available() else "cpu",
                                  history_len=None, expected_robot_spec=robot)
    horizon = args.max_horizon
    history_len = bundle.history_len

    # ---- anchors (same window semantics as evaluate_real_model.py) ----
    anchors, anchor_modes, anchor_variants, mode_ok_list = [], [], [], []
    for index in range(history_len - 1, len(states) - horizon):
        window = slice(index - history_len + 1, index + horizon)
        if not (test_mask[index] and np.all(valid[index:index + horizon])):
            continue
        if not (np.all(episodes[window] == episodes[index]) and np.all(groups[window] == groups[index])):
            continue
        anchors.append(index)
        anchor_modes.append(int(mode_ids[index]))
        anchor_variants.append(str(variants[index]))
        mode_ok_list.append(bool(np.all(mode_ids[window] == mode_ids[index]) and
                                 np.all(variants[window] == variants[index])))
    if not anchors:
        raise SystemExit(f"no valid {horizon}-step test windows")
    anchors = np.asarray(anchors)
    anchor_modes = np.asarray(anchor_modes)
    anchor_variants = np.asarray(anchor_variants, dtype=object)
    mode_ok = np.asarray(mode_ok_list)
    n_windows = len(anchors)

    # ---- single rollout for the full horizon; per-step errors for every horizon ----
    histories = np.stack([np.concatenate((states[i - history_len + 1:i + 1], actions[i - history_len + 1:i + 1]), axis=1)
                          for i in anchors])
    future_actions = np.stack([actions[i:i + horizon] for i in anchors])
    truth = np.stack([[next_states[i + k] for k in range(horizon)] for i in anchors])
    learned = rollout_dynamics_batch(bundle.model, bundle.normalizer, bundle.model_type,
                                     torch.as_tensor(histories, device=bundle.device),
                                     torch.as_tensor(future_actions, device=bundle.device), bundle.state_dim,
                                     bundle.target_mode, bundle.control_dt, rollout_batch_size=1024)[:, 1:].cpu().numpy()
    # err[:, k] is the (k+1)-step-ahead prediction error (q|dq stack)
    err = learned - truth  # [W, H, 10]

    initial = states[anchors]
    persistence = np.repeat(initial[:, None, :], horizon, axis=1)
    constant_velocity = np.empty_like(truth)
    first_order = np.empty_like(truth)
    cv_state, fo_state = initial.copy(), initial.copy()
    for step in range(horizon):
        cv_state = cv_state.copy()
        cv_state[:, :5] += cv_state[:, 5:] * bundle.control_dt
        constant_velocity[:, step] = cv_state
        dq = np.stack([coefficient[j, 0] + coefficient[j, 1] * fo_state[:, 5 + j] +
                       coefficient[j, 2] * (future_actions[:, step, j] - fo_state[:, j]) for j in range(5)], axis=1)
        fo_state = np.concatenate((fo_state[:, :5] + dq * bundle.control_dt, dq), axis=1)
        first_order[:, step] = fo_state
    baselines = {"learned": learned, "persistence": persistence,
                 "constant_velocity": constant_velocity, "first_order": first_order}

    # ---- 1. horizon summary (MPC-relevant set) ----
    report_horizons = [h for h in REPORT_HORIZONS if h <= horizon]
    divergence_horizons = [h for h in DIVERGENCE_HORIZONS if h <= horizon]
    amplitude_h = min(AMPLITUDE_HORIZON, horizon)
    horizon_summary = {}
    for h in report_horizons:
        horizon_summary[str(h)] = {name: rmse_stats(baselines[name][:, h - 1] - truth[:, h - 1])
                                   for name in baselines}

    # ---- 2. per-step error-growth curve ----
    growth = {"steps": list(range(1, horizon + 1)),
              "time_s": [round(step * bundle.control_dt, 4) for step in range(1, horizon + 1)]}
    for name in baselines:
        e = baselines[name] - truth
        growth[f"{name}_q_rmse"] = [float(np.sqrt(np.mean(np.abs(e[:, k, :5]) ** 2))) for k in range(horizon)]
        growth[f"{name}_dq_rmse"] = [float(np.sqrt(np.mean(np.abs(e[:, k, 5:]) ** 2))) for k in range(horizon)]
    with open(curve_path, "w", encoding="utf-8") as fh:
        fh.write("step,time_s," + ",".join(f"{name}_q_rmse,{name}_dq_rmse" for name in baselines) + "\n")
        for k in range(horizon):
            fh.write(f"{k + 1},{growth['time_s'][k]:.4f}," +
                     ",".join(f"{growth[f'{name}_q_rmse'][k]:.8f},{growth[f'{name}_dq_rmse'][k]:.8f}"
                              for name in baselines) + "\n")

    # ---- 3. per-joint NMSE / R^2 ----
    nmse_horizons = [h for h in (1, 5, 10) if h <= horizon]
    nmse_r2 = {}
    for h in nmse_horizons:
        nmse_r2[str(h)] = {name: per_joint_nmse_r2(baselines[name][:, h - 1] - truth[:, h - 1], truth[:, h - 1])
                           for name in baselines}

    # ---- 4. amplitude ratio (commanded subset, H = amplitude_h) ----
    commanded_mask = anchor_modes != 0
    amp = {}
    for name in baselines:
        amp[name] = {}
    for j in range(5):
        amp_truth = np.std(truth[:, :amplitude_h, j], axis=1)
        keep = commanded_mask & (amp_truth > AMPLITUDE_FLOOR_RAD)
        for name in baselines:
            amp_pred = np.std(baselines[name][:, :amplitude_h, j], axis=1)
            ratio = amp_pred[keep] / amp_truth[keep]
            amp[name][JOINTS[j]] = {
                "median": float(np.median(ratio)), "p10": float(np.percentile(ratio, 10)),
                "p90": float(np.percentile(ratio, 90)), "n": int(np.sum(keep))}
    ratio_all = {name: [] for name in baselines}
    for name in baselines:
        for j in range(5):
            amp_truth = np.std(truth[:, :amplitude_h, j], axis=1)
            keep = commanded_mask & (amp_truth > AMPLITUDE_FLOOR_RAD)
            amp_pred = np.std(baselines[name][:, :amplitude_h, j], axis=1)
            ratio_all[name].extend((amp_pred[keep] / amp_truth[keep]).tolist())
    amp["aggregate"] = {name: {"median": float(np.median(np.asarray(v))),
                               "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
                               "n": len(v)} for name, v in ratio_all.items()}
    amp["horizon"] = amplitude_h
    amp["floor_deg"] = np.rad2deg(AMPLITUDE_FLOOR_RAD)
    amp["commanded_only"] = True

    # ---- 5. divergence rates ----
    divergence = {}
    for h in divergence_horizons:
        maxerr_deg = np.rad2deg(np.max(np.abs(err[:, h - 1, :5]), axis=1))
        divergence[str(h)] = {f"{th:.1f}deg": float(np.mean(maxerr_deg > th))
                              for th in DIVERGENCE_THRESHOLDS_DEG}
    headline_h0 = HEADLINE_DIVERGENCE[0] if HEADLINE_DIVERGENCE[0] <= horizon else max(divergence_horizons)
    headline_h = headline_h0
    headline_th = HEADLINE_DIVERGENCE[1]
    maxerr10_deg = np.rad2deg(np.max(np.abs(err[:, headline_h - 1, :5]), axis=1))
    diverged_idx = np.flatnonzero(maxerr10_deg > headline_th)

    # ---- 6. per-motion-mode breakdown ----
    mode_labels = {}
    for m in np.unique(anchor_modes):
        if int(m) == 3:  # split sine by variant
            for v in ("single_joint_sine", "multi_joint_sine"):
                mode_labels[f"{v}"] = (anchor_modes == m) & (anchor_variants == v) & mode_ok
        else:
            mode_labels[MODE_NAMES[int(m)]] = (anchor_modes == m) & mode_ok
    mode_breakdown = {}
    for h in [h for h in (6, 10) if h <= horizon]:
        mode_breakdown[str(h)] = {}
        for label, m in mode_labels.items():
            n = int(np.sum(m))
            mode_breakdown[str(h)][label] = {
                "n": n,
                **({"q_rmse": float(np.sqrt(np.mean(np.abs(err[m, h - 1, :5]) ** 2))),
                    "q_p95": float(np.percentile(np.abs(err[m, h - 1, :5]), 95)),
                    "dq_rmse": float(np.sqrt(np.mean(np.abs(err[m, h - 1, 5:]) ** 2)))} if n else {})}

    # ---- 7. commanded-motion subset ----
    commanded_summary = {}
    for h in report_horizons:
        all_rmse = float(np.sqrt(np.mean(np.abs(err[:, h - 1, :5]) ** 2)))
        cmd_rmse = float(np.sqrt(np.mean(np.abs(err[commanded_mask, h - 1, :5]) ** 2)))
        hold_rmse = float(np.sqrt(np.mean(np.abs(err[~commanded_mask, h - 1, :5]) ** 2)))
        commanded_summary[str(h)] = {"n_all": n_windows, "n_commanded": int(np.sum(commanded_mask)),
                                     "n_hold": n_windows - int(np.sum(commanded_mask)),
                                     "all_q_rmse": all_rmse, "commanded_q_rmse": cmd_rmse, "hold_q_rmse": hold_rmse}

    # ---- 8. worst-window analysis (H = headline) ----
    top_idx = np.argsort(-maxerr10_deg)[:TOP_WINDOWS]
    worst_rows = []
    for rank, wi in enumerate(top_idx, start=1):
        jmax = int(np.argmax(np.abs(err[wi, headline_h - 1, :5])))
        ep_start = int(np.searchsorted(episodes, episodes[anchors[wi]], side="left"))
        ep_end = int(np.searchsorted(episodes, episodes[anchors[wi]], side="right"))
        frac = float((anchors[wi] - ep_start) / max(1, ep_end - ep_start))
        mode_label = (f"{anchor_variants[wi]}" if anchor_modes[wi] == 3 else MODE_NAMES[anchor_modes[wi]])
        worst_rows.append({
            "rank": rank, "session": str(session_ids[anchors[wi]]), "episode": int(episodes[anchors[wi]]),
            "mode": mode_label, "worst_joint": JOINTS[jmax],
            "max_err_deg": float(maxerr10_deg[wi]),
            "window_frac_in_episode": round(frac, 4),
            "max_abs_dq_anchor_deg_s": float(np.rad2deg(np.max(np.abs(states[anchors[wi], 5:]))))})
    by_mode = {}
    by_joint = {}
    for row in worst_rows:
        by_mode[row["mode"]] = by_mode.get(row["mode"], 0) + 1
        by_joint[row["worst_joint"]] = by_joint.get(row["worst_joint"], 0) + 1
    worst = {"horizon": headline_h, "threshold_deg": headline_th, "total_diverged_windows": int(len(diverged_idx)),
             "top": worst_rows, "by_mode": by_mode, "by_joint": by_joint}

    # ---- 9. isolation and artifact identity ----
    per_group = {int(g): int(np.sum(groups[anchors] == g)) for g in sorted(set(test_groups.tolist()))}
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    manifest = json.loads(Path(str(args.dataset).replace(".npz", "") + ".manifest.json").read_text(encoding="utf-8"))
    plant_cfg = cfg.get("plant_identity", {})
    plant_manifest = manifest.get("plant_identity", {})
    compare_keys = ["calibration_sha256", "lerobot_commit", "pid", "acceleration", "maximum_acceleration",
                    "control_dt", "domain", "hardware", "joint_names", "hardware_config_sha256"]
    identity = {"per_test_group_window_counts": per_group,
                "checkpoint_test_group_ids": cfg.get("test_group_ids"),
                "checkpoint_validation_group_ids": cfg.get("validation_group_ids"),
                "requested_test_group_ids": test_groups.tolist(),
                "dataset_sha256_checkpoint": cfg.get("dataset_sha256"),
                "dataset_sha256_manifest": manifest.get("dataset", {}).get("sha256"),
                "dataset_sha256_match": bool(cfg.get("dataset_sha256") == manifest.get("dataset", {}).get("sha256")),
                "plant_identity_fields": {k: {"checkpoint": plant_cfg.get(k), "manifest": plant_manifest.get(k),
                                              "match": bool(plant_cfg.get(k) == plant_manifest.get(k))}
                                          for k in compare_keys}}
    isolation_pass = all([identity["dataset_sha256_match"]] +
                         [v["match"] for v in identity["plant_identity_fields"].values()] +
                         [cfg.get("test_group_ids") == test_groups.tolist(),
                          cfg.get("validation_group_ids") == [38, 39, 40, 41, 42]])

    payload = {
        "checkpoint": str(args.checkpoint), "normalizer": str(args.normalizer),
        "dataset": str(args.dataset), "control_dt": bundle.control_dt,
        "window_count": n_windows, "max_horizon": horizon, "history_len": int(history_len),
        "test_groups": test_groups.tolist(),
        "horizon_summary": horizon_summary, "error_growth_curve_csv": str(curve_path.name),
        "per_joint_nmse_r2": nmse_r2, "amplitude_ratio": amp, "divergence_rates": divergence,
        "motion_mode_breakdown": mode_breakdown, "commanded_motion": commanded_summary,
        "worst_windows": worst, "isolation_identity": identity, "isolation_pass": bool(isolation_pass),
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nreport written to {report_path}")
    print(f"error-growth CSV written to {curve_path}")


if __name__ == "__main__":
    main()
