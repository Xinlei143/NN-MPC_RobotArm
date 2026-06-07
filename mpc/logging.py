from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mpc.utils import write_csv_rows


def save_mpc_run(save_dir: Path, arrays: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / "rollout.npz", **arrays)
    write_csv_rows(save_dir / "rollout.csv", rows)
    plot_mpc_run(save_dir, arrays)


def plot_mpc_run(save_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    actual_states = arrays["actual_states"]
    q_des = arrays["q_des"]
    actuator_q_ref = arrays["actuator_q_ref"]
    planning_time = arrays["planning_time"]
    best_cost = arrays["best_cost"]
    n_joints = q_des.shape[1]
    time = np.arange(q_des.shape[0])

    fig_q, axes_q = plt.subplots(n_joints, 1, figsize=(10, 2.0 * n_joints), sharex=True)
    fig_dq, axes_dq = plt.subplots(n_joints, 1, figsize=(10, 2.0 * n_joints), sharex=True)
    if n_joints == 1:
        axes_q = [axes_q]
        axes_dq = [axes_dq]
    for idx in range(n_joints):
        axes_q[idx].plot(time, actual_states[:, idx], label="actual_q")
        axes_q[idx].plot(time, q_des[:, idx], label="q_des", linestyle="--")
        axes_q[idx].plot(time, actuator_q_ref[:, idx], label="actuator_q_ref", linestyle=":")
        axes_q[idx].set_ylabel(f"q{idx}")
        axes_q[idx].legend(fontsize=8)
        axes_dq[idx].plot(time, actual_states[:, n_joints + idx], label="actual_dq")
        axes_dq[idx].set_ylabel(f"dq{idx}")
        axes_dq[idx].legend(fontsize=8)
    axes_q[-1].set_xlabel("step")
    axes_dq[-1].set_xlabel("step")
    fig_q.tight_layout()
    fig_dq.tight_layout()
    fig_q.savefig(save_dir / "q_tracking.png", dpi=150)
    fig_dq.savefig(save_dir / "dq.png", dpi=150)
    plt.close(fig_q)
    plt.close(fig_dq)

    tracking_error = np.linalg.norm(actual_states[:, :n_joints] - q_des, axis=1)
    fig_err, ax_err = plt.subplots(figsize=(10, 4))
    ax_err.plot(time, tracking_error)
    ax_err.set_xlabel("step")
    ax_err.set_ylabel("||q - q_des||")
    fig_err.tight_layout()
    fig_err.savefig(save_dir / "tracking_error.png", dpi=150)
    plt.close(fig_err)

    fig_ctrl, axes_ctrl = plt.subplots(n_joints, 1, figsize=(10, 2.0 * n_joints), sharex=True)
    if n_joints == 1:
        axes_ctrl = [axes_ctrl]
    for idx in range(n_joints):
        axes_ctrl[idx].plot(time, actuator_q_ref[:, idx])
        axes_ctrl[idx].set_ylabel(f"q_ref{idx}")
    axes_ctrl[-1].set_xlabel("step")
    fig_ctrl.tight_layout()
    fig_ctrl.savefig(save_dir / "control.png", dpi=150)
    plt.close(fig_ctrl)

    fig_diag, axes_diag = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes_diag[0].plot(time, planning_time)
    axes_diag[0].set_ylabel("planning_time_s")
    axes_diag[1].plot(time, best_cost)
    axes_diag[1].set_ylabel("best_cost")
    axes_diag[1].set_xlabel("step")
    fig_diag.tight_layout()
    fig_diag.savefig(save_dir / "planning_diagnostics.png", dpi=150)
    plt.close(fig_diag)
