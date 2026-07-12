from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mpc.utils import write_csv_rows


def save_mpc_run(save_dir: Path, arrays: dict[str, np.ndarray], rows: list[dict[str, Any]]) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / "rollout.npz", **arrays)
    write_csv_rows(save_dir / "rollout.csv", rows)
    summary = _task_tracking_summary(arrays)
    if summary is not None:
        with (save_dir / "task_tracking_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    plot_mpc_run(save_dir, arrays)


def _task_arrays(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, ...] | None:
    required = (
        "desired_ee_positions",
        "desired_ee_rotations",
        "actual_ee_positions",
        "actual_ee_rotations",
        "ee_position_errors",
        "ee_orientation_errors",
        "segment_ids",
        "lap_ids",
    )
    if not all(name in arrays for name in required):
        return None

    values = tuple(np.asarray(arrays[name]) for name in required)
    desired_position, desired_rotation, actual_position, actual_rotation, *_ = values
    if (
        desired_position.ndim != 2
        or desired_position.shape[1:] != (3,)
        or actual_position.shape != desired_position.shape
        or desired_rotation.ndim != 3
        or desired_rotation.shape[1:] != (3, 3)
        or actual_rotation.shape != desired_rotation.shape
    ):
        return None
    lengths = [value.shape[0] for value in values]
    if not lengths or min(lengths) == 0:
        return None
    length = min(lengths)
    return tuple(value[:length] for value in values)


def _error_metrics(position_errors: np.ndarray, orientation_errors: np.ndarray) -> dict[str, float | int]:
    return {
        "samples": int(position_errors.shape[0]),
        "position_rmse_m": float(np.sqrt(np.mean(np.square(position_errors)))),
        "orientation_rmse_rad": float(np.sqrt(np.mean(np.square(orientation_errors)))),
        "max_position_error_m": float(np.max(position_errors)),
        "max_orientation_error_rad": float(np.max(orientation_errors)),
    }


def _metrics_by_id(
    identifiers: np.ndarray,
    position_errors: np.ndarray,
    orientation_errors: np.ndarray,
    *,
    exclude_negative_ids: bool = False,
) -> dict[str, dict[str, float | int]]:
    unique_ids = np.unique(identifiers)
    if exclude_negative_ids:
        unique_ids = unique_ids[unique_ids >= 0]
    summary: dict[str, dict[str, float | int]] = {}
    for identifier in unique_ids:
        mask = identifiers == identifier
        summary[str(int(identifier))] = _error_metrics(position_errors[mask], orientation_errors[mask])
    return summary


def _task_tracking_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any] | None:
    task_arrays = _task_arrays(arrays)
    if task_arrays is None:
        return None
    (
        _desired_position,
        _desired_rotation,
        _actual_position,
        _actual_rotation,
        position_errors,
        orientation_errors,
        segment_ids,
        lap_ids,
    ) = task_arrays
    steps = position_errors.shape[0]
    summary: dict[str, Any] = {
        "recorded_steps": int(steps),
        "overall": _error_metrics(position_errors, orientation_errors),
        "segments": _metrics_by_id(segment_ids, position_errors, orientation_errors),
        "laps": _metrics_by_id(
            lap_ids,
            position_errors,
            orientation_errors,
            exclude_negative_ids=True,
        ),
        "final": {
            "tcp_position_error_m": float(position_errors[-1]),
            "tcp_orientation_error_rad": float(orientation_errors[-1]),
        },
    }

    actual_states = np.asarray(arrays.get("actual_states", np.empty((0,))))
    q_des = np.asarray(arrays.get("q_des", np.empty((0,))))
    if actual_states.ndim == 2 and q_des.ndim == 2 and actual_states.shape[1] >= q_des.shape[1]:
        joint_length = min(steps, actual_states.shape[0], q_des.shape[0])
        if joint_length:
            joint_error = actual_states[:joint_length, : q_des.shape[1]] - q_des[:joint_length]
            summary["joint_tracking"] = {
                "position_rmse_rad": float(np.sqrt(np.mean(np.square(joint_error)))),
                "max_position_error_rad": float(np.max(np.abs(joint_error))),
                "final_position_error_inf_rad": float(np.max(np.abs(joint_error[-1]))),
            }

    planning_time = np.asarray(arrays.get("planning_time", np.empty((0,))), dtype=np.float64)
    failure_flags = np.asarray(arrays.get("failure_flags", np.empty((0,))), dtype=np.float64)
    limit_flags = np.asarray(arrays.get("joint_limit_violation_flags", np.empty((0,))), dtype=np.float64)
    planning: dict[str, float | int] = {}
    if planning_time.size:
        planning["mean_planning_time_s"] = float(np.mean(planning_time))
        planning["max_planning_time_s"] = float(np.max(planning_time))
    if failure_flags.size:
        planning["failure_count"] = int(np.sum(failure_flags != 0.0))
        planning["failure_rate"] = float(np.mean(failure_flags != 0.0))
    if limit_flags.size:
        planning["joint_limit_violation_count"] = int(np.sum(limit_flags != 0.0))
        planning["joint_limit_violation_rate"] = float(np.mean(limit_flags != 0.0))
    if planning:
        summary["planning"] = planning
    return summary


def _plane_projection(desired_position: np.ndarray, actual_position: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Project TCP paths into their dominant desired-trajectory plane when possible."""
    centered = desired_position - desired_position.mean(axis=0, keepdims=True)
    try:
        _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        singular_values = np.empty(0)
        right_vectors = np.empty((0, 3))
    if singular_values.size >= 2 and singular_values[1] > 1e-10:
        axes = right_vectors[:2]
        return centered @ axes.T, (actual_position - desired_position.mean(axis=0, keepdims=True)) @ axes.T, "plane axis 1 (m)", "plane axis 2 (m)"
    return desired_position[:, :2], actual_position[:, :2], "world x (m)", "world y (m)"


def _plot_group_tracking_summary(
    plt: Any,
    save_path: Path,
    group_ids: np.ndarray,
    position_errors: np.ndarray,
    orientation_errors: np.ndarray,
    title: str,
    exclude_negative_ids: bool = False,
) -> None:
    unique_ids = np.unique(group_ids)
    if exclude_negative_ids:
        unique_ids = unique_ids[unique_ids >= 0]
    if unique_ids.size == 0:
        return

    position_rmse = []
    orientation_rmse = []
    for group_id in unique_ids:
        mask = group_ids == group_id
        position_rmse.append(float(np.sqrt(np.mean(np.square(position_errors[mask])))))
        orientation_rmse.append(float(np.sqrt(np.mean(np.square(orientation_errors[mask])))))

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    labels = [str(int(group_id)) for group_id in unique_ids]
    axes[0].bar(labels, position_rmse)
    axes[0].set_ylabel("TCP position RMSE (m)")
    axes[0].set_title(title)
    axes[1].bar(labels, orientation_rmse)
    axes[1].set_ylabel("TCP orientation RMSE (rad)")
    axes[1].set_xlabel("lap id" if exclude_negative_ids else "segment id")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_task_space_run(plt: Any, save_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    task_arrays = _task_arrays(arrays)
    if task_arrays is None:
        return
    (
        desired_position,
        _desired_rotation,
        actual_position,
        _actual_rotation,
        position_errors,
        orientation_errors,
        segment_ids,
        lap_ids,
    ) = task_arrays
    time = np.arange(desired_position.shape[0])

    fig_3d = plt.figure(figsize=(8, 7))
    axis_3d = fig_3d.add_subplot(111, projection="3d")
    axis_3d.plot(*desired_position.T, label="desired TCP", linestyle="--")
    axis_3d.plot(*actual_position.T, label="actual TCP")
    axis_3d.set_xlabel("x (m)")
    axis_3d.set_ylabel("y (m)")
    axis_3d.set_zlabel("z (m)")
    axis_3d.legend()
    fig_3d.tight_layout()
    fig_3d.savefig(save_dir / "ee_trajectory_3d.png", dpi=150)
    plt.close(fig_3d)

    desired_projection, actual_projection, horizontal_label, vertical_label = _plane_projection(desired_position, actual_position)
    fig_projection, axis_projection = plt.subplots(figsize=(7, 6))
    axis_projection.plot(desired_projection[:, 0], desired_projection[:, 1], label="desired TCP", linestyle="--")
    axis_projection.plot(actual_projection[:, 0], actual_projection[:, 1], label="actual TCP")
    axis_projection.set_xlabel(horizontal_label)
    axis_projection.set_ylabel(vertical_label)
    axis_projection.axis("equal")
    axis_projection.legend()
    fig_projection.tight_layout()
    fig_projection.savefig(save_dir / "ee_xy_or_plane_projection.png", dpi=150)
    plt.close(fig_projection)

    fig_position, axes_position = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for index, label in enumerate(("x", "y", "z")):
        axes_position[index].plot(time, desired_position[:, index], label="desired")
        axes_position[index].plot(time, actual_position[:, index], label="actual", linestyle="--")
        axes_position[index].set_ylabel(f"{label} (m)")
        axes_position[index].legend(fontsize=8)
    axes_position[-1].set_xlabel("step")
    fig_position.tight_layout()
    fig_position.savefig(save_dir / "ee_position_tracking.png", dpi=150)
    plt.close(fig_position)

    fig_position_error, axis_position_error = plt.subplots(figsize=(10, 4))
    axis_position_error.plot(time, position_errors)
    axis_position_error.set_xlabel("step")
    axis_position_error.set_ylabel("TCP position error (m)")
    fig_position_error.tight_layout()
    fig_position_error.savefig(save_dir / "ee_position_error.png", dpi=150)
    plt.close(fig_position_error)

    fig_orientation_error, axis_orientation_error = plt.subplots(figsize=(10, 4))
    axis_orientation_error.plot(time, orientation_errors)
    axis_orientation_error.set_xlabel("step")
    axis_orientation_error.set_ylabel("TCP orientation error (rad)")
    fig_orientation_error.tight_layout()
    fig_orientation_error.savefig(save_dir / "ee_orientation_error.png", dpi=150)
    plt.close(fig_orientation_error)

    _plot_group_tracking_summary(
        plt,
        save_dir / "segment_tracking_summary.png",
        segment_ids,
        position_errors,
        orientation_errors,
        title="TCP tracking by reference segment",
    )
    _plot_group_tracking_summary(
        plt,
        save_dir / "lap_tracking_summary.png",
        lap_ids,
        position_errors,
        orientation_errors,
        title="TCP tracking by shape lap",
        exclude_negative_ids=True,
    )


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

    _plot_task_space_run(plt, save_dir, arrays)
