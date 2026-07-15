from __future__ import annotations

import numpy as np
import torch

from neural_dynamics.rollout import rollout_dynamics_batch


def _history_batch(
    states: np.ndarray,
    q_ref_history: np.ndarray,
    history_len: int,
    anchors: int,
    device: torch.device,
) -> torch.Tensor:
    entries = np.concatenate([states, q_ref_history], axis=1).astype(np.float32)
    histories: list[np.ndarray] = []
    for anchor in range(anchors):
        start = max(0, anchor - history_len + 1)
        history = entries[start : anchor + 1]
        if history.shape[0] < history_len:
            history = np.concatenate([np.repeat(history[:1], history_len - history.shape[0], axis=0), history], axis=0)
        histories.append(history)
    return torch.as_tensor(np.stack(histories), dtype=torch.float32, device=device)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    x = left[valid]
    y = right[valid]
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def replay_executed_commands(
    *,
    model: torch.nn.Module,
    normalizer: object,
    model_type: str,
    state_dim: int,
    target_mode: str,
    control_dt: float,
    history_len: int,
    states_history: list[np.ndarray],
    q_ref_history: list[np.ndarray],
    executed_q_ref: list[np.ndarray],
    horizon: int,
    device: torch.device,
    rollout_batch_size: int | None,
    command_velocity: list[np.ndarray] | None = None,
    command_acceleration: list[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Replay the learned model under commands actually executed after each state.

    The first dimension is the original closed-loop control index. Tail indices
    without a complete future horizon are deliberately NaN-padded rather than
    being compared against a shorter, differently defined rollout.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    successful_steps = min(len(executed_q_ref), len(states_history) - 1, len(q_ref_history) - 1)
    q_error = np.full((max(successful_steps, 0), horizon), np.nan, dtype=np.float32)
    dq_error = np.full_like(q_error, np.nan)
    valid_horizon = np.zeros(max(successful_steps, 0), dtype=np.int64)
    if successful_steps < horizon:
        return {
            "replay_q_error_norm": q_error,
            "replay_dq_error_norm": dq_error,
            "replay_valid_horizon": valid_horizon,
            "replay_q_error_command_velocity_corr": np.asarray(float("nan"), dtype=np.float32),
            "replay_q_error_command_acceleration_corr": np.asarray(float("nan"), dtype=np.float32),
        }

    anchors = successful_steps - horizon + 1
    state_history_array = np.asarray(states_history[: successful_steps + 1], dtype=np.float32)
    q_ref_history_array = np.asarray(q_ref_history[: successful_steps + 1], dtype=np.float32)
    history = _history_batch(state_history_array, q_ref_history_array, history_len, anchors, device)
    future_q_ref = np.stack(
        [np.asarray(executed_q_ref[index : index + horizon], dtype=np.float32) for index in range(anchors)], axis=0
    )
    predicted = rollout_dynamics_batch(
        model=model,
        normalizer=normalizer,
        model_type=model_type,
        initial_history=history,
        future_q_ref=torch.as_tensor(future_q_ref, dtype=torch.float32, device=device),
        state_dim=state_dim,
        target_mode=target_mode,
        control_dt=control_dt,
        rollout_batch_size=rollout_batch_size,
    ).detach().cpu().numpy()
    actual_future = np.stack(
        [state_history_array[index + 1 : index + horizon + 1] for index in range(anchors)], axis=0
    )
    n_joints = state_dim // 2
    q_error[:anchors] = np.linalg.norm(predicted[:, 1:, :n_joints] - actual_future[:, :, :n_joints], axis=2)
    dq_error[:anchors] = np.linalg.norm(predicted[:, 1:, n_joints:] - actual_future[:, :, n_joints:], axis=2)
    valid_horizon[:anchors] = horizon

    first_step_q_error = q_error[:anchors, 0]
    command_velocity_norm = (
        np.linalg.norm(np.asarray(command_velocity[:anchors], dtype=np.float32), axis=1)
        if command_velocity is not None
        else np.full(anchors, np.nan, dtype=np.float32)
    )
    command_acceleration_norm = (
        np.linalg.norm(np.asarray(command_acceleration[:anchors], dtype=np.float32), axis=1)
        if command_acceleration is not None
        else np.full(anchors, np.nan, dtype=np.float32)
    )
    return {
        "replay_q_error_norm": q_error,
        "replay_dq_error_norm": dq_error,
        "replay_valid_horizon": valid_horizon,
        "replay_q_error_command_velocity_corr": np.asarray(_correlation(first_step_q_error, command_velocity_norm), dtype=np.float32),
        "replay_q_error_command_acceleration_corr": np.asarray(
            _correlation(first_step_q_error, command_acceleration_norm), dtype=np.float32
        ),
    }
