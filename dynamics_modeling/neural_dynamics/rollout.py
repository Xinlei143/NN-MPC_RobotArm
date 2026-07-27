from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from neural_dynamics.integration import reconstruct_next_state
from neural_dynamics.normalization import StandardNormalizer
from neural_dynamics.train_utils import build_model, load_checkpoint
from mpc.robot_config import RobotSpec, file_sha256


@dataclass(frozen=True)
class DynamicsBundle:
    model: nn.Module
    normalizer: StandardNormalizer
    model_type: str
    history_len: int
    state_dim: int
    action_dim: int
    target_mode: str
    control_dt: float
    device: torch.device
    config: dict


def resolve_history_len(model_type: str, requested_history_len: int | None, config: dict) -> int:
    if model_type == "mlp":
        return 1
    if requested_history_len is not None and requested_history_len > 1:
        return int(requested_history_len)
    checkpoint_history_len = config.get("history_len")
    if checkpoint_history_len is not None:
        return int(checkpoint_history_len)
    return 1 if requested_history_len is None else int(requested_history_len)


def load_dynamics_bundle(
    checkpoint_path: str | Path,
    normalizer_path: str | Path,
    model_type: str,
    n_joints: int,
    device: str | torch.device,
    history_len: int | None = None,
    expected_robot_spec: RobotSpec | None = None,
) -> DynamicsBundle:
    device = torch.device(device)
    checkpoint = load_checkpoint(Path(checkpoint_path), map_location=device)
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}
    config = dict(config)
    if expected_robot_spec is not None:
        checkpoint_identity = config.get("robot_identity")
        if checkpoint_identity is None:
            digest = file_sha256(checkpoint_path)
            if not expected_robot_spec.is_legacy_artifact_allowed("checkpoint", digest):
                raise ValueError(
                    "Checkpoint is missing robot identity and is not an allowlisted legacy artifact"
                )
            config["legacy_identity"] = True
        elif checkpoint_identity != expected_robot_spec.artifact_identity():
            raise ValueError("Checkpoint robot identity does not match the active RobotSpec")

    state_dim = 2 * int(n_joints)
    action_dim = int(n_joints)
    checkpoint_state_dim = int(config.get("state_dim", state_dim))
    checkpoint_action_dim = int(config.get("action_dim", action_dim))
    if checkpoint_state_dim != state_dim:
        raise ValueError(f"Checkpoint state_dim={checkpoint_state_dim} does not match n_joints={n_joints}")
    if checkpoint_action_dim != action_dim:
        raise ValueError(f"Checkpoint action_dim={checkpoint_action_dim} does not match n_joints={n_joints}")
    checkpoint_model_type = str(config.get("model_type", model_type))
    if checkpoint_model_type != model_type:
        raise ValueError(
            f"Checkpoint model_type={checkpoint_model_type!r} does not match requested {model_type!r}"
        )

    resolved_history_len = resolve_history_len(model_type, history_len, config)
    output_dim = int(config.get("output_dim", state_dim))
    model = build_model(
        model_type,
        state_dim,
        action_dim,
        resolved_history_len,
        output_dim=output_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = StandardNormalizer.load(Path(normalizer_path), map_location=device)
    if expected_robot_spec is not None:
        normalizer_identity = normalizer.metadata.get("robot_identity")
        if normalizer_identity is None:
            digest = file_sha256(normalizer_path)
            if not expected_robot_spec.is_legacy_artifact_allowed("normalizer", digest):
                raise ValueError(
                    "Normalizer is missing robot identity and is not an allowlisted legacy artifact"
                )
            config["legacy_normalizer_identity"] = True
        elif normalizer_identity != expected_robot_spec.artifact_identity():
            raise ValueError("Normalizer robot identity does not match the active RobotSpec")
        checkpoint_dataset = config.get("dataset_manifest_sha256")
        normalizer_dataset = normalizer.metadata.get("dataset_manifest_sha256")
        if checkpoint_dataset is not None and checkpoint_dataset != normalizer_dataset:
            raise ValueError("Checkpoint and normalizer were produced from different dataset manifests")
        checkpoint_dt = float(config.get("control_dt", expected_robot_spec.expected_control_dt))
        if abs(checkpoint_dt - expected_robot_spec.expected_control_dt) > 1e-12:
            raise ValueError(
                "Checkpoint control_dt does not match the active RobotSpec: "
                f"{checkpoint_dt} != {expected_robot_spec.expected_control_dt}"
            )

    return DynamicsBundle(
        model=model,
        normalizer=normalizer,
        model_type=model_type,
        history_len=resolved_history_len,
        state_dim=state_dim,
        action_dim=action_dim,
        target_mode=str(config.get("target_mode", "delta_state")),
        control_dt=float(config.get("control_dt", 0.01)),
        device=device,
        config=config,
    )


def _as_batched_history(initial_history: torch.Tensor) -> torch.Tensor:
    if initial_history.ndim == 2:
        return initial_history.unsqueeze(0)
    if initial_history.ndim != 3:
        raise ValueError(f"initial_history must have shape [history, dim] or [batch, history, dim], got {tuple(initial_history.shape)}")
    return initial_history


def _rollout_dynamics_batch_no_chunk(
    model: nn.Module,
    normalizer: StandardNormalizer,
    model_type: str,
    initial_history: torch.Tensor,
    future_q_ref: torch.Tensor,
    state_dim: int,
    target_mode: str,
    control_dt: float,
) -> torch.Tensor:
    if future_q_ref.ndim != 3:
        raise ValueError(f"future_q_ref must have shape [batch, horizon, action_dim], got {tuple(future_q_ref.shape)}")
    history = _as_batched_history(initial_history)
    if history.shape[0] != future_q_ref.shape[0]:
        if history.shape[0] == 1:
            history = history.expand(future_q_ref.shape[0], -1, -1).clone()
        else:
            raise ValueError(f"history batch={history.shape[0]} does not match future_q_ref batch={future_q_ref.shape[0]}")
    if history.shape[-1] <= state_dim:
        raise ValueError("initial_history must contain concatenated [state, q_ref] entries")

    pred_state = history[:, -1, :state_dim]
    pred_states = [pred_state]
    n_joints = state_dim // 2

    with torch.no_grad():
        for step_idx in range(future_q_ref.shape[1]):
            action_i = future_q_ref[:, step_idx]
            if model_type == "mlp":
                model_input = normalizer.normalize_single_input(pred_state, action_i)
            else:
                # The learned model was trained on absolute q_ref actuator targets, so
                # replace the last token action with this candidate absolute q_ref.
                history = history.clone()
                history[:, -1, :state_dim] = pred_state
                history[:, -1, state_dim:] = action_i
                model_input = normalizer.normalize_sequence_input(history, state_dim)
            pred_target = normalizer.denormalize_delta(model(model_input))
            pred_state = reconstruct_next_state(pred_state, pred_target, target_mode, control_dt, n_joints)
            pred_states.append(pred_state)
            if model_type != "mlp" and step_idx + 1 < future_q_ref.shape[1]:
                next_entry = torch.cat([pred_state, future_q_ref[:, step_idx + 1]], dim=-1).unsqueeze(1)
                history = torch.cat([history[:, 1:], next_entry], dim=1)

    return torch.stack(pred_states, dim=1)


def rollout_dynamics_batch(
    model: nn.Module,
    normalizer: StandardNormalizer,
    model_type: str,
    initial_history: torch.Tensor,
    future_q_ref: torch.Tensor,
    state_dim: int,
    target_mode: str,
    control_dt: float,
    rollout_batch_size: int | None = None,
) -> torch.Tensor:
    if rollout_batch_size is None or rollout_batch_size <= 0 or future_q_ref.shape[0] <= rollout_batch_size:
        return _rollout_dynamics_batch_no_chunk(
            model,
            normalizer,
            model_type,
            initial_history,
            future_q_ref,
            state_dim,
            target_mode,
            control_dt,
        )

    history = _as_batched_history(initial_history)
    chunks: list[torch.Tensor] = []
    for start in range(0, future_q_ref.shape[0], rollout_batch_size):
        end = min(start + rollout_batch_size, future_q_ref.shape[0])
        history_chunk = history if history.shape[0] == 1 else history[start:end]
        chunks.append(
            _rollout_dynamics_batch_no_chunk(
                model,
                normalizer,
                model_type,
                history_chunk,
                future_q_ref[start:end],
                state_dim,
                target_mode,
                control_dt,
            )
        )
    return torch.cat(chunks, dim=0)
