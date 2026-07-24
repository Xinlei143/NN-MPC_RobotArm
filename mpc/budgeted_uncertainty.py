"""Budgeted, selected-trajectory uncertainty supervision for residual MPC.

Model A remains the only model used to optimise CEM.  The ensemble evaluates
the one selected executable trajectory after CEM and reports disagreement as
an uncertainty signal; a caller may monitor it, attenuate residual authority,
or invoke a separate high-risk fallback policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
import torch

from neural_dynamics.rollout import DynamicsBundle, load_dynamics_bundle, rollout_dynamics_batch


def normalized_rms_disagreement(predictions: torch.Tensor, state_scale: torch.Tensor) -> torch.Tensor:
    """Return normalized inter-model RMS disagreement per candidate."""
    if predictions.ndim != 4 or predictions.shape[0] < 2:
        raise ValueError("predictions must have shape [models>=2, candidates, horizon+1, state]")
    if state_scale.ndim != 1 or state_scale.shape[0] != predictions.shape[-1]:
        raise ValueError("state_scale must have one value per state dimension")
    if not bool(torch.all(torch.isfinite(predictions))) or not bool(torch.all(torch.isfinite(state_scale))):
        raise ValueError("predictions and state_scale must be finite")
    if bool(torch.any(state_scale <= 0)):
        raise ValueError("state_scale must be positive")
    normalized_std = predictions[:, :, 1:, :].std(dim=0, unbiased=False) / state_scale.view(1, 1, -1)
    return torch.sqrt(torch.mean(torch.square(normalized_std), dim=(1, 2)))


class DynamicsEnsemble:
    """Primary Model A plus compatible replicas used only after CEM selection."""

    def __init__(self, bundles: list[DynamicsBundle], device: torch.device) -> None:
        if len(bundles) < 2:
            raise ValueError("uncertainty ensemble requires a primary model and at least one replica")
        primary = bundles[0]
        if any(
            bundle.model_type != primary.model_type
            or bundle.state_dim != primary.state_dim
            or bundle.target_mode != primary.target_mode
            or bundle.history_len != primary.history_len
            or not np.isclose(bundle.control_dt, primary.control_dt)
            for bundle in bundles[1:]
        ):
            raise ValueError("all ensemble members must share model type, state dimensions, history length and control_dt")
        self.bundles = bundles
        self.device = device
        if primary.normalizer.state_std is None:
            raise RuntimeError("primary normalizer does not contain state_std")
        self.state_scale = primary.normalizer.state_std.to(device=device, dtype=torch.float32)

    @classmethod
    def from_replica_paths(
        cls,
        primary: DynamicsBundle,
        checkpoint_paths: list[Path],
        normalizer_paths: list[Path],
        device: torch.device,
    ) -> "DynamicsEnsemble":
        if len(checkpoint_paths) != len(normalizer_paths):
            raise ValueError("uncertainty checkpoint and normalizer counts must match")
        replicas = [
            load_dynamics_bundle(
                checkpoint_path=checkpoint,
                normalizer_path=normalizer,
                model_type=primary.model_type,
                n_joints=primary.state_dim // 2,
                device=device,
                history_len=primary.history_len,
            )
            for checkpoint, normalizer in zip(checkpoint_paths, normalizer_paths)
        ]
        return cls([primary, *replicas], device)


@dataclass(frozen=True)
class BudgetedUncertainty:
    selected_score: float
    max_candidate_score: float
    candidate_scores: dict[str, float]
    evaluation_time_s: float
    timed_out: bool


def selected_cem_candidates(result: object, *, horizon: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return only the selected executable command and cached Model-A rollout.

    The gate is driven exclusively by the selected trajectory.  Evaluating
    best/mean diagnostic branches online used GPU budget without influencing a
    control decision, so those diagnostics are intentionally offline-only.
    """
    if horizon <= 0:
        raise ValueError("uncertainty horizon must be positive")
    selected_sequence = np.asarray(getattr(result, "selected_q_ref_sequence"), dtype=np.float32)
    selected_prediction = np.asarray(getattr(result, "selected_predicted_state_sequence"), dtype=np.float32)
    if selected_sequence.ndim != 2 or selected_prediction.ndim != 2 or selected_prediction.shape[0] < horizon + 1:
        raise ValueError("CEM result lacks a valid selected trajectory for budgeted uncertainty")
    return (
        {"selected": selected_sequence[:horizon].copy()},
        {"selected": selected_prediction[: horizon + 1].copy()},
    )


def soft_residual_scale(score: float, low_threshold: float, high_threshold: float) -> float:
    """Map finite ensemble disagreement to a continuous residual authority."""
    if not np.isfinite(score):
        return 0.0
    if not 0.0 < low_threshold < high_threshold:
        raise ValueError("uncertainty thresholds must satisfy 0 < low < high")
    if score <= low_threshold:
        return 1.0
    if score >= high_threshold:
        return 0.0
    return float((high_threshold - score) / (high_threshold - low_threshold))


def high_risk_prediction(
    predicted_state_sequence: np.ndarray,
    reference_q_sequence: np.ndarray,
    residual_sequence: np.ndarray,
    residual_max: np.ndarray,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
    *,
    joint_limit_margin: float,
    residual_saturation_fraction: float,
    current_tracking_error: float,
    tracking_error_growth_ratio: float,
    min_tracking_error: float,
) -> bool:
    """Return whether a high-disagreement plan also has a concrete risk signal."""
    if not 0.0 < residual_saturation_fraction <= 1.0:
        raise ValueError("residual_saturation_fraction must be in (0, 1]")
    if tracking_error_growth_ratio <= 1.0 or min_tracking_error < 0.0:
        raise ValueError("tracking-error risk parameters are invalid")
    predicted = np.asarray(predicted_state_sequence, dtype=np.float32)
    reference = np.asarray(reference_q_sequence, dtype=np.float32)
    if predicted.ndim != 2 or reference.ndim != 2 or predicted.shape[0] < 2:
        raise ValueError("predicted/reference trajectories are invalid")
    n_joints = joint_low.shape[0]
    future_q = predicted[1:, :n_joints]
    reference = reference[: future_q.shape[0]]
    if reference.shape != future_q.shape:
        raise ValueError("reference trajectory does not match predicted trajectory")
    lower = np.asarray(joint_low, dtype=np.float32) + joint_limit_margin
    upper = np.asarray(joint_high, dtype=np.float32) - joint_limit_margin
    predicted_limit_risk = bool(np.any(future_q < lower) or np.any(future_q > upper))
    residual_saturated = bool(np.any(np.abs(residual_sequence) >= residual_saturation_fraction * residual_max))
    predicted_error = float(np.linalg.norm(future_q[0] - reference[0]))
    error_growing = bool(
        current_tracking_error >= min_tracking_error
        and predicted_error >= tracking_error_growth_ratio * current_tracking_error
    )
    return predicted_limit_risk or residual_saturated or error_growing


@torch.inference_mode()
def evaluate_replicas_with_primary_predictions(
    ensemble: DynamicsEnsemble,
    initial_history: torch.Tensor,
    candidate_q_ref_sequences: Mapping[str, np.ndarray],
    primary_predictions: Mapping[str, np.ndarray],
    *,
    budget_ms: float,
    selected_key: str = "selected",
) -> BudgetedUncertainty:
    """Evaluate compatible replicas under a hard-safe, cooperative wall-clock budget.

    CUDA kernels cannot be safely cancelled mid-launch.  The routine therefore
    checks the elapsed synchronized time between replica calls; if a call
    consumes the budget, it returns ``timed_out`` and the caller must use the
    nominal IK command rather than trusting a partial disagreement estimate.
    """
    if budget_ms <= 0.0:
        raise ValueError("uncertainty budget must be positive")
    if selected_key not in candidate_q_ref_sequences or selected_key not in primary_predictions:
        raise ValueError("selected candidate and primary prediction are required")
    names = list(candidate_q_ref_sequences)
    sequences = np.stack([np.asarray(candidate_q_ref_sequences[name], dtype=np.float32) for name in names])
    primary = np.stack([np.asarray(primary_predictions[name], dtype=np.float32) for name in names])
    if sequences.ndim != 3 or primary.ndim != 3 or primary.shape[0] != sequences.shape[0] or primary.shape[1] != sequences.shape[1] + 1:
        raise ValueError("candidate commands and primary predictions have incompatible shapes")
    future_q_ref = torch.as_tensor(sequences, dtype=torch.float32, device=ensemble.device)
    primary_tensor = torch.as_tensor(primary, dtype=torch.float32, device=ensemble.device)
    torch.cuda.synchronize(ensemble.device)
    started = perf_counter()
    replica_predictions: list[torch.Tensor] = []
    for replica in ensemble.bundles[1:]:
        if 1e3 * (perf_counter() - started) >= budget_ms:
            return BudgetedUncertainty(float("nan"), float("nan"), {}, perf_counter() - started, True)
        predicted = rollout_dynamics_batch(
            model=replica.model,
            normalizer=replica.normalizer,
            model_type=replica.model_type,
            initial_history=initial_history,
            future_q_ref=future_q_ref,
            state_dim=replica.state_dim,
            target_mode=replica.target_mode,
            control_dt=replica.control_dt,
            rollout_batch_size=future_q_ref.shape[0],
        )
        torch.cuda.synchronize(ensemble.device)
        replica_predictions.append(predicted)
        if 1e3 * (perf_counter() - started) >= budget_ms:
            return BudgetedUncertainty(float("nan"), float("nan"), {}, perf_counter() - started, True)
    predictions = torch.stack([primary_tensor, *replica_predictions], dim=0)
    scores = normalized_rms_disagreement(predictions, ensemble.state_scale)
    score_map = {name: float(scores[index].detach().cpu()) for index, name in enumerate(names)}
    return BudgetedUncertainty(
        selected_score=score_map[selected_key],
        max_candidate_score=max(score_map.values()),
        candidate_scores=score_map,
        evaluation_time_s=perf_counter() - started,
        timed_out=False,
    )
