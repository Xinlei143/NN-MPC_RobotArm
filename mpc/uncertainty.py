"""Lightweight deep-ensemble uncertainty checks for residual MPC.

The primary dynamics model remains the only model used by CEM to score its
full candidate population.  This module is deliberately used after CEM has
selected a small number of executable command sequences, so uncertainty
monitoring does not multiply the real-time planning cost by the population
size.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
import torch

from neural_dynamics.rollout import DynamicsBundle, load_dynamics_bundle, rollout_dynamics_batch


@dataclass(frozen=True)
class EnsembleUncertainty:
    """Disagreement diagnostics for one post-CEM evaluation."""

    selected_score: float
    max_candidate_score: float
    candidate_scores: dict[str, float]
    evaluation_time_s: float
    selected_mean_prediction: np.ndarray


def normalized_rms_disagreement(predictions: torch.Tensor, state_scale: torch.Tensor) -> torch.Tensor:
    """Return RMS inter-model disagreement for every candidate trajectory.

    ``predictions`` has shape ``[models, candidates, horizon_plus_one, state]``.
    The common initial state is removed because it has zero epistemic
    uncertainty by construction.  Dividing by the baseline training-state
    standard deviation makes a single threshold meaningful across q and dq.
    """
    if predictions.ndim != 4 or predictions.shape[0] < 2:
        raise ValueError("predictions must have shape [models>=2, candidates, horizon+1, state]")
    if state_scale.ndim != 1 or state_scale.shape[0] != predictions.shape[-1]:
        raise ValueError("state_scale must have one positive value per state dimension")
    if not bool(torch.all(torch.isfinite(predictions))) or not bool(torch.all(torch.isfinite(state_scale))):
        raise ValueError("predictions and state_scale must be finite")
    if bool(torch.any(state_scale <= 0)):
        raise ValueError("state_scale must be positive")
    normalized_std = predictions[:, :, 1:, :].std(dim=0, unbiased=False) / state_scale.view(1, 1, -1)
    return torch.sqrt(torch.mean(torch.square(normalized_std), dim=(1, 2)))


class DynamicsEnsemble:
    """Post-planning ensemble evaluator with a fixed primary baseline member."""

    def __init__(self, bundles: list[DynamicsBundle], device: torch.device) -> None:
        if len(bundles) < 2:
            raise ValueError("uncertainty ensemble requires a baseline plus at least one replica")
        baseline = bundles[0]
        if any(
            bundle.model_type != baseline.model_type
            or bundle.state_dim != baseline.state_dim
            or bundle.target_mode != baseline.target_mode
            or bundle.history_len != baseline.history_len
            or not np.isclose(bundle.control_dt, baseline.control_dt)
            for bundle in bundles[1:]
        ):
            raise ValueError("all ensemble members must share model type, dimensions, history length and control_dt")
        self.bundles = bundles
        self.device = device
        self.state_scale = baseline.normalizer.state_std
        if self.state_scale is None:
            raise RuntimeError("baseline normalizer does not contain state_std")
        self.state_scale = self.state_scale.to(device=device, dtype=torch.float32)

    @property
    def size(self) -> int:
        return len(self.bundles)

    @classmethod
    def from_replica_paths(
        cls,
        baseline: DynamicsBundle,
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
                model_type=baseline.model_type,
                n_joints=baseline.state_dim // 2,
                device=device,
                history_len=baseline.history_len,
            )
            for checkpoint, normalizer in zip(checkpoint_paths, normalizer_paths)
        ]
        return cls([baseline, *replicas], device)

    @torch.inference_mode()
    def evaluate(
        self,
        initial_history: torch.Tensor,
        candidate_q_ref_sequences: Mapping[str, np.ndarray],
        *,
        selected_key: str = "selected",
    ) -> EnsembleUncertainty:
        if not candidate_q_ref_sequences:
            raise ValueError("at least one candidate command sequence is required")
        if selected_key not in candidate_q_ref_sequences:
            raise ValueError(f"selected candidate {selected_key!r} is missing")
        names = list(candidate_q_ref_sequences)
        sequences = np.stack([np.asarray(candidate_q_ref_sequences[name], dtype=np.float32) for name in names])
        if sequences.ndim != 3:
            raise ValueError("candidate command sequences must have shape [horizon, action_dim]")
        future_q_ref = torch.as_tensor(sequences, dtype=torch.float32, device=self.device)
        start = perf_counter()
        predictions = torch.stack(
            [
                rollout_dynamics_batch(
                    model=bundle.model,
                    normalizer=bundle.normalizer,
                    model_type=bundle.model_type,
                    initial_history=initial_history,
                    future_q_ref=future_q_ref,
                    state_dim=bundle.state_dim,
                    target_mode=bundle.target_mode,
                    control_dt=bundle.control_dt,
                    rollout_batch_size=future_q_ref.shape[0],
                )
                for bundle in self.bundles
            ],
            dim=0,
        )
        scores = normalized_rms_disagreement(predictions, self.state_scale)
        score_map = {name: float(scores[index].detach().cpu()) for index, name in enumerate(names)}
        selected_index = names.index(selected_key)
        return EnsembleUncertainty(
            selected_score=score_map[selected_key],
            max_candidate_score=max(score_map.values()),
            candidate_scores=score_map,
            evaluation_time_s=perf_counter() - start,
            selected_mean_prediction=predictions[:, selected_index].mean(dim=0).detach().cpu().numpy().astype(np.float32),
        )


def selected_branch_sequences(result: object) -> dict[str, np.ndarray]:
    """Extract baseline/best/mean/selected command sequences from a CEM result.

    CEM records the branches used by ``lowest_cost`` selection.  Older result
    objects may not contain every role, so the chosen sequence is always
    included explicitly and remains sufficient for a safe gate.
    """
    sequences: dict[str, np.ndarray] = {
        "selected": np.asarray(getattr(result, "selected_q_ref_sequence"), dtype=np.float32)
    }
    for branch in getattr(result, "branch_candidates", ()):
        sequence = np.asarray(branch.q_ref_sequence, dtype=np.float32)
        for role in branch.role_mask:
            if role in {"baseline", "best", "mean"}:
                sequences.setdefault(role, sequence)
    return sequences
