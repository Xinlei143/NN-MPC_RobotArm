from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

import numpy as np
import torch


class PlannerProtocol(Protocol):
    def evaluate(self, candidate_delta_q_ref: torch.Tensor) -> dict[str, torch.Tensor]:
        ...


@dataclass(frozen=True)
class CEMMPCConfig:
    horizon: int
    action_dim: int
    num_samples: int = 1024
    num_elites: int | None = None
    elite_ratio: float = 0.08
    cem_iters: int = 4
    init_std: float = 0.15
    min_std: float = 0.01
    smoothing_alpha: float = 0.2
    temporal_noise_alpha: float = 0.8
    seed: int = 0
    device: str = "cpu"
    execute: str = "best"


@dataclass
class CEMMPCResult:
    q_ref: np.ndarray
    delta_q_ref: np.ndarray
    best_cost: float
    elite_mean_cost: float
    planning_time: float
    failure: bool
    failure_reason: str
    best_sequence: np.ndarray


class CEMMPCController:
    def __init__(
        self,
        config: CEMMPCConfig,
        planner: PlannerProtocol,
        joint_low: np.ndarray,
        joint_high: np.ndarray,
    ) -> None:
        if config.horizon <= 0 or config.action_dim <= 0:
            raise ValueError("horizon and action_dim must be positive")
        if config.num_samples <= 0 or config.cem_iters <= 0:
            raise ValueError("num_samples and cem_iters must be positive")
        self.config = config
        self.planner = planner
        self.device = torch.device(config.device)
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)
        self.mean = torch.zeros((config.horizon, config.action_dim), dtype=torch.float32, device=self.device)
        self.std = torch.full_like(self.mean, float(config.init_std))
        self.joint_low = torch.as_tensor(joint_low, dtype=torch.float32, device=self.device)
        self.joint_high = torch.as_tensor(joint_high, dtype=torch.float32, device=self.device)

    @property
    def num_elites(self) -> int:
        if self.config.num_elites is not None:
            return max(1, min(int(self.config.num_elites), self.config.num_samples))
        return max(1, min(self.config.num_samples, int(round(self.config.num_samples * self.config.elite_ratio))))

    def _sample_temporal_noise(self) -> torch.Tensor:
        noise = torch.randn(
            (self.config.num_samples, self.config.horizon, self.config.action_dim),
            generator=self.generator,
            device=self.device,
        )
        alpha = float(self.config.temporal_noise_alpha)
        if alpha <= 0:
            return noise
        for step_idx in range(1, self.config.horizon):
            noise[:, step_idx] = alpha * noise[:, step_idx - 1] + (1.0 - alpha) * noise[:, step_idx]
        return noise

    def _fallback(self, previous_q_ref: np.ndarray, start_time: float, reason: str) -> CEMMPCResult:
        previous = np.asarray(previous_q_ref, dtype=np.float32)
        return CEMMPCResult(
            q_ref=previous.copy(),
            delta_q_ref=np.zeros(self.config.action_dim, dtype=np.float32),
            best_cost=float("inf"),
            elite_mean_cost=float("inf"),
            planning_time=perf_counter() - start_time,
            failure=True,
            failure_reason=reason,
            best_sequence=np.zeros((self.config.horizon, self.config.action_dim), dtype=np.float32),
        )

    def plan(self, current_state: np.ndarray, previous_q_ref: np.ndarray) -> CEMMPCResult:
        del current_state
        start_time = perf_counter()
        mean = self.mean.clone()
        std = self.std.clone()
        best_sequence = None
        best_q_ref_sequence = None
        best_cost = torch.as_tensor(float("inf"), device=self.device)
        elite_mean_cost = torch.as_tensor(float("inf"), device=self.device)

        try:
            for _ in range(self.config.cem_iters):
                samples = mean.unsqueeze(0) + self._sample_temporal_noise() * std.unsqueeze(0)
                evaluation = self.planner.evaluate(samples)
                costs = evaluation["costs"].to(self.device)
                if costs.ndim != 1 or costs.shape[0] != self.config.num_samples:
                    return self._fallback(previous_q_ref, start_time, "invalid_cost_shape")
                valid = torch.isfinite(costs)
                if not bool(torch.any(valid)):
                    return self._fallback(previous_q_ref, start_time, "all_costs_invalid")
                safe_costs = torch.where(valid, costs, torch.full_like(costs, float("inf")))
                elite_indices = torch.topk(safe_costs, k=self.num_elites, largest=False).indices
                elites = samples[elite_indices]
                elite_costs = safe_costs[elite_indices]
                elite_mean_cost = elite_costs.mean()
                candidate_best_cost, best_local = torch.min(safe_costs, dim=0)
                if candidate_best_cost < best_cost:
                    best_cost = candidate_best_cost
                    best_sequence = samples[best_local].detach().clone()
                    if "q_ref_sequences" in evaluation:
                        best_q_ref_sequence = evaluation["q_ref_sequences"][best_local].detach().clone()
                new_mean = elites.mean(dim=0)
                new_std = elites.std(dim=0, unbiased=False).clamp_min(float(self.config.min_std))
                alpha = float(self.config.smoothing_alpha)
                mean = alpha * mean + (1.0 - alpha) * new_mean
                std = alpha * std + (1.0 - alpha) * new_std
        except RuntimeError as exc:
            return self._fallback(previous_q_ref, start_time, f"planner_runtime_error:{exc}")

        if best_sequence is None or best_q_ref_sequence is None:
            return self._fallback(previous_q_ref, start_time, "no_valid_sequence")
        if not torch.all(torch.isfinite(best_sequence)) or not torch.all(torch.isfinite(best_q_ref_sequence)):
            return self._fallback(previous_q_ref, start_time, "invalid_selected_action")

        self.mean = torch.cat([best_sequence[1:], best_sequence[-1:].clone()], dim=0)
        self.std = std.detach().clone()
        selected_q_ref = best_q_ref_sequence[0].detach().cpu().numpy().astype(np.float32)
        previous = np.asarray(previous_q_ref, dtype=np.float32)
        selected_delta = (selected_q_ref - previous).astype(np.float32)
        return CEMMPCResult(
            q_ref=selected_q_ref,
            delta_q_ref=selected_delta,
            best_cost=float(best_cost.detach().cpu()),
            elite_mean_cost=float(elite_mean_cost.detach().cpu()),
            planning_time=perf_counter() - start_time,
            failure=False,
            failure_reason="",
            best_sequence=best_sequence.detach().cpu().numpy().astype(np.float32),
        )
