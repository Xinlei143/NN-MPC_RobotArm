from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class StandardNormalizer:
    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.state_mean: torch.Tensor | None = None
        self.state_std: torch.Tensor | None = None
        self.action_mean: torch.Tensor | None = None
        self.action_std: torch.Tensor | None = None
        self.delta_mean: torch.Tensor | None = None
        self.delta_std: torch.Tensor | None = None

    def fit(self, states: torch.Tensor, actions: torch.Tensor, deltas: torch.Tensor) -> None:
        self.state_mean = states.mean(dim=0)
        self.state_std = states.std(dim=0, unbiased=False).clamp_min(self.eps)
        self.action_mean = actions.mean(dim=0)
        self.action_std = actions.std(dim=0, unbiased=False).clamp_min(self.eps)
        self.delta_mean = deltas.mean(dim=0)
        self.delta_std = deltas.std(dim=0, unbiased=False).clamp_min(self.eps)

    def _require(self, name: str) -> torch.Tensor:
        value = getattr(self, name)
        if value is None:
            raise RuntimeError("StandardNormalizer has not been fitted or loaded.")
        return value

    @staticmethod
    def _to_device(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return value.to(device=reference.device, dtype=reference.dtype)

    def normalize_state(self, states: torch.Tensor) -> torch.Tensor:
        return (states - self._to_device(self._require("state_mean"), states)) / self._to_device(
            self._require("state_std"), states
        )

    def normalize_action(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self._to_device(self._require("action_mean"), actions)) / self._to_device(
            self._require("action_std"), actions
        )

    def normalize_delta(self, deltas: torch.Tensor) -> torch.Tensor:
        return (deltas - self._to_device(self._require("delta_mean"), deltas)) / self._to_device(
            self._require("delta_std"), deltas
        )

    def denormalize_delta(self, deltas: torch.Tensor) -> torch.Tensor:
        return deltas * self._to_device(self._require("delta_std"), deltas) + self._to_device(
            self._require("delta_mean"), deltas
        )

    def normalize_single_input(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.normalize_state(states), self.normalize_action(actions)], dim=-1)

    def normalize_sequence_input(self, sequence: torch.Tensor, state_dim: int) -> torch.Tensor:
        states = sequence[..., :state_dim]
        actions = sequence[..., state_dim:]
        return torch.cat([self.normalize_state(states), self.normalize_action(actions)], dim=-1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "eps": self.eps,
            "state_mean": self._require("state_mean").cpu(),
            "state_std": self._require("state_std").cpu(),
            "action_mean": self._require("action_mean").cpu(),
            "action_std": self._require("action_std").cpu(),
            "delta_mean": self._require("delta_mean").cpu(),
            "delta_std": self._require("delta_std").cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.eps = float(state.get("eps", self.eps))
        for key in ("state_mean", "state_std", "action_mean", "action_std", "delta_mean", "delta_std"):
            if key not in state:
                raise KeyError(f"Normalizer checkpoint is missing key: {key}")
            setattr(self, key, state[key].detach().clone().float())

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, map_location: str | torch.device = "cpu") -> "StandardNormalizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normalizer file does not exist: {path}")
        normalizer = cls()
        normalizer.load_state_dict(torch.load(path, map_location=map_location))
        return normalizer
