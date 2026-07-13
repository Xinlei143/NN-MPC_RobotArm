from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from neural_dynamics.models import GRUDynamics, MLPDynamics, TransformerDynamics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_type: str,
    state_dim: int,
    action_dim: int,
    history_len: int = 1,
    hidden_size: int = 256,
    output_dim: int | None = None,
) -> nn.Module:
    if model_type == "mlp":
        return MLPDynamics(state_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size, output_dim=output_dim)
    if model_type == "gru":
        return GRUDynamics(state_dim=state_dim, action_dim=action_dim, hidden_size=hidden_size, output_dim=output_dim)
    if model_type == "transformer":
        return TransformerDynamics(
            state_dim=state_dim,
            action_dim=action_dim,
            max_history_len=max(256, history_len),
            output_dim=output_dim,
        )
    raise ValueError(f"model_type must be one of mlp/gru/transformer, got {model_type!r}")


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=True)


def load_yaml(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {"model_state_dict": model.state_dict(), "config": config}
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()
    if metadata is not None:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {path}")
    return checkpoint


def require_resume_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    metadata = checkpoint.get("metadata")
    missing = []
    if "optimizer_state_dict" not in checkpoint:
        missing.append("optimizer_state_dict")
    if "scaler_state_dict" not in checkpoint:
        missing.append("scaler_state_dict")
    if not isinstance(metadata, dict):
        missing.append("metadata")
    else:
        for key in ("epoch", "best_val"):
            if key not in metadata:
                missing.append(f"metadata.{key}")
    if missing:
        raise ValueError(f"Checkpoint is not a full resume checkpoint; missing: {', '.join(missing)}")
    return checkpoint
