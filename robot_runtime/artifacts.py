from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def verify_real_artifact_identity(checkpoint_path: str | Path, normalizer_path: str | Path,
                                  expected_plant_identity: dict[str, Any]) -> None:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    checkpoint_identity = checkpoint.get("config", {}).get("plant_identity")
    if checkpoint_identity != expected_plant_identity:
        raise ValueError("dynamics checkpoint plant identity does not match connected SO101")
    normalizer = torch.load(Path(normalizer_path), map_location="cpu", weights_only=False)
    if isinstance(normalizer, dict):
        metadata = normalizer.get("metadata", {})
    else:
        metadata = getattr(normalizer, "metadata", {})
    if metadata.get("plant_identity") != expected_plant_identity:
        raise ValueError("normalizer plant identity does not match connected SO101")
