from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def verify_real_artifact_identity(checkpoint_path: str | Path, normalizer_path: str | Path,
                                  expected_plant_identity: dict[str, Any]) -> None:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    _require_plant_match(expected_plant_identity, checkpoint.get("config", {}).get("plant_identity"),
                         "dynamics checkpoint")
    normalizer = torch.load(Path(normalizer_path), map_location="cpu", weights_only=False)
    if isinstance(normalizer, dict):
        metadata = normalizer.get("metadata", {})
    else:
        metadata = getattr(normalizer, "metadata", {})
    _require_plant_match(expected_plant_identity, metadata.get("plant_identity"), "normalizer")


def _require_plant_match(expected: dict[str, Any], actual: Any, label: str) -> None:
    """Subset match: every plant field the runtime declares must be present
    and identical in the artifact.  Artifacts may carry extra provenance
    fields (e.g. ``table_safety`` captured at collection time); those are not
    runtime gates, so they are allowed to differ from the current files."""
    if not isinstance(actual, dict):
        raise ValueError(f"{label} plant identity is missing or malformed")
    for key, value in expected.items():
        if key not in actual:
            raise ValueError(f"{label} plant identity does not match connected SO101 (missing {key!r})")
        if actual[key] != value:
            raise ValueError(f"{label} plant identity does not match connected SO101 ({key!r} differs)")
