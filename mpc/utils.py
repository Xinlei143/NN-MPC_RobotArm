from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch


def build_history_tensor(states: list[np.ndarray], actuator_q_refs: list[np.ndarray], history_len: int, device: torch.device) -> torch.Tensor:
    if not states or not actuator_q_refs:
        raise ValueError("states and actuator_q_refs must not be empty")
    if len(states) != len(actuator_q_refs):
        raise ValueError("states and actuator_q_refs must have the same length")
    start = max(0, len(states) - history_len)
    entries = [np.concatenate([state, action]).astype(np.float32) for state, action in zip(states[start:], actuator_q_refs[start:])]
    while len(entries) < history_len:
        entries.insert(0, entries[0].copy())
    return torch.as_tensor(np.stack(entries, axis=0), dtype=torch.float32, device=device)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_or_nan(value: float | np.ndarray) -> float:
    scalar = float(np.asarray(value).reshape(-1)[0])
    return scalar if np.isfinite(scalar) else float("nan")
