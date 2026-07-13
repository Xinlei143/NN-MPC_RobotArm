from __future__ import annotations

from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = ("states", "actions", "next_states")


def merge_npz_datasets(input_paths: list[Path], output_path: Path) -> dict[str, tuple[int, ...]]:
    if len(input_paths) < 2:
        raise ValueError("At least two input datasets are required to merge")

    loaded = []
    for path in input_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file does not exist: {path}")
        data = np.load(path)
        missing = set(REQUIRED_ARRAYS).difference(data.files)
        if missing:
            raise KeyError(f"Dataset file {path} is missing arrays: {sorted(missing)}")
        loaded.append((path, data))

    reference_shapes = {name: loaded[0][1][name].shape[1:] for name in REQUIRED_ARRAYS}
    for path, data in loaded:
        lengths = {data[name].shape[0] for name in REQUIRED_ARRAYS}
        if len(lengths) != 1:
            raise ValueError(f"Dataset file {path} has inconsistent sample counts")
        for name in REQUIRED_ARRAYS:
            if data[name].shape[1:] != reference_shapes[name]:
                raise ValueError(
                    f"Dataset file {path} has incompatible {name} shape {data[name].shape}; "
                    f"expected trailing shape {reference_shapes[name]}"
                )

    merged = {name: np.concatenate([data[name] for _, data in loaded], axis=0) for name in REQUIRED_ARRAYS}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **merged)
    return {name: merged[name].shape for name in REQUIRED_ARRAYS}
