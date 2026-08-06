"""Strict, audit-preserving append store for real robot dataset sessions."""
from __future__ import annotations

import json
from pathlib import Path
import uuid
from typing import Any

import numpy as np

from robot_runtime.config import file_sha256


_INVARIANT_KEYS = (
    "schema_version", "action_semantics", "robot_identity",
    "collection_mode", "collection_plan", "workspace_bounds",
)


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"strict append requires session manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_arrays(arrays: dict[str, np.ndarray]) -> int:
    if not arrays:
        raise ValueError("cannot append an empty real-data session")
    if "states" not in arrays or "session_ids" not in arrays or "episode_ids" not in arrays:
        raise KeyError("strict append requires states, session_ids, and episode_ids")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("every real dataset array must have the same sample count")
    samples = lengths.pop()
    if samples <= 0:
        raise ValueError("cannot append an empty real-data session")
    return int(samples)


def _plant_identity_without_table_safety(manifest: dict[str, Any]) -> dict[str, Any]:
    """Keep real-plant identity strict while allowing audited safety profiles per run."""
    identity = dict(manifest.get("plant_identity", {}))
    identity.pop("table_safety", None)
    return identity


def _table_profiles(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    profiles = dict(manifest.get("table_safety_profiles", {}))
    legacy = manifest.get("table_safety")
    if isinstance(legacy, dict) and legacy.get("sha256"):
        profiles.setdefault(str(legacy["sha256"]), legacy)
    return profiles


def append_completed_session(canonical_path: str | Path, session_path: str | Path) -> tuple[Path, Path]:
    """Append a completed shard or create a canonical dataset without overwrites.

    Both schemas, identity payloads, array names/shapes/dtypes, session IDs and
    episode IDs must agree.  This makes a failed or accidental rerun visible
    instead of silently changing a training dataset.
    """
    canonical, session = Path(canonical_path), Path(session_path)
    if canonical.resolve() == session.resolve():
        raise ValueError("canonical dataset and per-session shard must be different files")
    session_manifest = _read_manifest(session)
    if session_manifest.get("status") != "completed":
        raise ValueError("only a completed session may be appended to the canonical dataset")
    with np.load(session, allow_pickle=False) as new_data:
        new_arrays = {name: new_data[name].copy() for name in new_data.files}
    samples = _validate_arrays(new_arrays)
    session_ids = set(np.asarray(new_arrays["session_ids"]).astype(str).tolist())
    if len(session_ids) != 1 or "" in session_ids:
        raise ValueError("a shard must contain exactly one non-empty session_id")

    canonical_manifest_path = canonical.with_suffix(".manifest.json")
    if canonical.exists():
        if not canonical_manifest_path.is_file():
            raise FileNotFoundError("refusing to append to a dataset without its strict manifest")
        canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
        for key in _INVARIANT_KEYS:
            if canonical_manifest.get(key) != session_manifest.get(key):
                raise ValueError(f"append invariant differs for {key!r}")
        if _plant_identity_without_table_safety(canonical_manifest) != _plant_identity_without_table_safety(session_manifest):
            raise ValueError("append invariant differs for real plant identity outside table_safety")
        with np.load(canonical, allow_pickle=False) as old_data:
            old_arrays = {name: old_data[name].copy() for name in old_data.files}
        _validate_arrays(old_arrays)
        if set(old_arrays) != set(new_arrays):
            raise ValueError("canonical and session archive fields differ")
        old_session_ids = set(np.asarray(old_arrays["session_ids"]).astype(str).tolist())
        if old_session_ids.intersection(session_ids):
            raise ValueError(f"session already exists in canonical dataset: {sorted(session_ids)}")
        old_episode_ids = set(np.asarray(old_arrays["episode_ids"], dtype=np.int64).tolist())
        new_episode_ids = set(np.asarray(new_arrays["episode_ids"], dtype=np.int64).tolist())
        if old_episode_ids.intersection(new_episode_ids):
            raise ValueError("episode_ids must be globally unique across appended sessions")
        merged: dict[str, np.ndarray] = {}
        for name in sorted(old_arrays):
            if old_arrays[name].shape[1:] != new_arrays[name].shape[1:]:
                raise ValueError(f"array schema differs for {name!r}")
            old_dtype, new_dtype = old_arrays[name].dtype, new_arrays[name].dtype
            if old_dtype == new_dtype:
                merged[name] = np.concatenate((old_arrays[name], new_arrays[name]), axis=0)
                continue
            # NumPy archives use fixed-width Unicode/bytes dtypes.  Per-run
            # audit labels legitimately grow (e.g. ``gripper`` versus
            # ``moving_jaw_so101_v1``); widen only text columns and retain
            # exact dtype matching for all numerical/control fields.
            if old_dtype.kind in "US" and new_dtype.kind in "US":
                common_dtype = np.promote_types(old_dtype, new_dtype)
                merged[name] = np.concatenate((
                    old_arrays[name].astype(common_dtype, copy=False),
                    new_arrays[name].astype(common_dtype, copy=False),
                ), axis=0)
                continue
            raise ValueError(f"array schema differs for {name!r}")
        runs = list(canonical_manifest.get("collection_runs", []))
        profiles = _table_profiles(canonical_manifest)
    else:
        merged = new_arrays
        canonical_manifest = {key: session_manifest.get(key) for key in _INVARIANT_KEYS}
        canonical_manifest["plant_identity"] = session_manifest.get("plant_identity")
        canonical_manifest["table_safety"] = session_manifest.get("table_safety")
        profiles = _table_profiles(session_manifest)
        runs = []

    _atomic_npz(canonical, merged)
    runs.append({
        "session_id": next(iter(session_ids)), "sample_count": samples,
        "session_dataset": str(session), "session_sha256": file_sha256(session),
        "split": session_manifest.get("split"), "session_index": session_manifest.get("session_index"),
        "table_safety_sha256": session_manifest.get("table_safety", {}).get("sha256"),
    })
    session_table = session_manifest.get("table_safety")
    if isinstance(session_table, dict) and session_table.get("sha256"):
        profiles[str(session_table["sha256"])] = session_table
    canonical_manifest.update({
        "sample_count": int(len(merged["states"])),
        "collection_runs": runs,
        "table_safety_profiles": profiles,
        "dataset": {"path": str(canonical), "sha256": file_sha256(canonical)},
    })
    _atomic_json(canonical_manifest_path, canonical_manifest)
    return canonical, canonical_manifest_path
