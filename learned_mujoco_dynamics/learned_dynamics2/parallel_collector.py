from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from learned_dynamics2.mujoco_env import MuJoCoArmEnv

REQUIRED_DATASET_ARRAYS = ("states", "actions", "next_states")


def sample_smooth_action(
    rng: np.random.Generator,
    previous_action: np.ndarray,
    action_std: float | np.ndarray,
    n_joints: int,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=action_std, size=n_joints)
    action = 0.8 * previous_action + 0.2 * noise
    low = -np.ones(n_joints, dtype=np.float32) if action_low is None else np.asarray(action_low, dtype=np.float32)
    high = np.ones(n_joints, dtype=np.float32) if action_high is None else np.asarray(action_high, dtype=np.float32)
    if low.shape != (n_joints,) or high.shape != (n_joints,):
        raise ValueError(f"action_low/action_high must have shape ({n_joints},), got {low.shape} and {high.shape}")
    return np.clip(action, low, high).astype(np.float32)


def parse_action_std(value: float | str | np.ndarray, n_joints: int) -> float | np.ndarray:
    if isinstance(value, np.ndarray):
        if value.shape != (n_joints,):
            raise ValueError(f"action_std must be scalar or shape ({n_joints},), got {value.shape}")
        if np.any(value < 0):
            raise ValueError(f"action_std values must be non-negative, got {value}")
        return value.astype(np.float32, copy=False)
    if isinstance(value, str):
        parts = [float(item.strip()) for item in value.split(",") if item.strip()]
        if len(parts) == 1:
            value = parts[0]
        elif len(parts) == n_joints:
            arr = np.asarray(parts, dtype=np.float32)
            if np.any(arr < 0):
                raise ValueError(f"action_std values must be non-negative, got {value!r}")
            return arr
        else:
            raise ValueError(f"action_std must be scalar or {n_joints} comma-separated values, got {value!r}")
    scalar = float(value)
    if scalar < 0:
        raise ValueError(f"action_std must be non-negative, got {scalar}")
    return scalar


def collect_rollouts(
    model_xml: str,
    n_joints: int,
    num_episodes: int,
    episode_len: int,
    action_std: float | str | np.ndarray,
    seed: int,
    worker_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    if episode_len <= 0:
        raise ValueError(f"episode_len must be positive, got {episode_len}")
    action_std = parse_action_std(action_std, n_joints)

    rng = np.random.default_rng(seed)
    env = MuJoCoArmEnv(model_xml=model_xml, n_joints=n_joints, seed=seed)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    try:
        iterator = range(num_episodes)
        if worker_id == 0 and num_episodes > 1:
            iterator = tqdm(iterator, desc="collect", unit="episode")
        for _ in iterator:
            state = env.reset_random()
            action = np.zeros(n_joints, dtype=np.float32)
            for _step in range(episode_len):
                action = sample_smooth_action(
                    rng,
                    action,
                    action_std,
                    n_joints,
                    action_low=env.action_low,
                    action_high=env.action_high,
                )
                next_state = env.step(action)
                states.append(state.copy())
                actions.append(action.copy())
                next_states.append(next_state.copy())
                state = next_state
    finally:
        env.close()

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(next_states, dtype=np.float32),
    )


def collect_rollouts_with_episode_ids(
    model_xml: str,
    n_joints: int,
    num_episodes: int,
    episode_len: int,
    action_std: float,
    seed: int,
    episode_id_offset: int = 0,
    worker_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states, actions, next_states = collect_rollouts(
        model_xml=model_xml,
        n_joints=n_joints,
        num_episodes=num_episodes,
        episode_len=episode_len,
        action_std=action_std,
        seed=seed,
        worker_id=worker_id,
    )
    episode_ids = np.repeat(
        np.arange(episode_id_offset, episode_id_offset + num_episodes, dtype=np.int64),
        episode_len,
    )
    return states, actions, next_states, episode_ids


def collect_worker(
    worker_id: int,
    model_xml: str,
    n_joints: int,
    num_episodes: int,
    episode_len: int,
    action_std: float,
    seed: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    worker_seed = seed + 1009 * worker_id
    states, actions, next_states = collect_rollouts(
        model_xml=model_xml,
        n_joints=n_joints,
        num_episodes=num_episodes,
        episode_len=episode_len,
        action_std=action_std,
        seed=worker_seed,
        worker_id=worker_id,
    )
    return worker_id, states, actions, next_states


def collect_worker_with_episode_ids(
    worker_id: int,
    model_xml: str,
    n_joints: int,
    num_episodes: int,
    episode_len: int,
    action_std: float,
    seed: int,
    episode_id_offset: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    worker_seed = seed + 1009 * worker_id
    states, actions, next_states, episode_ids = collect_rollouts_with_episode_ids(
        model_xml=model_xml,
        n_joints=n_joints,
        num_episodes=num_episodes,
        episode_len=episode_len,
        action_std=action_std,
        seed=worker_seed,
        episode_id_offset=episode_id_offset,
        worker_id=worker_id,
    )
    return worker_id, states, actions, next_states, episode_ids


def split_episode_counts(num_episodes: int, num_envs: int) -> list[int]:
    if num_envs < 1:
        raise ValueError(f"num_envs must be at least 1, got {num_envs}")
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    base = num_episodes // num_envs
    remainder = num_episodes % num_envs
    return [base + (1 if idx < remainder else 0) for idx in range(num_envs)]


def collect_parallel(
    model_xml: str,
    n_joints: int,
    num_episodes: int,
    episode_len: int,
    action_std: float,
    seed: int,
    num_envs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = split_episode_counts(num_episodes, num_envs)
    jobs = [(idx, count) for idx, count in enumerate(counts) if count > 0]
    if not jobs:
        raise ValueError("No episodes were assigned to workers.")

    state_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    next_state_batches: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [
            executor.submit(
                collect_worker,
                worker_id,
                model_xml,
                n_joints,
                count,
                episode_len,
                action_std,
                seed,
            )
            for worker_id, count in jobs
        ]
        for future in as_completed(futures):
            try:
                worker_id, states, actions, next_states = future.result()
            except Exception as exc:  # pragma: no cover - exercised in real worker failures
                raise RuntimeError(f"Parallel collection worker failed: {exc}") from exc
            print(f"worker_id={worker_id} collected samples={len(states)}")
            state_batches.append(states)
            action_batches.append(actions)
            next_state_batches.append(next_states)

    return (
        np.concatenate(state_batches, axis=0),
        np.concatenate(action_batches, axis=0),
        np.concatenate(next_state_batches, axis=0),
    )


def collect_parallel_with_episode_ids(
    model_xml: str,
    n_joints: int,
    num_episodes: int,
    episode_len: int,
    action_std: float,
    seed: int,
    num_envs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = split_episode_counts(num_episodes, num_envs)
    offsets = np.cumsum([0, *counts[:-1]], dtype=np.int64)
    jobs = [(idx, count, int(offsets[idx])) for idx, count in enumerate(counts) if count > 0]
    if not jobs:
        raise ValueError("No episodes were assigned to workers.")

    state_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    next_state_batches: list[np.ndarray] = []
    episode_id_batches: list[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [
            executor.submit(
                collect_worker_with_episode_ids,
                worker_id,
                model_xml,
                n_joints,
                count,
                episode_len,
                action_std,
                seed,
                episode_id_offset,
            )
            for worker_id, count, episode_id_offset in jobs
        ]
        for future in as_completed(futures):
            try:
                worker_id, states, actions, next_states, episode_ids = future.result()
            except Exception as exc:  # pragma: no cover - exercised in real worker failures
                raise RuntimeError(f"Parallel collection worker failed: {exc}") from exc
            print(f"worker_id={worker_id} collected samples={len(states)}")
            state_batches.append(states)
            action_batches.append(actions)
            next_state_batches.append(next_states)
            episode_id_batches.append(episode_ids)

    return (
        np.concatenate(state_batches, axis=0),
        np.concatenate(action_batches, axis=0),
        np.concatenate(next_state_batches, axis=0),
        np.concatenate(episode_id_batches, axis=0),
    )


def save_dataset(
    save_path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    append: bool = False,
    episode_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if episode_ids is not None:
        episode_ids = np.asarray(episode_ids, dtype=np.int64)
        if episode_ids.ndim != 1:
            raise ValueError(f"episode_ids must be rank-1, got shape {episode_ids.shape}")
        if len(episode_ids) != len(states):
            raise ValueError(f"episode_ids length={len(episode_ids)} does not match samples={len(states)}")
    if append and save_path.exists():
        existing = validate_append_dataset(save_path, require_episode_ids=episode_ids is not None)
        states = np.concatenate([existing["states"], states], axis=0)
        actions = np.concatenate([existing["actions"], actions], axis=0)
        next_states = np.concatenate([existing["next_states"], next_states], axis=0)
        if episode_ids is not None:
            existing_episode_ids = existing["episode_ids"].astype(np.int64, copy=False)
            offset = int(existing_episode_ids.max()) + 1 if len(existing_episode_ids) else 0
            episode_ids = np.concatenate([existing_episode_ids, episode_ids + offset], axis=0)
        elif "episode_ids" in existing.files:
            raise ValueError("Cannot append samples without episode_ids to an existing episode-aware dataset")
    if episode_ids is None:
        np.savez_compressed(save_path, states=states, actions=actions, next_states=next_states)
        return states, actions, next_states
    np.savez_compressed(save_path, states=states, actions=actions, next_states=next_states, episode_ids=episode_ids)
    return states, actions, next_states, episode_ids


def validate_append_dataset(save_path: Path, require_episode_ids: bool = False) -> np.lib.npyio.NpzFile | None:
    save_path = Path(save_path)
    if not save_path.exists():
        return None
    existing = np.load(save_path)
    required = set(REQUIRED_DATASET_ARRAYS)
    if require_episode_ids:
        required.add("episode_ids")
    missing = required.difference(existing.files)
    if missing:
        raise KeyError(f"Existing dataset {save_path} is missing arrays: {sorted(missing)}")
    for name in required:
        # Force array loading here so corrupted partial .npz files fail before collection starts.
        existing[name]
    return existing
