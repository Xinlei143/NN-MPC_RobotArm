from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from learned_dynamics.mujoco_env import MuJoCoArmEnv

REQUIRED_DATASET_ARRAYS = ("states", "actions", "next_states")
MOTION_MODE_NAMES = ("hold", "step", "smooth_random", "sine", "delta_ref_random", "mpc_correlated_random")
TERMINATION_REASON_CODES = {
    "ok": 0,
    "nan": 1,
    "near_joint_limit": 2,
    "high_velocity": 3,
    "high_torque": 4,
    "state_spike": 5,
}
LEGACY_APPEND_FILL_EXTRA_ARRAYS = {"action_std_normalized", "settle_steps"}


def sample_smooth_action(
    rng: np.random.Generator,
    previous_action: np.ndarray,
    action_std: float | np.ndarray,
    n_joints: int,
    action_low: np.ndarray | None = None,
    action_high: np.ndarray | None = None,
) -> np.ndarray:
    if action_low is None or action_high is None:
        raise ValueError("action_low/action_high are required for normalized action_std sampling")
    low = np.asarray(action_low, dtype=np.float64)
    high = np.asarray(action_high, dtype=np.float64)
    if low.shape != (n_joints,) or high.shape != (n_joints,):
        raise ValueError(f"action_low/action_high must have shape ({n_joints},), got {low.shape} and {high.shape}")
    half_range = (high - low) / 2.0
    if np.any(half_range <= 0.0):
        raise ValueError(f"action ranges must have positive width, got low={low} high={high}")

    center = (high + low) / 2.0
    previous = np.asarray(previous_action, dtype=np.float64)
    if previous.shape != (n_joints,):
        raise ValueError(f"previous_action must have shape ({n_joints},), got {previous.shape}")

    previous_norm = (np.clip(previous, low, high) - center) / half_range
    noise_norm = rng.normal(loc=0.0, scale=action_std, size=n_joints)
    action_norm = np.clip(0.8 * previous_norm + 0.2 * noise_norm, -1.0, 1.0)
    action = center + action_norm * half_range
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


def _action_to_normalized(action: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    low = np.asarray(action_low, dtype=np.float64)
    high = np.asarray(action_high, dtype=np.float64)
    center = (high + low) / 2.0
    half_range = (high - low) / 2.0
    return (np.asarray(action, dtype=np.float64) - center) / half_range


def _normalized_to_action(action_norm: np.ndarray, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    low = np.asarray(action_low, dtype=np.float64)
    high = np.asarray(action_high, dtype=np.float64)
    center = (high + low) / 2.0
    half_range = (high - low) / 2.0
    return np.clip(center + np.asarray(action_norm, dtype=np.float64) * half_range, low, high).astype(np.float32)


def _action_std_array(action_std: float | str | np.ndarray, n_joints: int) -> np.ndarray:
    parsed = parse_action_std(action_std, n_joints)
    if np.isscalar(parsed):
        return np.full(n_joints, float(parsed), dtype=np.float32)
    return np.asarray(parsed, dtype=np.float32)


def _safe_workspace_bounds(n_joints: int) -> tuple[np.ndarray, np.ndarray]:
    low = np.full(n_joints, -0.20, dtype=np.float64)
    high = np.full(n_joints, 0.20, dtype=np.float64)
    if n_joints >= 2:
        low[1] = -0.45
        high[1] = -0.10
    if n_joints >= 3:
        low[2] = -0.25
        high[2] = 0.15
    return low, high


def _target_workspace_bounds(action_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    span = np.clip(np.asarray(action_std, dtype=np.float64) * 0.9, 0.25, 0.75)
    low = -span
    high = span
    if len(high) >= 2:
        high[1] = min(high[1], 0.35)
        low[1] = max(low[1], -0.70)
    return low, high


def reset_safe_workspace(env: MuJoCoArmEnv, rng: np.random.Generator, n_joints: int) -> np.ndarray:
    low_norm, high_norm = _safe_workspace_bounds(n_joints)
    q_norm = rng.uniform(low_norm, high_norm)
    qpos = _normalized_to_action(q_norm, env.joint_low, env.joint_high)

    import mujoco

    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[:n_joints] = qpos
    env.data.qvel[:n_joints] = rng.uniform(-0.01, 0.01, size=n_joints)
    env.data.ctrl[:n_joints] = qpos
    env.data.qfrc_applied[:n_joints] = 0.0
    mujoco.mj_forward(env.model, env.data)
    env.validate_joint_positions("reset_safe_workspace")
    return env.get_state()


def _sample_target_norm(rng: np.random.Generator, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return rng.uniform(low, high)


def _filter_normalized_sequence(sequence_norm: np.ndarray, start_norm: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    filtered = np.zeros_like(sequence_norm)
    previous = start_norm.astype(np.float64, copy=True)
    for idx, target in enumerate(sequence_norm):
        previous = (1.0 - alpha) * previous + alpha * target
        filtered[idx] = previous
    return filtered


def generate_q_ref_sequence(
    rng: np.random.Generator,
    start_q_ref: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    episode_len: int,
    action_std: np.ndarray,
    mode_id: int,
) -> np.ndarray:
    n_joints = len(start_q_ref)
    low_norm, high_norm = _target_workspace_bounds(action_std)
    start_norm = np.clip(_action_to_normalized(start_q_ref, action_low, action_high), low_norm, high_norm)
    mode = MOTION_MODE_NAMES[mode_id % len(MOTION_MODE_NAMES)]

    if mode == "hold":
        target = _sample_target_norm(rng, low_norm, high_norm)
        sequence_norm = np.repeat(target[None, :], episode_len, axis=0)
    elif mode == "step":
        segment_len = max(10, episode_len // 4)
        sequence_norm = np.zeros((episode_len, n_joints), dtype=np.float64)
        target = start_norm.copy()
        for start in range(0, episode_len, segment_len):
            target = _sample_target_norm(rng, low_norm, high_norm)
            sequence_norm[start : start + segment_len] = target
    elif mode == "smooth_random":
        segment_len = max(10, episode_len // 5)
        knots = [start_norm]
        for _ in range(max(2, episode_len // segment_len) + 1):
            knots.append(_sample_target_norm(rng, low_norm, high_norm))
        sequence_norm = np.zeros((episode_len, n_joints), dtype=np.float64)
        for step in range(episode_len):
            knot_idx = min(step // segment_len, len(knots) - 2)
            alpha = (step % segment_len) / float(segment_len)
            sequence_norm[step] = (1.0 - alpha) * knots[knot_idx] + alpha * knots[knot_idx + 1]
    elif mode == "sine":
        base = _sample_target_norm(rng, low_norm * 0.5, high_norm * 0.5)
        amplitude = np.minimum(np.abs(high_norm - low_norm) * 0.25, 0.35)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n_joints)
        cycles = rng.uniform(0.5, 1.5, size=n_joints)
        steps = np.arange(episode_len, dtype=np.float64)[:, None]
        sequence_norm = base + amplitude * np.sin(2.0 * np.pi * steps * cycles / max(episode_len, 1) + phase)
        sequence_norm = np.clip(sequence_norm, low_norm, high_norm)
    elif mode == "delta_ref_random":
        delta_scale = np.maximum(action_std * 0.04, 0.01)
        deltas = rng.normal(loc=0.0, scale=delta_scale, size=(episode_len, n_joints))
        sequence_norm = np.cumsum(deltas, axis=0) + start_norm
        sequence_norm = np.clip(sequence_norm, low_norm, high_norm)
    elif mode == "mpc_correlated_random":
        delta_scale = np.maximum(action_std * 0.035, 0.008)
        noise = rng.normal(loc=0.0, scale=delta_scale, size=(episode_len, n_joints))
        correlated = np.zeros_like(noise)
        previous_delta = np.zeros(n_joints, dtype=np.float64)
        for step in range(episode_len):
            previous_delta = 0.85 * previous_delta + 0.15 * noise[step]
            correlated[step] = previous_delta
        sequence_norm = np.cumsum(correlated, axis=0) + start_norm
        sequence_norm = np.clip(sequence_norm, low_norm, high_norm)
    else:
        raise ValueError(f"Unknown motion mode: {mode!r}")

    sequence_norm = _filter_normalized_sequence(sequence_norm, start_norm)
    return _normalized_to_action(sequence_norm, action_low, action_high)


def _safety_code(
    env: MuJoCoArmEnv,
    previous_state: np.ndarray,
    next_state: np.ndarray,
    torque: dict[str, np.ndarray],
    n_joints: int,
    near_limit_norm: float = 0.92,
    max_qvel: float = 40.0,
    max_tau: float = 20000.0,
    max_delta_dq: float = 30.0,
) -> int:
    arrays = [previous_state, next_state, torque["actuator_tau"], torque["gravity_tau"], torque["total_tau"]]
    if any(not np.all(np.isfinite(value)) for value in arrays):
        return TERMINATION_REASON_CODES["nan"]
    q_norm = _action_to_normalized(next_state[:n_joints], env.joint_low, env.joint_high)
    if np.any(np.abs(q_norm) >= near_limit_norm):
        return TERMINATION_REASON_CODES["near_joint_limit"]
    if np.any(np.abs(next_state[n_joints:]) > max_qvel):
        return TERMINATION_REASON_CODES["high_velocity"]
    if np.any(np.abs(torque["total_tau"]) > max_tau):
        return TERMINATION_REASON_CODES["high_torque"]
    if np.any(np.abs(next_state[n_joints:] - previous_state[n_joints:]) > max_delta_dq):
        return TERMINATION_REASON_CODES["state_spike"]
    return TERMINATION_REASON_CODES["ok"]


def collect_rollouts_detailed(
    model_xml: str,
    n_joints: int,
    num_episodes: int,
    episode_len: int,
    action_std: float | str | np.ndarray,
    seed: int,
    worker_id: int = 0,
    episode_id_offset: int = 0,
    settle_steps: int = 50,
) -> dict[str, np.ndarray]:
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    if episode_len <= 0:
        raise ValueError(f"episode_len must be positive, got {episode_len}")
    if settle_steps < 0:
        raise ValueError(f"settle_steps must be non-negative, got {settle_steps}")

    action_std_array = _action_std_array(action_std, n_joints)
    rng = np.random.default_rng(seed)
    env = MuJoCoArmEnv(model_xml=model_xml, n_joints=n_joints, seed=seed)

    records: dict[str, list[np.ndarray | int]] = {
        "states": [],
        "actions": [],
        "next_states": [],
        "q_ref": [],
        "delta_q_ref": [],
        "tau_actuator": [],
        "tau_gravity": [],
        "tau_total": [],
        "action_std_normalized": [],
        "settle_steps": [],
        "episode_ids": [],
        "motion_mode_ids": [],
        "termination_reasons": [],
    }
    try:
        iterator = range(num_episodes)
        if worker_id == 0 and num_episodes > 1:
            iterator = tqdm(iterator, desc="collect", unit="episode")
        for episode_idx in iterator:
            state = reset_safe_workspace(env, rng, n_joints)
            q_ref = np.asarray(state[:n_joints], dtype=np.float32).copy()
            for _ in range(settle_steps):
                state = env.step(q_ref)

            mode_id = int((episode_id_offset + episode_idx) % len(MOTION_MODE_NAMES))
            q_ref_sequence = generate_q_ref_sequence(
                rng,
                q_ref,
                env.action_low,
                env.action_high,
                episode_len,
                action_std_array,
                mode_id,
            )
            previous_q_ref = q_ref.copy()
            for step_idx in range(episode_len):
                q_ref = q_ref_sequence[step_idx].astype(np.float32, copy=False)
                torque = env.compute_torque_components(q_ref)
                next_state = env.step(q_ref)
                termination_code = _safety_code(env, state, next_state, torque, n_joints)

                records["states"].append(state.copy())
                records["actions"].append(q_ref.copy())
                records["next_states"].append(next_state.copy())
                records["q_ref"].append(q_ref.copy())
                records["delta_q_ref"].append((q_ref - previous_q_ref).astype(np.float32))
                records["tau_actuator"].append(torque["actuator_tau"].astype(np.float32))
                records["tau_gravity"].append(torque["gravity_tau"].astype(np.float32))
                records["tau_total"].append(torque["total_tau"].astype(np.float32))
                records["action_std_normalized"].append(action_std_array.astype(np.float32))
                records["settle_steps"].append(settle_steps)
                records["episode_ids"].append(episode_id_offset + episode_idx)
                records["motion_mode_ids"].append(mode_id)
                records["termination_reasons"].append(termination_code)

                previous_q_ref = q_ref.copy()
                state = next_state
                if termination_code != TERMINATION_REASON_CODES["ok"]:
                    break
    finally:
        env.close()

    arrays: dict[str, np.ndarray] = {}
    for key, values in records.items():
        if key in {"episode_ids", "motion_mode_ids", "termination_reasons", "settle_steps"}:
            arrays[key] = np.asarray(values, dtype=np.int64)
        else:
            arrays[key] = np.asarray(values, dtype=np.float32)
    return arrays


def collect_rollouts(
    model_xml: str,
    n_joints: int,
    num_episodes: int,
    episode_len: int,
    action_std: float | str | np.ndarray,
    seed: int,
    worker_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = collect_rollouts_detailed(
        model_xml=model_xml,
        n_joints=n_joints,
        num_episodes=num_episodes,
        episode_len=episode_len,
        action_std=action_std,
        seed=seed,
        worker_id=worker_id,
    )
    return data["states"], data["actions"], data["next_states"]


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
    data = collect_rollouts_detailed(
        model_xml=model_xml,
        n_joints=n_joints,
        num_episodes=num_episodes,
        episode_len=episode_len,
        action_std=action_std,
        seed=seed,
        episode_id_offset=episode_id_offset,
        worker_id=worker_id,
    )
    return data["states"], data["actions"], data["next_states"], data["episode_ids"]


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


def collect_worker_detailed(
    worker_id: int,
    model_xml: str,
    n_joints: int,
    num_episodes: int,
    episode_len: int,
    action_std: float,
    seed: int,
    episode_id_offset: int,
    settle_steps: int,
) -> tuple[int, dict[str, np.ndarray]]:
    worker_seed = seed + 1009 * worker_id
    data = collect_rollouts_detailed(
        model_xml=model_xml,
        n_joints=n_joints,
        num_episodes=num_episodes,
        episode_len=episode_len,
        action_std=action_std,
        seed=worker_seed,
        worker_id=worker_id,
        episode_id_offset=episode_id_offset,
        settle_steps=settle_steps,
    )
    return worker_id, data


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


def collect_parallel_detailed(
    model_xml: str,
    n_joints: int,
    num_episodes: int,
    episode_len: int,
    action_std: float,
    seed: int,
    num_envs: int,
    settle_steps: int = 50,
) -> dict[str, np.ndarray]:
    counts = split_episode_counts(num_episodes, num_envs)
    offsets = np.cumsum([0, *counts[:-1]], dtype=np.int64)
    jobs = [(idx, count, int(offsets[idx])) for idx, count in enumerate(counts) if count > 0]
    if not jobs:
        raise ValueError("No episodes were assigned to workers.")

    batches: list[dict[str, np.ndarray]] = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [
            executor.submit(
                collect_worker_detailed,
                worker_id,
                model_xml,
                n_joints,
                count,
                episode_len,
                action_std,
                seed,
                episode_id_offset,
                settle_steps,
            )
            for worker_id, count, episode_id_offset in jobs
        ]
        for future in as_completed(futures):
            try:
                worker_id, data = future.result()
            except Exception as exc:  # pragma: no cover - exercised in real worker failures
                raise RuntimeError(f"Parallel collection worker failed: {exc}") from exc
            print(f"worker_id={worker_id} collected samples={len(data['states'])}")
            batches.append(data)

    keys = batches[0].keys()
    return {key: np.concatenate([batch[key] for batch in batches], axis=0) for key in keys}


def _legacy_extra_array_fill(key: str, existing_len: int, new_value: np.ndarray) -> np.ndarray:
    if key not in LEGACY_APPEND_FILL_EXTRA_ARRAYS:
        raise ValueError(f"Cannot append extra array {key!r}; existing dataset does not contain it")
    fill_shape = (existing_len, *new_value.shape[1:])
    return np.full(fill_shape, -1, dtype=new_value.dtype)


def save_dataset(
    save_path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    append: bool = False,
    episode_ids: np.ndarray | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    extra_arrays = {} if extra_arrays is None else {key: np.asarray(value) for key, value in extra_arrays.items()}
    for key, value in extra_arrays.items():
        if len(value) != len(states):
            raise ValueError(f"extra array {key!r} length={len(value)} does not match samples={len(states)}")
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
        for key, value in list(extra_arrays.items()):
            if key not in existing.files:
                existing_value = _legacy_extra_array_fill(key, len(existing["states"]), value)
            else:
                existing_value = existing[key]
            extra_arrays[key] = np.concatenate([existing_value, value], axis=0)
    arrays = {"states": states, "actions": actions, "next_states": next_states, **extra_arrays}
    if episode_ids is None:
        np.savez_compressed(save_path, **arrays)
        return states, actions, next_states
    arrays["episode_ids"] = episode_ids
    np.savez_compressed(save_path, **arrays)
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
