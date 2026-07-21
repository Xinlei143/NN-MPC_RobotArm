from __future__ import annotations

import numpy as np


def generate_joint_reference(
    mode: str,
    initial_q: np.ndarray,
    joint_low: np.ndarray,
    joint_high: np.ndarray,
    total_steps: int,
    control_dt: float,
    seed: int = 0,
    amplitude: float = 0.15,
) -> np.ndarray:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    rng = np.random.default_rng(seed)
    initial_q = np.asarray(initial_q, dtype=np.float32)
    joint_low = np.asarray(joint_low, dtype=np.float32)
    joint_high = np.asarray(joint_high, dtype=np.float32)
    n_joints = initial_q.shape[0]
    time = np.arange(total_steps, dtype=np.float32)[:, None] * float(control_dt)
    q_des = np.repeat(initial_q[None, :], total_steps, axis=0)

    if mode == "hold":
        pass
    elif mode == "step":
        target = np.clip(initial_q + rng.uniform(-amplitude, amplitude, size=n_joints), joint_low, joint_high)
        switch = max(1, total_steps // 3)
        q_des[switch:] = target
    elif mode == "joint_sine":
        q_des[:, 0] = initial_q[0] + amplitude * np.sin(2.0 * np.pi * 0.25 * time[:, 0])
    elif mode == "multi_joint_sine":
        freq = rng.uniform(0.12, 0.35, size=n_joints)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n_joints)
        scale = amplitude * rng.uniform(0.5, 1.0, size=n_joints)
        q_des = initial_q[None, :] + scale[None, :] * np.sin(2.0 * np.pi * time * freq[None, :] + phase[None, :])
    elif mode == "waypoint":
        waypoint_count = 4
        knots = np.vstack([
            initial_q,
            rng.uniform(initial_q - amplitude, initial_q + amplitude, size=(waypoint_count - 1, n_joints)),
            initial_q,
        ])
        q_des = np.empty_like(q_des)
        segment_edges = np.linspace(0, total_steps, len(knots), dtype=int)
        for index in range(len(knots) - 1):
            start, end = segment_edges[index], segment_edges[index + 1]
            tau = np.linspace(0.0, 1.0, max(1, end - start), endpoint=False, dtype=np.float32)[:, None]
            smooth = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
            q_des[start:end] = knots[index] + smooth * (knots[index + 1] - knots[index])
    elif mode == "chirp":
        duration = max(float(total_steps - 1) * float(control_dt), float(control_dt))
        f0, f1 = 0.10, 0.80
        phase = rng.uniform(0.0, 2.0 * np.pi, size=n_joints)
        scale = amplitude * rng.uniform(0.4, 0.8, size=n_joints)
        t = time[:, 0]
        chirp_phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * np.square(t) / duration)
        q_des = initial_q[None, :] + scale[None, :] * np.sin(chirp_phase[:, None] + phase[None, :])
    else:
        raise ValueError(f"Unknown reference mode {mode!r}; expected hold/step/joint_sine/multi_joint_sine/waypoint/chirp")

    return np.clip(q_des, joint_low, joint_high).astype(np.float32)


def finite_difference_dq(q_des: np.ndarray, control_dt: float) -> np.ndarray:
    if len(q_des) <= 1:
        return np.zeros_like(q_des, dtype=np.float32)
    return np.gradient(q_des, float(control_dt), axis=0).astype(np.float32)
