"""State-action history helpers with the learned-dynamics training semantics.

Each token represents ``[x_t, u_t]``: the command issued from ``x_t`` to
produce ``x_{t+1}``.  The newest state has no known command until the current
control tick chooses one, so its stored action is only a placeholder.
"""
from __future__ import annotations

import numpy as np


def history_tokens(
    states: list[np.ndarray] | np.ndarray,
    commands: list[np.ndarray] | np.ndarray,
    history_len: int,
) -> np.ndarray:
    """Return padded concatenated ``[state, command]`` tokens."""
    if len(states) != len(commands) or len(states) == 0:
        raise ValueError("states and commands must be non-empty and have equal length")
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    entries = [np.concatenate([state, command]).astype(np.float32) for state, command in zip(states, commands)]
    entries = entries[-history_len:]
    while len(entries) < history_len:
        entries.insert(0, entries[0].copy())
    return np.stack(entries)


def commit_command_and_append_placeholder(
    states: list[np.ndarray],
    commands: list[np.ndarray],
    command: np.ndarray,
    next_state: np.ndarray,
) -> None:
    """Commit ``[x_t, u_t]`` and append ``[x_{t+1}, placeholder]`` in place."""
    if len(states) != len(commands) or len(states) == 0:
        raise ValueError("states and commands must be non-empty and have equal length")
    command_copy = np.asarray(command, dtype=np.float32).copy()
    commands[-1] = command_copy
    states.append(np.asarray(next_state, dtype=np.float32).copy())
    # The rollout function always replaces this latest action before it is
    # consumed. Keeping the last command here makes the placeholder explicit
    # and avoids a second sentinel representation in serialized snapshots.
    commands.append(command_copy.copy())


def future_history_tokens(
    states: list[np.ndarray] | np.ndarray,
    commands: list[np.ndarray] | np.ndarray,
    predicted_states: np.ndarray,
    forecast_commands: np.ndarray,
    history_len: int,
) -> np.ndarray:
    """Build history at a future anchor from a rollout that starts at ``x_t``.

    ``predicted_states`` must contain ``[x_t, x_{t+1}, ..., x_a]`` and
    ``forecast_commands`` contains ``[u_t, ..., u_{a-1}]``.  The returned
    final token is ``[x_a, placeholder]`` for the next CEM candidate action.
    """
    actions = np.asarray(forecast_commands, dtype=np.float32)
    predicted = np.asarray(predicted_states, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError("forecast_commands must have shape [positive_steps, action_dim]")
    if predicted.ndim != 2 or predicted.shape[0] != actions.shape[0] + 1:
        raise ValueError("predicted_states must contain the rollout start plus one state per forecast command")
    base_states = [np.asarray(value, dtype=np.float32) for value in states]
    base_commands = [np.asarray(value, dtype=np.float32) for value in commands]
    if len(base_states) != len(base_commands) or not base_states:
        raise ValueError("states and commands must be non-empty and have equal length")
    # Replace the current placeholder with u_t, append all intermediate
    # state-action pairs, then create the anchor placeholder.
    base_commands[-1] = actions[0]
    for index in range(1, actions.shape[0]):
        base_states.append(predicted[index])
        base_commands.append(actions[index])
    base_states.append(predicted[-1])
    base_commands.append(actions[-1])
    return history_tokens(base_states, base_commands, history_len)
