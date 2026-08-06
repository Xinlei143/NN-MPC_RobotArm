from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpc.history import history_tokens


@dataclass(frozen=True)
class CompletedTransition:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    command_tick_index: int
    history_generation: int


class PendingActionHistory:
    """Real-time [x_t,u_t] history where x_(t+1) arrives one tick later."""

    def __init__(self, initial_state: np.ndarray, action_dim: int, history_len: int):
        if action_dim <= 0 or history_len <= 0:
            raise ValueError("action_dim and history_len must be positive")
        self.action_dim = int(action_dim)
        self.history_len = int(history_len)
        self.generation = 0
        self.reset(initial_state)

    def reset(self, state: np.ndarray) -> None:
        value = np.asarray(state, dtype=np.float32).copy()
        self.generation += 1
        self.states = [value]
        self.commands = [np.zeros(self.action_dim, dtype=np.float32)]
        self._pending: tuple[np.ndarray, int] | None = None

    def record_transmission(self, command: np.ndarray, tick_index: int) -> None:
        if self._pending is not None:
            raise RuntimeError("pending command has not been completed by a new state")
        value = np.asarray(command, dtype=np.float32)
        if value.shape != (self.action_dim,):
            raise ValueError(f"command must have shape ({self.action_dim},)")
        self._pending = (value.copy(), int(tick_index))

    def observe(self, next_state: np.ndarray) -> CompletedTransition | None:
        value = np.asarray(next_state, dtype=np.float32).copy()
        if self._pending is None:
            self.states[-1] = value
            return None
        action, tick_index = self._pending
        previous = self.states[-1].copy()
        self.commands[-1] = action.copy()
        self.states.append(value)
        self.commands.append(action.copy())  # explicit placeholder, replaced on transmission completion
        self._pending = None
        return CompletedTransition(previous, action.copy(), value.copy(), tick_index, self.generation)

    def skip_and_reanchor(self, state: np.ndarray) -> None:
        """Drop a bad-dt transition without declaring a new history generation."""
        value = np.asarray(state, dtype=np.float32).copy()
        self.states[-1] = value
        self.commands[-1] = np.zeros(self.action_dim, dtype=np.float32)
        self._pending = None

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, int]:
        return (
            np.stack(self.states[-self.history_len :]).astype(np.float32),
            np.stack(self.commands[-self.history_len :]).astype(np.float32),
            self.generation,
        )

    def tokens(self) -> np.ndarray:
        return history_tokens(self.states, self.commands, self.history_len)
