"""Tests for aligned MPC history construction."""
from __future__ import annotations

import unittest

import numpy as np

from mpc.history import (
    commit_command_and_append_placeholder,
    future_history_tokens,
    history_tokens,
    stale_history_tokens_at_anchor,
)


class ActionAlignedHistoryTests(unittest.TestCase):
    def test_commit_pairs_current_state_with_executed_command(self) -> None:
        states = [np.array([0.0], dtype=np.float32)]
        commands = [np.array([-1.0], dtype=np.float32)]
        commit_command_and_append_placeholder(states, commands, np.array([0.3], dtype=np.float32), np.array([1.0], dtype=np.float32))
        np.testing.assert_allclose(history_tokens(states, commands, 2), [[0.0, 0.3], [1.0, 0.3]])

    def test_history_tokens_accept_snapshot_ndarrays(self) -> None:
        tokens = history_tokens(
            np.array([[1.0], [2.0]], dtype=np.float32),
            np.array([[0.1], [0.2]], dtype=np.float32),
            2,
        )
        np.testing.assert_allclose(tokens, [[1.0, 0.1], [2.0, 0.2]])

    def test_future_anchor_has_correct_intermediate_actions_and_placeholder(self) -> None:
        states = [np.array([4.0], dtype=np.float32), np.array([5.0], dtype=np.float32)]
        commands = [np.array([0.1], dtype=np.float32), np.array([-1.0], dtype=np.float32)]
        tokens = future_history_tokens(
            states, commands,
            predicted_states=np.array([[5.0], [6.0], [7.0]], dtype=np.float32),
            forecast_commands=np.array([[0.2], [0.3]], dtype=np.float32),
            history_len=4,
        )
        np.testing.assert_allclose(tokens, [[4.0, 0.1], [5.0, 0.2], [6.0, 0.3], [7.0, 0.3]])

    def test_stale_history_replaces_only_newest_state(self) -> None:
        states = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        commands = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        launch = history_tokens(states, commands, 3)
        stale = stale_history_tokens_at_anchor(states, commands, np.array([9.0], dtype=np.float32), 3)
        np.testing.assert_array_equal(stale[:-1], launch[:-1])
        np.testing.assert_allclose(stale[-1], [9.0, 0.3])


if __name__ == "__main__":
    unittest.main()
