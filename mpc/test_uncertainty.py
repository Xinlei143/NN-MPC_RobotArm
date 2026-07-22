from __future__ import annotations

import numpy as np
import torch

from mpc.uncertainty import normalized_rms_disagreement, selected_branch_sequences


def test_normalized_rms_disagreement_is_zero_for_identical_members() -> None:
    predictions = torch.zeros((5, 2, 4, 12), dtype=torch.float32)
    scores = normalized_rms_disagreement(predictions, torch.ones(12))
    assert torch.allclose(scores, torch.zeros(2))


def test_normalized_rms_disagreement_ignores_common_initial_state() -> None:
    predictions = torch.zeros((2, 1, 3, 2), dtype=torch.float32)
    predictions[1, 0, 0] = torch.tensor((100.0, -100.0))
    predictions[1, 0, 1:] = torch.tensor((2.0, 0.0))
    score = normalized_rms_disagreement(predictions, torch.tensor((2.0, 1.0)))
    # Population std is one on q; normalization makes it 0.5, then RMS spans q/dq.
    assert np.isclose(float(score[0]), np.sqrt(0.125), atol=1e-6)


def test_selected_branch_sequences_always_contains_selected() -> None:
    selected = np.zeros((3, 6), dtype=np.float32)

    class Result:
        selected_q_ref_sequence = selected
        branch_candidates = ()

    sequences = selected_branch_sequences(Result())
    assert list(sequences) == ["selected"]
    assert np.array_equal(sequences["selected"], selected)
