from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from training.imitation import (
    compute_train_val_indices_by_source,
    compute_train_val_indices_row_shuffle,
    evaluate_teacher_agreement,
)


class _PreferActionZero(nn.Module):
    """Argmax always prefers move 0 (when legal under masking test harness)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        out = torch.full((b, 4), -1e9, dtype=torch.float32, device=x.device)
        out[:, 0] = 0.0
        return out


def test_row_shuffle_split_disjoint_covers_all_rows() -> None:
    tr, va = compute_train_val_indices_row_shuffle(
        100, val_fraction=0.23, split_seed=3
    )
    assert np.unique(np.concatenate([tr, va])).shape[0] == 100
    assert set(tr.tolist()).isdisjoint(va.tolist())
    assert va.shape[0] == int(np.floor(0.23 * 100))


def test_source_group_split_no_shared_sources() -> None:
    src = np.array([1, 1, 2, 2, 3, 3, 40], dtype=np.int64)
    tr, va = compute_train_val_indices_by_source(src, val_fraction=0.34, split_seed=11)
    assert set(tr.tolist()).isdisjoint(va.tolist())
    st = set(src[tr].tolist())
    sv = set(src[va].tolist())
    assert not st & sv


def test_evaluate_teacher_agreement_perfect_match() -> None:
    net = _PreferActionZero()
    n = 24
    boards = np.zeros((n, 4, 4), dtype=np.int64)
    masks = np.zeros((n, 4), dtype=np.bool_)
    masks[:, 0] = True
    tac = np.zeros(n, dtype=np.int64)
    idx = np.arange(n, dtype=np.int64)
    dev = torch.device("cpu")
    m = evaluate_teacher_agreement(
        net,
        boards,
        masks,
        tac,
        idx,
        device=dev,
        batch_size=8,
    )
    assert m.n == n
    assert m.exact_match_rate == pytest.approx(1.0)
    assert m.teacher_action_prob_mean == pytest.approx(1.0)


def test_teacher_agreement_metrics_nan_when_empty() -> None:
    net = _PreferActionZero()
    m = evaluate_teacher_agreement(
        net,
        np.zeros((2, 4, 4), dtype=np.int64),
        np.zeros((2, 4), dtype=np.bool_),
        np.zeros(2, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        device=torch.device("cpu"),
        batch_size=4,
    )
    assert m.n == 0
    assert np.isnan(m.exact_match_rate)
    assert np.isnan(m.teacher_action_prob_mean)
