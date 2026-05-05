"""Merge teacher label npz files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.imitation import load_labels_npz, save_labels_npz
from training.merge_teacher_labels import merge_teacher_npz_files


def test_merge_two_npz_roundtrip(tmp_path: Path) -> None:
    p1 = tmp_path / "a.npz"
    p2 = tmp_path / "b.npz"
    out = tmp_path / "merged.npz"

    save_labels_npz(
        path=p1,
        boards=np.ones((2, 4, 4), dtype=np.int64),
        action_masks=np.ones((2, 4), dtype=np.bool_),
        teacher_actions=np.zeros(2, dtype=np.int64),
        teacher_q=np.zeros((2, 4), dtype=np.float32),
        source_indexes=np.array([10, 11], dtype=np.int64),
        stages=3,
        scenarios=10,
        seed=7,
        dataset_path="a.npy",
    )
    save_labels_npz(
        path=p2,
        boards=np.full((3, 4, 4), 2, dtype=np.int64),
        action_masks=np.ones((3, 4), dtype=np.bool_),
        teacher_actions=np.ones(3, dtype=np.int64),
        teacher_q=np.ones((3, 4), dtype=np.float32) * 0.5,
        source_indexes=None,
        stages=3,
        scenarios=10,
        seed=8,
        dataset_path="b.npy",
    )

    merge_teacher_npz_files([p1, p2], preserve_sources=False, output_path=out)
    m = load_labels_npz(out)
    assert m["boards"].shape[0] == 5
    assert np.array_equal(m["boards"][0], np.ones((4, 4), dtype=np.int64))
    assert np.array_equal(m["boards"][-1], np.full((4, 4), 2, dtype=np.int64))
    si = m["source_indexes"]
    assert si is not None
    assert np.array_equal(si, np.arange(5, dtype=np.int64))


def test_merge_preserve_sources_prefix(tmp_path: Path) -> None:
    p1 = tmp_path / "a.npz"
    p2 = tmp_path / "b.npz"
    out = tmp_path / "merged.npz"

    save_labels_npz(
        path=p1,
        boards=np.zeros((1, 4, 4), dtype=np.int64),
        action_masks=np.ones((1, 4), dtype=np.bool_),
        teacher_actions=np.zeros(1, dtype=np.int64),
        teacher_q=np.zeros((1, 4), dtype=np.float32),
        source_indexes=np.array([5], dtype=np.int64),
        stages=1,
        scenarios=1,
        seed=1,
        dataset_path="x",
    )
    save_labels_npz(
        path=p2,
        boards=np.zeros((1, 4, 4), dtype=np.int64),
        action_masks=np.ones((1, 4), dtype=np.bool_),
        teacher_actions=np.zeros(1, dtype=np.int64),
        teacher_q=np.zeros((1, 4), dtype=np.float32),
        source_indexes=np.array([5], dtype=np.int64),
        stages=1,
        scenarios=1,
        seed=1,
        dataset_path="y",
    )

    merge_teacher_npz_files([p1, p2], preserve_sources=True, output_path=out)
    si = load_labels_npz(out)["source_indexes"]
    assert si is not None
    assert int(si[0]) == 5
    assert int(si[1]) == 1_000_000_000 + 5
