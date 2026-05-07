"""Merge teacher label npz files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training.imitation import ShardLabelManifest, load_labels_npz, save_labels_npz, save_shard_manifest_atomic
from training.merge_teacher_labels import blend_teacher_labels, merge_teacher_npz_files


def _write_labels(path: Path, *, rows: int, tile: int, source_offset: int) -> None:
    save_labels_npz(
        path=path,
        boards=np.full((rows, 4, 4), tile, dtype=np.int64),
        action_masks=np.ones((rows, 4), dtype=np.bool_),
        teacher_actions=np.full(rows, tile % 4, dtype=np.int64),
        teacher_q=np.full((rows, 4), float(tile), dtype=np.float32),
        source_indexes=np.arange(rows, dtype=np.int64) + source_offset,
        stages=1,
        scenarios=1,
        seed=1,
        dataset_path=str(path),
    )


def _write_sharded_dir(path: Path, *, shard_name: str, rows: int, tile: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    shard_path = path / shard_name
    _write_labels(shard_path, rows=rows, tile=tile, source_offset=100)
    manifest = ShardLabelManifest(
        dataset_path="synthetic",
        dataset_encoding="log2",
        stages=1,
        scenarios=1,
        seed=1,
        chunk_rows=rows,
        shard_usable_rows=rows,
        next_raw_row=rows,
        global_usable_labeled=rows,
        shard_files=[shard_name],
        complete=True,
        interrupted=False,
    )
    save_shard_manifest_atomic(path, manifest)


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


def test_blend_rows_counts_seed_and_metadata(tmp_path: Path) -> None:
    expert = tmp_path / "expert.npz"
    mcts = tmp_path / "mcts_run"
    out1 = tmp_path / "blend1.npz"
    out2 = tmp_path / "blend2.npz"
    _write_labels(expert, rows=8, tile=7, source_offset=0)
    _write_sharded_dir(mcts, shard_name="shard_000001.npz", rows=2, tile=9)

    specs = [f"{expert}:80", f"{mcts}:20"]
    blend_teacher_labels(input_specs=specs, rows=10, seed=1000, output_path=out1)
    blend_teacher_labels(input_specs=specs, rows=10, seed=1000, output_path=out2)

    p1 = load_labels_npz(out1)
    p2 = load_labels_npz(out2)
    assert int(p1["boards"].shape[0]) == 10
    assert np.array_equal(p1["boards"], p2["boards"])
    assert np.array_equal(p1["teacher_actions"], p2["teacher_actions"])
    assert np.array_equal(p1["source_indexes"], p2["source_indexes"])

    si = p1["source_indexes"]
    assert si is not None
    source_ids = (si // 1_000_000_000).astype(np.int64)
    assert int(np.sum(source_ids == 0)) == 8
    assert int(np.sum(source_ids == 1)) == 2

    dataset_path = str(p1["dataset_path"])
    assert dataset_path.startswith("blend|")
    meta = dataset_path.split("|", 1)[1]
    assert '"rows":10' in meta
    assert '"seed":1000' in meta
    assert '"selected_rows":8' in meta
    assert '"selected_rows":2' in meta


def test_blend_raises_when_rows_exceed_available(tmp_path: Path) -> None:
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    _write_labels(a, rows=5, tile=1, source_offset=0)
    _write_labels(b, rows=4, tile=2, source_offset=0)

    try:
        blend_teacher_labels(
            input_specs=[f"{a}:80", f"{b}:20"],
            rows=11,
            seed=1,
            output_path=tmp_path / "unused.npz",
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "requested" in str(exc)


def test_blend_largest_remainder_allocation_is_exact(tmp_path: Path) -> None:
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    c = tmp_path / "c.npz"
    out = tmp_path / "blend.npz"
    _write_labels(a, rows=10, tile=1, source_offset=0)
    _write_labels(b, rows=10, tile=2, source_offset=0)
    _write_labels(c, rows=10, tile=3, source_offset=0)

    blend_teacher_labels(
        input_specs=[f"{a}:1", f"{b}:1", f"{c}:1"],
        rows=7,
        seed=5,
        output_path=out,
    )

    payload = load_labels_npz(out)
    si = payload["source_indexes"]
    assert si is not None
    source_ids = (si // 1_000_000_000).astype(np.int64)
    counts = [int(np.sum(source_ids == i)) for i in range(3)]
    # floor([2.333, 2.333, 2.333]) + one largest remainder slot -> [3, 2, 2]
    assert counts == [3, 2, 2]
