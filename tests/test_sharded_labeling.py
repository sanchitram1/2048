from __future__ import annotations

import json

import numpy as np
import pytest

from training.imitation import (
    ShardLabelManifest,
    load_board_dataset,
    load_labels_merged,
    label_board_states,
    manifest_path,
    run_sharded_labeling,
)


def _synthetic_boards(n: int) -> np.ndarray:
    boards = np.zeros((n, 4, 4), dtype=np.int64)
    boards[:, 0, 0] = 1
    boards[:, 1, 0] = 2
    return boards


def test_sharded_matches_monolithic(tmp_path) -> None:
    ds = tmp_path / "boards.npy"
    np.save(ds, _synthetic_boards(31))

    run_dir = tmp_path / "labels_run"
    clean, manifest = run_sharded_labeling(
        dataset_path=ds,
        dataset_encoding="log2",
        run_dir=run_dir,
        stages=1,
        scenarios=2,
        seed=99,
        chunk_rows=7,
        shard_usable_rows=13,
        resume=False,
        force=False,
        max_boards=None,
        log_every=0,
    )
    assert clean
    assert manifest.complete
    merged = load_labels_merged(run_dir)

    full = load_board_dataset(ds)
    mono_u, mono_m, mono_a, mono_q, mono_s = label_board_states(
        full,
        stages=1,
        scenarios=2,
        seed=99,
        max_boards=None,
        log_every=0,
    )
    assert np.array_equal(merged["boards"], mono_u)
    assert np.array_equal(merged["action_masks"], mono_m)
    assert np.array_equal(merged["teacher_actions"], mono_a)
    assert np.array_equal(merged["teacher_q"], mono_q)
    assert np.array_equal(merged["source_indexes"], mono_s)


def test_resume_completes_dataset(tmp_path) -> None:
    ds = tmp_path / "boards.npy"
    np.save(ds, _synthetic_boards(40))

    run_dir = tmp_path / "lr"
    clean1, m1 = run_sharded_labeling(
        dataset_path=ds,
        dataset_encoding="log2",
        run_dir=run_dir,
        stages=1,
        scenarios=2,
        seed=12,
        chunk_rows=10,
        shard_usable_rows=50,
        resume=False,
        force=False,
        max_boards=8,
        log_every=0,
    )
    assert clean1
    assert not m1.complete
    n_partial = load_labels_merged(run_dir)["boards"].shape[0]
    assert n_partial == 8

    clean2, m2 = run_sharded_labeling(
        dataset_path=ds,
        dataset_encoding="log2",
        run_dir=run_dir,
        stages=1,
        scenarios=2,
        seed=12,
        chunk_rows=10,
        shard_usable_rows=50,
        resume=True,
        force=False,
        max_boards=None,
        log_every=0,
    )
    assert clean2
    assert m2.complete

    merged = load_labels_merged(run_dir)
    full = load_board_dataset(ds)
    mono_u, mono_m, mono_a, mono_q, mono_s = label_board_states(
        full,
        stages=1,
        scenarios=2,
        seed=12,
        max_boards=None,
        log_every=0,
    )
    assert merged["boards"].shape[0] == mono_u.shape[0]
    assert np.array_equal(merged["source_indexes"], mono_s)


def test_resume_manifest_mismatch_exits(tmp_path) -> None:
    ds = tmp_path / "boards.npy"
    np.save(ds, _synthetic_boards(15))
    run_dir = tmp_path / "lr2"
    run_sharded_labeling(
        dataset_path=ds,
        dataset_encoding="log2",
        run_dir=run_dir,
        stages=1,
        scenarios=2,
        seed=1,
        chunk_rows=20,
        shard_usable_rows=100,
        resume=False,
        force=False,
        max_boards=None,
        log_every=0,
    )

    mp = manifest_path(run_dir)
    data = json.loads(mp.read_text(encoding="utf-8"))
    data["seed"] = 999
    mp.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SystemExit):
        run_sharded_labeling(
            dataset_path=ds,
            dataset_encoding="log2",
            run_dir=run_dir,
            stages=1,
            scenarios=2,
            seed=1,
            chunk_rows=20,
            shard_usable_rows=100,
            resume=True,
            force=False,
            max_boards=None,
            log_every=0,
        )


def test_manifest_roundtrip() -> None:
    m = ShardLabelManifest(
        dataset_path="/tmp/x.npy",
        dataset_encoding="log2",
        stages=2,
        scenarios=5,
        seed=3,
        chunk_rows=100,
        shard_usable_rows=200,
        next_raw_row=50,
        global_usable_labeled=40,
        shard_files=["shard_000001.npz"],
        complete=False,
        interrupted=True,
    )
    d = m.to_json_dict()
    m2 = ShardLabelManifest.from_json_dict(d)
    assert m2 == m
