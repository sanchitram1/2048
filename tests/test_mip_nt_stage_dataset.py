"""Tests for MCTS N-stage dataset generation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from training.env import Game2048Env
from training.label_schema import DEFAULT_INVALID_Q_THRESHOLD
from training.mip_nt_stage_dataset import (
    MctsDatasetManifest,
    _validate_shard_rows,
    _flush_shard_buffer,
    _save_manifest_atomic,
    apply_move_face,
    collect_one_game_transitions,
    face_board_to_log2_row,
    legal_action_mask_log2,
    load_mcts_dataset_shards,
    spawn_tile_face,
)


def test_spawn_and_masks_are_consistent() -> None:
    rng = np.random.default_rng(0)
    board = np.zeros((4, 4), dtype=np.int32)
    board = spawn_tile_face(board, rng)
    board = spawn_tile_face(board, rng)
    log2 = face_board_to_log2_row(board)
    mask = legal_action_mask_log2(log2)
    assert mask.shape == (4,)
    assert mask.any()
    d = int(np.argmax(mask))
    nb, _ = apply_move_face(board, d)
    assert not np.array_equal(nb, board)


def test_action_ids_match_env_strings() -> None:
    assert Game2048Env.ACTION_TO_MOVE[0] == "l"
    assert Game2048Env.ACTION_TO_MOVE[3] == "d"


def test_collect_transitions_emit_dense_legal_q_values() -> None:
    rng = np.random.default_rng(7)
    boards, masks, actions, teacher_q, episode_ids, move_idxs = collect_one_game_transitions(
        game_id=11,
        n_horizon=2,
        n_scenarios=3,
        rng=rng,
    )

    assert len(boards) > 0
    assert len(boards) == len(masks) == len(actions) == len(teacher_q)
    assert len(episode_ids) == len(move_idxs) == len(actions)

    q = np.stack(teacher_q, axis=0)
    m = np.stack(masks, axis=0)
    a = np.asarray(actions, dtype=np.int64)

    assert np.all(q[~m] <= DEFAULT_INVALID_Q_THRESHOLD)
    assert np.all(q[m] > DEFAULT_INVALID_Q_THRESHOLD)
    assert np.all(m[np.arange(m.shape[0]), a])
    np.testing.assert_array_equal(np.asarray(episode_ids), np.full(len(actions), 11))
    np.testing.assert_array_equal(np.asarray(move_idxs), np.arange(len(actions)))


def test_sharded_schema_v2_flush_and_load(tmp_path) -> None:
    output_dir = tmp_path / "mcts"
    manifest = MctsDatasetManifest(
        output_dir=str(output_dir),
        stages=2,
        scenarios=10,
        seed=1,
        games_target=2,
        shard_rows=3,
    )
    _save_manifest_atomic(output_dir, manifest)

    boards = np.zeros((3, 4, 4), dtype=np.int64)
    masks = np.array([[True, True, False, False]] * 3, dtype=np.bool_)
    actions = np.array([0, 0, 0], dtype=np.int64)
    teacher_q = np.array(
        [[1.0, 0.5, DEFAULT_INVALID_Q_THRESHOLD, DEFAULT_INVALID_Q_THRESHOLD]] * 3,
        dtype=np.float32,
    )
    episode_id = np.array([0, 0, 1], dtype=np.int64)
    move_idx = np.array([0, 1, 0], dtype=np.int64)
    source_indexes = np.array([0, 1, 2], dtype=np.int64)

    wrote = _flush_shard_buffer(
        output_dir=output_dir,
        manifest=manifest,
        parts_boards=[boards],
        parts_masks=[masks],
        parts_actions=[actions],
        parts_q=[teacher_q],
        parts_episode_id=[episode_id],
        parts_move_idx=[move_idx],
        parts_source_indexes=[source_indexes],
        max_rows=0,
    )

    assert wrote == 3
    manifest_payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["rows"] == 3
    assert manifest_payload["shard_files"] == ["shard_000001.npz"]
    assert not (output_dir / "shard_000001.partial.npz").exists()

    merged = load_mcts_dataset_shards(output_dir)
    tq = merged["teacher_q"]
    pol = merged["teacher_policy"]
    assert isinstance(tq, np.ndarray)
    assert isinstance(pol, np.ndarray)
    assert tq.shape == (3, 4)
    assert pol.shape == (3, 4)
    np.testing.assert_array_equal(merged["episode_id"], episode_id)
    np.testing.assert_array_equal(merged["move_idx"], move_idx)


def test_validate_shard_rows_rejects_all_rows_all_zero_legal_q() -> None:
    masks = np.array([[True, True, False, False], [True, False, True, False]], dtype=np.bool_)
    actions = np.array([0, 0], dtype=np.int64)
    teacher_q = np.array(
        [
            [0.0, 0.0, DEFAULT_INVALID_Q_THRESHOLD, DEFAULT_INVALID_Q_THRESHOLD],
            [0.0, DEFAULT_INVALID_Q_THRESHOLD, 0.0, DEFAULT_INVALID_Q_THRESHOLD],
        ],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="all rows have all legal teacher_q values equal to zero"):
        _validate_shard_rows(masks=masks, actions=actions, teacher_q=teacher_q)


def test_validate_shard_rows_allows_single_zero_legal_q_row() -> None:
    masks = np.array([[True, True, False, False], [True, False, True, False]], dtype=np.bool_)
    actions = np.array([0, 0], dtype=np.int64)
    teacher_q = np.array(
        [
            [0.0, 0.0, DEFAULT_INVALID_Q_THRESHOLD, DEFAULT_INVALID_Q_THRESHOLD],
            [1.0, DEFAULT_INVALID_Q_THRESHOLD, 0.5, DEFAULT_INVALID_Q_THRESHOLD],
        ],
        dtype=np.float32,
    )
    counters = _validate_shard_rows(masks=masks, actions=actions, teacher_q=teacher_q)
    assert counters["rows"] == 2
    assert counters["zero_legal_q_rows"] == 1


def test_validate_shard_rows_rejects_nonfinite_legal_q() -> None:
    masks = np.array([[True, False, True, False]], dtype=np.bool_)
    actions = np.array([0], dtype=np.int64)
    teacher_q = np.array(
        [[np.nan, DEFAULT_INVALID_Q_THRESHOLD, 1.0, DEFAULT_INVALID_Q_THRESHOLD]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="non-finite legal teacher_q"):
        _validate_shard_rows(masks=masks, actions=actions, teacher_q=teacher_q)


def test_validate_shard_rows_rejects_illegal_q_and_argmax_mismatch() -> None:
    masks = np.array([[True, False, True, False]], dtype=np.bool_)
    bad_illegal_q = np.array([[1.0, 5.0, 0.5, DEFAULT_INVALID_Q_THRESHOLD]], dtype=np.float32)
    with pytest.raises(ValueError, match="illegal actions must use invalid threshold"):
        _validate_shard_rows(
            masks=masks,
            actions=np.array([0], dtype=np.int64),
            teacher_q=bad_illegal_q,
        )

    bad_illegal_q_below = np.array(
        [[1.0, DEFAULT_INVALID_Q_THRESHOLD - 123.0, 0.5, DEFAULT_INVALID_Q_THRESHOLD]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="illegal actions must use invalid threshold"):
        _validate_shard_rows(
            masks=masks,
            actions=np.array([0], dtype=np.int64),
            teacher_q=bad_illegal_q_below,
        )

    good_q = np.array(
        [[1.0, DEFAULT_INVALID_Q_THRESHOLD, 3.0, DEFAULT_INVALID_Q_THRESHOLD]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="teacher action 0 != masked argmax 2"):
        _validate_shard_rows(
            masks=masks,
            actions=np.array([0], dtype=np.int64),
            teacher_q=good_q,
        )
