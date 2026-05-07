from __future__ import annotations

import numpy as np
import pytest

from training.label_schema import (
    SCHEMA_VERSION_V1,
    SCHEMA_VERSION_V2,
    TeacherLabelManifest,
    board_hashes,
    load_labels_npz_any,
    read_teacher_manifest,
    save_labels_npz_v2,
    teacher_policy_from_q,
    write_teacher_manifest,
)


def _labels() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boards = np.array(
        [
            [[1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[3, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=np.int64,
    )
    masks = np.array(
        [[True, False, True, False], [False, True, True, False]],
        dtype=np.bool_,
    )
    actions = np.array([2, 1], dtype=np.int64)
    teacher_q = np.array(
        [[4.0, -1.0e9, 8.0, -1.0e9], [-1.0e9, -1.0e9, -1.0e9, -1.0e9]],
        dtype=np.float32,
    )
    source_indexes = np.array([10, 11], dtype=np.int64)
    return boards, masks, actions, teacher_q, source_indexes


def test_teacher_policy_from_q_masks_illegal_and_falls_back_to_uniform() -> None:
    _boards, masks, _actions, teacher_q, _source_indexes = _labels()

    policy = teacher_policy_from_q(teacher_q, masks)

    assert policy.shape == teacher_q.shape
    np.testing.assert_allclose(policy.sum(axis=1), np.ones(2), rtol=0, atol=1e-6)
    assert policy[0, 1] == 0.0
    assert policy[0, 3] == 0.0
    assert policy[0, 2] > policy[0, 0]
    np.testing.assert_allclose(policy[1], [0.0, 0.5, 0.5, 0.0], rtol=0, atol=1e-6)


def test_board_hashes_are_stable_and_content_based() -> None:
    boards, *_ = _labels()
    hashes_1 = board_hashes(boards)
    hashes_2 = board_hashes(boards.astype(np.int16))

    assert hashes_1.dtype.kind == "U"
    assert np.array_equal(hashes_1, hashes_2)
    assert hashes_1[0] != hashes_1[1]


def test_schema_v2_roundtrip_preserves_q_policy_and_provenance(tmp_path) -> None:
    boards, masks, actions, teacher_q, source_indexes = _labels()
    path = tmp_path / "labels_v2.npz"

    save_labels_npz_v2(
        path=path,
        boards=boards,
        action_masks=masks,
        teacher_actions=actions,
        teacher_q=teacher_q,
        source_indexes=source_indexes,
        stages=3,
        scenarios=10,
        seed=7,
        dataset_path="/tmp/boards.npy",
        teacher_value=np.array([1.0, 2.0], dtype=np.float32),
        policy_checkpoint="models/checkpoint_best.pt",
        episode_id=np.array([100, 100], dtype=np.int64),
        move_idx=np.array([0, 1], dtype=np.int64),
    )

    loaded = load_labels_npz_any(path)

    assert loaded["schema_version"] == SCHEMA_VERSION_V2
    assert loaded["stages"] == 3
    assert loaded["scenarios"] == 10
    assert loaded["seed"] == 7
    assert loaded["dataset_path"] == "/tmp/boards.npy"
    assert loaded["policy_checkpoint"] == "models/checkpoint_best.pt"
    np.testing.assert_array_equal(loaded["boards"], boards)
    np.testing.assert_array_equal(loaded["source_indexes"], source_indexes)
    np.testing.assert_array_equal(loaded["episode_id"], np.array([100, 100]))
    np.testing.assert_array_equal(loaded["move_idx"], np.array([0, 1]))
    np.testing.assert_allclose(loaded["teacher_q"], teacher_q, rtol=0, atol=0)
    np.testing.assert_allclose(
        loaded["teacher_policy"],
        teacher_policy_from_q(teacher_q, masks),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(loaded["teacher_value"], [1.0, 2.0])


def test_schema_v1_load_derives_v2_compat_fields(tmp_path) -> None:
    boards, masks, actions, teacher_q, source_indexes = _labels()
    path = tmp_path / "labels_v1.npz"
    np.savez_compressed(
        path,
        boards=boards,
        action_masks=masks,
        teacher_actions=actions,
        teacher_q=teacher_q,
        source_indexes=source_indexes,
        stages=np.array([1], dtype=np.int64),
        scenarios=np.array([2], dtype=np.int64),
        seed=np.array([3], dtype=np.int64),
    )

    loaded = load_labels_npz_any(path)

    assert loaded["schema_version"] == SCHEMA_VERSION_V1
    assert loaded["stages"] == 1
    assert loaded["scenarios"] == 2
    assert loaded["seed"] == 3
    np.testing.assert_array_equal(loaded["board_hash"], board_hashes(boards))
    np.testing.assert_array_equal(loaded["episode_id"], np.array([-1, -1]))
    np.testing.assert_array_equal(loaded["move_idx"], np.array([-1, -1]))
    np.testing.assert_allclose(
        loaded["teacher_policy"],
        teacher_policy_from_q(teacher_q, masks),
        rtol=0,
        atol=1e-6,
    )


def test_teacher_label_manifest_roundtrip(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    manifest = TeacherLabelManifest(
        dataset_path="/tmp/boards.npy",
        shard_files=["shard_000001.npz"],
        rows=123,
        stages=3,
        scenarios=10,
        seed=7,
        policy_checkpoint="models/best.pt",
    )

    write_teacher_manifest(path, manifest)
    loaded = read_teacher_manifest(path)

    assert loaded == manifest


def test_save_rejects_illegal_teacher_action(tmp_path) -> None:
    boards, masks, actions, teacher_q, source_indexes = _labels()
    actions = actions.copy()
    actions[0] = 1

    with pytest.raises(ValueError, match="teacher_actions must be legal"):
        save_labels_npz_v2(
            path=tmp_path / "bad.npz",
            boards=boards,
            action_masks=masks,
            teacher_actions=actions,
            teacher_q=teacher_q,
            source_indexes=source_indexes,
            stages=1,
            scenarios=1,
            seed=1,
            dataset_path="x",
        )


def test_save_rejects_policy_mass_on_illegal_actions(tmp_path) -> None:
    boards, masks, actions, teacher_q, source_indexes = _labels()
    bad_policy = teacher_policy_from_q(teacher_q, masks)
    bad_policy[0, 1] = 0.1

    with pytest.raises(ValueError, match="zero on illegal"):
        save_labels_npz_v2(
            path=tmp_path / "bad_policy.npz",
            boards=boards,
            action_masks=masks,
            teacher_actions=actions,
            teacher_q=teacher_q,
            source_indexes=source_indexes,
            stages=1,
            scenarios=1,
            seed=1,
            dataset_path="x",
            teacher_policy=bad_policy,
        )
