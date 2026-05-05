"""Tests for MIP N-stage trajectory dataset generation (no cvxpy required)."""

from __future__ import annotations

import numpy as np

from training.env import Game2048Env
from training.mip_nt_stage_dataset import (
    apply_move_face,
    face_board_to_log2_row,
    legal_action_mask_log2,
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
    # Mirror notebook directions 0–3 and env mapping
    d = int(np.argmax(mask))
    nb, _ = apply_move_face(board, d)
    assert not np.array_equal(nb, board)


def test_action_ids_match_env_strings() -> None:
    assert Game2048Env.ACTION_TO_MOVE[0] == "l"
    assert Game2048Env.ACTION_TO_MOVE[3] == "d"
