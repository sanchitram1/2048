from __future__ import annotations

import numpy as np

from game2048.game import GameLogic
from training.td_ntuple import (
    NTupleValueFunction,
    TDNTupleAgentRunner,
    TDNTupleConfig,
    systematic_4_tuples,
)


def test_systematic_4_tuples_match_paper_layout() -> None:
    tuples = systematic_4_tuples()

    assert len(tuples) == 17
    assert all(len(tuple_) == 4 for tuple_ in tuples)
    assert tuples[0] == (0, 1, 2, 3)
    assert tuples[4] == (0, 4, 8, 12)
    assert tuples[-1] == (10, 11, 14, 15)


def test_preview_move_returns_afterstate_without_spawn() -> None:
    game = GameLogic()
    game.grid = np.array(
        [
            [1, 1, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int16,
    )

    afterstate, score_gain, moved = game.preview_move("l")

    assert moved is True
    assert score_gain == 4
    np.testing.assert_array_equal(afterstate[0], np.array([2, 0, 0, 0]))
    np.testing.assert_array_equal(game.grid[0], np.array([1, 1, 0, 0]))


def test_ntuple_checkpoint_round_trip(tmp_path) -> None:
    config = TDNTupleConfig(episodes=3, model_dir=str(tmp_path))
    value_function = NTupleValueFunction(max_exponent=config.max_exponent)
    board = np.zeros((4, 4), dtype=np.int16)

    value_function.update(board, td_error=10.0, alpha=0.1)
    checkpoint_path = value_function.save(
        tmp_path / "td_ntuple_checkpoint_3.npz",
        config,
        episodes=3,
    )

    loaded, loaded_config, episodes = NTupleValueFunction.load(checkpoint_path)

    assert episodes == 3
    assert loaded_config == config
    assert loaded.value(board) == value_function.value(board)


def test_td_runner_streams_board_payload(tmp_path) -> None:
    config = TDNTupleConfig(episodes=1, model_dir=str(tmp_path), seed=1)
    value_function = NTupleValueFunction(max_exponent=config.max_exponent)
    checkpoint_path = value_function.save(
        tmp_path / "td_ntuple_checkpoint_1.npz",
        config,
        episodes=1,
    )
    runner = TDNTupleAgentRunner(checkpoint_path=checkpoint_path)

    first = runner.reset()
    second = runner.step()

    assert first["event"] == "state"
    assert second["event"] in {"agent_move", "game_over"}
    assert second["model_type"] == "td_ntuple"
    assert len(second["tiles"]) == 16
