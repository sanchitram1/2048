"""Spawn metadata returned by ``GameLogic.make_move``."""

from __future__ import annotations

import random

from game2048.game import GameLogic


def test_make_move_reports_spawn_after_slide() -> None:
    random.seed(0)
    game = GameLogic()
    changed, score_gain, spawn_flat, spawn_value = game.make_move("l")
    assert changed is True
    assert score_gain == 4
    assert spawn_flat == 8
    assert spawn_value == 2


def test_stop_at_max_tile_ends_episode_when_reached() -> None:
    game = GameLogic(stop_at_max_tile=4)
    game.grid.fill(0)
    game.grid[0, 0] = 1
    game.grid[0, 1] = 1
    game.score = 0
    game.done = False
    random.seed(0)
    changed, _, _, _ = game.make_move("l")
    assert changed is True
    assert game.max_square() >= 4
    assert game.done is True


def test_make_move_no_change_means_no_spawn() -> None:
    random.seed(13)
    game = GameLogic()
    changed, score_gain, spawn_flat, spawn_value = game.make_move("l")
    assert changed is False
    assert score_gain == 0
    assert spawn_flat is None
    assert spawn_value is None
