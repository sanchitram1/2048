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


def test_make_move_no_change_means_no_spawn() -> None:
    random.seed(13)
    game = GameLogic()
    changed, score_gain, spawn_flat, spawn_value = game.make_move("l")
    assert changed is False
    assert score_gain == 0
    assert spawn_flat is None
    assert spawn_value is None
