"""Tests for Monte Carlo planning helpers."""

from __future__ import annotations

import random

import numpy as np

from game2048.game import GameLogic
from training.planning import choose_n_step_mc, _merge_gain_to_planner_float


def test_merge_gain_huge_int_caps_without_overflow() -> None:
    """Pathological merges (huge exponents) must not raise when cast to float."""
    gain = 2**2000
    x = _merge_gain_to_planner_float(gain)
    assert x == float(np.finfo(np.float32).max)


def test_choose_n_step_mc_huge_merge_score_no_crash() -> None:
    """Dataset-style boards with enormous tiles must not break MC labeling."""
    game = GameLogic()
    grid = np.zeros((4, 4), dtype=np.int16)
    grid[0, 0] = 800
    grid[0, 1] = 800
    game.grid = grid
    planned = choose_n_step_mc(
        game, stages=1, scenarios=2, rng=random.Random(0)
    )
    assert planned.action in planned.legal_actions
    assert len(planned.q_values) == 4
