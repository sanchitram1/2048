from __future__ import annotations

from evaluation.diagnostics import collect_myopic_greedy_rollouts


def test_myopic_greedy_rollouts_are_deterministic() -> None:
    a_scores, a_tiles = collect_myopic_greedy_rollouts(episodes=5, eval_base_seed=1000)
    b_scores, b_tiles = collect_myopic_greedy_rollouts(episodes=5, eval_base_seed=1000)
    assert a_scores == b_scores
    assert a_tiles == b_tiles
