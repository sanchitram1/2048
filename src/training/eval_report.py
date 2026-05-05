"""Shared rollout metrics for greedy policy evaluation (diagnostics, train-loop eval)."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


def summarize_rollouts(
    scores: Sequence[float],
    max_tiles: Sequence[int],
) -> dict[str, float | int]:
    scores_a = np.asarray(scores, dtype=np.float64)
    n = int(scores_a.size)
    if n == 0:
        return {
            "mean_score": 0.0,
            "median_score": 0.0,
            "score_variance": 0.0,
            "times_reached_256": 0,
            "times_reached_512": 0,
            "times_reached_1024": 0,
            "times_reached_2048": 0,
        }
    mean_score = float(np.mean(scores_a))
    median_score = float(np.median(scores_a))
    score_variance = float(scores_a.var(ddof=1)) if n > 1 else 0.0

    def count_at_least(threshold: int) -> int:
        return sum(1 for t in max_tiles if int(t) >= threshold)

    return {
        "mean_score": mean_score,
        "median_score": median_score,
        "score_variance": score_variance,
        "times_reached_256": count_at_least(256),
        "times_reached_512": count_at_least(512),
        "times_reached_1024": count_at_least(1024),
        "times_reached_2048": count_at_least(2048),
    }


def print_rollout_eval_summary(
    *,
    episodes: int,
    scores: list[float],
    max_tiles: list[int],
    eval_base_seed: int,
    header: str | None = None,
) -> None:
    """Human-readable summary for CLI diagnostics."""
    m = summarize_rollouts(scores, max_tiles)
    tile_counts = Counter(max_tiles)

    if header:
        print(header)

    print("\nTrue 2048 performance (greedy policy)")
    print(f"Episodes: {episodes}")
    print(f"Eval RNG base_seed: {eval_base_seed} (episode i uses base_seed + i)")

    print("\nTile distribution:")
    for tile in sorted(tile_counts):
        count = tile_counts[tile]
        pct = 100 * count / episodes
        print(f"{tile}: {count}/{episodes} games ({pct:.1f}%)")

    print("\nReach counts (max tile ≥ threshold):")
    print(f"  times_reached_256:  {m['times_reached_256']}/{episodes}")
    print(f"  times_reached_512:  {m['times_reached_512']}/{episodes}")
    print(f"  times_reached_1024: {m['times_reached_1024']}/{episodes}")
    print(f"  times_reached_2048: {m['times_reached_2048']}/{episodes}")

    print("\nScore summary:")
    if scores:
        print(f"Mean score:   {m['mean_score']:.2f}")
        print(f"Median score: {m['median_score']:.2f}")
        print(f"Var(score):   {m['score_variance']:.2f} (sample variance, ddof=1)")
        print(f"Max score: {max(scores):.2f}")
        print(f"Min score: {min(scores):.2f}")
