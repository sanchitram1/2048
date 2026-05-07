from __future__ import annotations

import itertools
import statistics
import time

import numpy as np

from training.mip_nt_stage_dataset import (
    face_board_to_log2_row,
    legal_action_mask_log2,
    simulate_sequence_stochastic,
    spawn_tile_face,
)


def main() -> None:
    rng = np.random.default_rng(42)
    board = np.zeros((4, 4), dtype=np.int32)
    board = spawn_tile_face(board, rng)
    board = spawn_tile_face(board, rng)

    n_horizon = 3
    n_scenarios = 10
    repeats = 3
    sequences = list(itertools.product(range(4), repeat=n_horizon))

    steps = [
        ("board_to_log2", "convert face board to log2 board"),
        ("legal_mask", "compute legal actions from the board"),
        ("build_sequences", "enumerate all action sequences"),
        (
            "simulate_all",
            f"simulate all {len(sequences)} sequences across {n_scenarios} scenarios",
        ),
        ("select_best", "pick the best sequence with argmax"),
    ]

    results: dict[str, list[float]] = {k: [] for k, _ in steps}
    mask = None
    exp_scores = None
    best_idx = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        row = face_board_to_log2_row(board)
        results["board_to_log2"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        mask = legal_action_mask_log2(row)
        results["legal_mask"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        seqs = list(itertools.product(range(4), repeat=n_horizon))
        results["build_sequences"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        exp_scores = np.array(
            [
                simulate_sequence_stochastic(
                    board,
                    seq,
                    n_scenarios=n_scenarios,
                    rng=rng,
                )
                for seq in seqs
            ],
            dtype=np.float64,
        )
        results["simulate_all"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        best_idx = int(np.argmax(exp_scores))
        results["select_best"].append(time.perf_counter() - t0)

    print("step_idx | step | description | mean_s | min_s | max_s")
    print("-" * 74)
    for i, (key, desc) in enumerate(steps, start=1):
        vals = results[key]
        print(
            f"{i:>8} | {key:<14} | {desc:<42} | "
            f"{statistics.mean(vals):>6.4f} | {min(vals):>6.4f} | {max(vals):>6.4f}"
        )
    print()
    print(f"mask={mask.tolist() if mask is not None else None}")
    print(f"best_idx={best_idx}")
    print(
        f"best_score={float(exp_scores[best_idx]) if exp_scores is not None else None}"
    )


if __name__ == "__main__":
    main()
