from __future__ import annotations

import argparse
import itertools
import random

import cvxpy as cp
import numpy as np


def _compress(row: np.ndarray) -> np.ndarray:
    non_zero = row[row != 0]
    return np.concatenate(
        [non_zero, np.zeros(len(row) - len(non_zero), dtype=row.dtype)]
    )


def _merge(row: np.ndarray) -> tuple[np.ndarray, int]:
    row = row.copy()
    score = 0
    for i in range(len(row) - 1):
        if row[i] != 0 and row[i] == row[i + 1]:
            row[i] *= 2
            row[i + 1] = 0
            score += int(row[i])
    return row, score


def _move_left(board: np.ndarray) -> tuple[np.ndarray, int]:
    new_board = np.zeros_like(board)
    total_score = 0
    for i in range(board.shape[0]):
        row = board[i, :]
        row = _compress(row)
        row, score = _merge(row)
        row = _compress(row)
        new_board[i, :] = row
        total_score += score
    return new_board, total_score


def _move_right(board: np.ndarray) -> tuple[np.ndarray, int]:
    flipped = np.fliplr(board)
    moved, score = _move_left(flipped)
    return np.fliplr(moved), score


def _move_up(board: np.ndarray) -> tuple[np.ndarray, int]:
    rotated = board.T
    moved, score = _move_left(rotated)
    return moved.T, score


def _move_down(board: np.ndarray) -> tuple[np.ndarray, int]:
    rotated = board.T
    moved, score = _move_right(rotated)
    return moved.T, score


def apply_move(board: np.ndarray, direction: int) -> tuple[np.ndarray, int]:
    if direction == 0:
        return _move_left(board)
    if direction == 1:
        return _move_right(board)
    if direction == 2:
        return _move_up(board)
    if direction == 3:
        return _move_down(board)
    raise ValueError("Invalid direction")


def spawn_tile(board: np.ndarray, *, rng: random.Random) -> np.ndarray:
    empty = list(zip(*np.where(board == 0)))
    if not empty:
        return board
    i, j = rng.choice(empty)
    board[i, j] = 2 if rng.random() < 0.9 else 4
    return board


def simulate_sequence_stochastic(
    board: np.ndarray,
    seq: tuple[int, ...],
    *,
    n_scenarios: int = 20,
    rng: random.Random,
) -> float:
    """Monte Carlo rollout for a fixed move sequence."""
    scores: list[float] = []
    for _ in range(n_scenarios):
        b = board.copy()
        total = 0
        feasible = True
        for action in seq:
            new_b, gained = apply_move(b, action)
            if np.array_equal(new_b, b):
                feasible = False
                break
            total += gained
            b = spawn_tile(new_b, rng=rng)
        scores.append(float(total) if feasible else -1.0e6)
    return float(np.mean(scores))


def mip_n_stage_expected_move(
    board: np.ndarray,
    *,
    n_stages: int = 3,
    n_scenarios: int = 20,
    rng: random.Random,
) -> tuple[int, tuple[int, ...], float]:
    """Pick the first action of the best N-step sequence under MC expected value."""
    action_dim = 4
    sequences = list(itertools.product(range(action_dim), repeat=n_stages))
    exp_scores = np.array(
        [
            simulate_sequence_stochastic(board, seq, n_scenarios=n_scenarios, rng=rng)
            for seq in sequences
        ],
        dtype=float,
    )

    y = cp.Variable(len(sequences), boolean=True)
    constraints = [cp.sum(y) == 1]
    objective = cp.Maximize(exp_scores @ y)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS)

    best_idx = int(np.argmax(y.value))
    best_seq = sequences[best_idx]
    return (
        int(best_seq[0]),
        tuple(int(a) for a in best_seq),
        float(exp_scores[best_idx]),
    )


def run_n_stage_ev_game(
    *,
    n_stages: int = 3,
    n_scenarios: int = 20,
    seed: int | None = None,
) -> dict[str, object]:
    rng = random.Random(seed)
    if seed is not None:
        np.random.seed(seed)

    board = np.zeros((4, 4), dtype=int)
    board = spawn_tile(board, rng=rng)
    board = spawn_tile(board, rng=rng)

    total_score = 0
    moves = 0

    while True:
        action, _seq, _ev = mip_n_stage_expected_move(
            board, n_stages=n_stages, n_scenarios=n_scenarios, rng=rng
        )
        new_board, gained = apply_move(board, action)
        if np.array_equal(new_board, board):
            break
        total_score += gained
        moves += 1
        board = spawn_tile(new_board, rng=rng)

    return {
        "score": int(total_score),
        "max_tile": int(board.max()),
        "moves": int(moves),
        "final_board": board,
    }


def simulate_n_stage_ev(
    *,
    n_stages: int = 3,
    n_scenarios: int = 20,
    n_games: int = 10,
    seed: int | None = None,
) -> list[dict[str, object]]:
    results = [
        run_n_stage_ev_game(
            n_stages=n_stages,
            n_scenarios=n_scenarios,
            seed=None if seed is None else seed + i,
        )
        for i in range(n_games)
    ]
    scores = [int(result["score"]) for result in results]
    max_tiles = [int(result["max_tile"]) for result in results]

    print(f"N-stage EV planner: N={n_stages}, scenarios={n_scenarios}")
    print(f"Games: {n_games}")
    print(f"Avg score: {np.mean(scores):.1f}")
    print(f"Median score: {np.median(scores):.1f}")
    print(f"Max score: {np.max(scores)}")
    print(f"Avg max tile: {np.mean(max_tiles):.1f}")
    print("Max tile distribution:")
    for tile in sorted(set(max_tiles)):
        print(f"  {tile}: {max_tiles.count(tile)} games")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="N-stage expected-value 2048 planner.")
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    simulate_n_stage_ev(
        n_stages=args.stages,
        n_scenarios=args.scenarios,
        n_games=args.games,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
