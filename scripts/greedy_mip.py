from __future__ import annotations

import argparse
import random

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
    """direction: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN."""
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


def greedy_move(board: np.ndarray, *, rng: random.Random) -> tuple[int | None, float]:
    """Choose the move with maximal immediate merge score."""
    action_dim = 4
    boards_after: list[np.ndarray] = []
    scores: list[float] = []

    for action in range(action_dim):
        new_board, merge_score = apply_move(board, action)
        boards_after.append(new_board)
        scores.append(float(merge_score))

    if all(np.array_equal(boards_after[action], board) for action in range(action_dim)):
        return None, 0.0

    best_score = max(scores)
    best_actions = [i for i, score in enumerate(scores) if score == best_score]
    best_action = int(rng.choice(best_actions))
    return best_action, float(best_score)


def run_mip_greedy_game(*, seed: int | None = None) -> dict[str, object]:
    rng = random.Random(seed)
    if seed is not None:
        np.random.seed(seed)

    board = np.zeros((4, 4), dtype=int)
    board = spawn_tile(board, rng=rng)
    board = spawn_tile(board, rng=rng)

    total_score = 0
    moves = 0

    while True:
        action, _score = greedy_move(board, rng=rng)
        if action is None:
            break

        board, gained = apply_move(board, action)
        total_score += int(gained)
        moves += 1
        board = spawn_tile(board, rng=rng)

    return {
        "score": int(total_score),
        "max_tile": int(board.max()),
        "moves": int(moves),
        "final_board": board,
    }


def simulate_mip_greedy(
    *, n_games: int = 20, seed: int | None = None
) -> list[dict[str, object]]:
    results = [
        run_mip_greedy_game(seed=None if seed is None else seed + i)
        for i in range(n_games)
    ]
    scores = [int(result["score"]) for result in results]
    max_tiles = [int(result["max_tile"]) for result in results]

    print(f"Games played: {n_games}")
    print(f"Average score: {np.mean(scores):.1f}")
    print(f"Median score:  {np.median(scores):.1f}")
    print(f"Max score:     {np.max(scores)}")
    print(f"Average max tile: {np.mean(max_tiles):.1f}")
    print("Max tile distribution:")
    for tile in sorted(set(max_tiles)):
        print(f"  {tile}: {max_tiles.count(tile)} games")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedy 2048 player using a MIP argmax."
    )
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    simulate_mip_greedy(n_games=args.games, seed=args.seed)


if __name__ == "__main__":
    main()
