#!/usr/bin/env pkgx uv run
"""Generate labeled 2048 datasets by rolling out the notebook N-stage MIP-EV policy.

This ports ``notebooks/Greedy and MCTS/2048 MTS N-Trajectories.ipynb`` (face-value
tiles on a NumPy grid + CVXPY one-hot over length-N move sequences). Each recorded
row is a board **before** the chosen slide, with ``teacher_actions`` = planner move
and ``action_masks`` from ``game2048.game.GameLogic`` (log2 encoding).

Requires optional extra::

    uv sync --extra mip-dataset

Install::

    uv run --extra mip-dataset gen-mip-dataset --help
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path

import numpy as np

from game2048.game import GameLogic
from training.env import Game2048Env
from training.imitation import boards_face_values_to_log2, save_labels_npz

_LOG = logging.getLogger("game2048.mip_nt_stage_dataset")

_shutdown_requested = False


def _request_shutdown(_signum: int, _frame: object | None) -> None:
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    _LOG.warning(
        "[mip-dataset] shutdown requested — will finish the current game, save, then exit"
    )


def shutdown_requested() -> bool:
    return bool(_shutdown_requested)


def _reset_shutdown() -> None:
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = False


# --- 2048 engine (face tile values 2, 4, …), copied from the notebook ----------------


def _compress(row: np.ndarray) -> np.ndarray:
    non_zero = row[row != 0]
    return np.concatenate([non_zero, np.zeros(len(row) - len(non_zero), dtype=row.dtype)])


def _merge(row: np.ndarray) -> tuple[np.ndarray, int]:
    row = row.copy()
    score = 0
    for i in range(len(row) - 1):
        if row[i] != 0 and row[i] == row[i + 1]:
            row[i] *= 2
            row[i + 1] = 0
            score += row[i]
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


def apply_move_face(board: np.ndarray, direction: int) -> tuple[np.ndarray, int]:
    if direction == 0:
        return _move_left(board)
    if direction == 1:
        return _move_right(board)
    if direction == 2:
        return _move_up(board)
    if direction == 3:
        return _move_down(board)
    raise ValueError("Invalid direction")


def spawn_tile_face(board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    empty = list(zip(*np.where(board == 0)))
    if not empty:
        return board
    i, j = empty[int(rng.integers(0, len(empty)))]
    board = board.copy()
    board[i, j] = 2 if rng.random() < 0.9 else 4
    return board


def simulate_sequence_stochastic(
    board: np.ndarray,
    seq: tuple[int, ...],
    *,
    n_scenarios: int,
    rng: np.random.Generator,
) -> float:
    scores = []
    for _ in range(n_scenarios):
        b = board.copy()
        total = 0
        feasible = True
        for d in seq:
            new_b, s = apply_move_face(b, d)
            if np.array_equal(new_b, b):
                feasible = False
                break
            total += s
            b = spawn_tile_face(new_b, rng)
        if feasible:
            scores.append(total)
        else:
            scores.append(-1e6)
    return float(np.mean(scores))


def mip_n_stage_expected_move(
    board: np.ndarray,
    *,
    n_horizon: int,
    n_scenarios: int,
    rng: np.random.Generator,
) -> tuple[int, tuple[int, ...], float]:
    import itertools

    import cvxpy as cp

    sequences = list(itertools.product(range(4), repeat=n_horizon))
    k = len(sequences)
    exp_scores = [
        simulate_sequence_stochastic(board, seq, n_scenarios=n_scenarios, rng=rng)
        for seq in sequences
    ]
    exp_scores_arr = np.array(exp_scores, dtype=np.float64)

    y = cp.Variable(k, boolean=True)
    constraints = [cp.sum(y) == 1]
    objective = cp.Maximize(exp_scores_arr @ y)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS)

    best_idx = int(np.argmax(np.asarray(y.value).reshape(-1)))
    best_seq = sequences[best_idx]
    best_score = float(exp_scores_arr[best_idx])
    return int(best_seq[0]), best_seq, best_score


def legal_action_mask_log2(log2_board: np.ndarray) -> np.ndarray:
    logic = GameLogic()
    logic.grid = np.asarray(log2_board, dtype=np.int16).reshape(4, 4)
    logic.score = 0
    logic.done = False
    mask = np.zeros(4, dtype=np.bool_)
    for aid in range(4):
        mv = Game2048Env.ACTION_TO_MOVE[aid]
        _, _, moved = logic.preview_move(mv)
        mask[aid] = moved
    return mask


def face_board_to_log2_row(face_board: np.ndarray) -> np.ndarray:
    slab = np.asarray(face_board, dtype=np.int64).reshape(1, 4, 4)
    return boards_face_values_to_log2(slab)[0]


def collect_one_game_transitions(
    *,
    n_horizon: int,
    n_scenarios: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Return parallel lists: log2 boards (4x4 int64), masks (4,), teacher actions."""
    board = np.zeros((4, 4), dtype=np.int32)
    board = spawn_tile_face(board, rng)
    board = spawn_tile_face(board, rng)

    boards_log2: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []

    while True:
        log2 = face_board_to_log2_row(board)
        mask = legal_action_mask_log2(log2)
        if not mask.any():
            break

        direction, _seq, _ev = mip_n_stage_expected_move(
            board, n_horizon=n_horizon, n_scenarios=n_scenarios, rng=rng
        )
        if not mask[direction]:
            _LOG.warning(
                "[mip-dataset] planner picked illegal move %s — dropping remainder of game",
                direction,
            )
            break

        new_board, _gained = apply_move_face(board, direction)
        if np.array_equal(new_board, board):
            break

        boards_log2.append(log2.astype(np.int64, copy=False))
        masks.append(mask.copy())
        actions.append(int(direction))

        board = spawn_tile_face(new_board, rng)

    return boards_log2, masks, actions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Roll out N-stage MIP-EV 2048 games (notebook policy); save imitation-style "
            ".npz (boards log2, masks, teacher_actions). Ctrl+C saves progress."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/mip_nt_generated_labels.npz"),
        help="Labels artifact compatible with uv run imitate --train-only --labels …",
    )
    p.add_argument(
        "--games",
        type=int,
        default=0,
        help="Stop after this many completed games (0 = run until Ctrl+C).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-stage", type=int, default=3, metavar="N")
    p.add_argument("--n-scenarios", type=int, default=20)
    p.add_argument(
        "--autosave-every-games",
        type=int,
        default=1,
        metavar="K",
        help="Rewrite output after every K finished games (1 = safest against crash).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _concat_batches(
    parts_boards: list[np.ndarray],
    parts_masks: list[np.ndarray],
    parts_actions: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not parts_boards:
        zero_b = np.zeros((0, 4, 4), dtype=np.int64)
        zero_m = np.zeros((0, 4), dtype=np.bool_)
        zero_a = np.zeros(0, dtype=np.int64)
        return zero_b, zero_m, zero_a
    return (
        np.concatenate(parts_boards, axis=0),
        np.concatenate(parts_masks, axis=0),
        np.concatenate(parts_actions, axis=0),
    )


def _persist(
    path: Path,
    *,
    boards: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    n_stage: int,
    n_scenarios: int,
    seed: int,
) -> None:
    if boards.shape[0] == 0:
        _LOG.warning("[mip-dataset] nothing to save yet")
        return
    n = int(boards.shape[0])
    teacher_q = np.zeros((n, 4), dtype=np.float32)
    src = np.arange(n, dtype=np.int64)
    save_labels_npz(
        path=path,
        boards=boards,
        action_masks=masks,
        teacher_actions=actions,
        teacher_q=teacher_q,
        source_indexes=src,
        stages=int(n_stage),
        scenarios=int(n_scenarios),
        seed=int(seed),
        dataset_path=f"mip_nt_stage_generated|n_stage={n_stage}|n_scenarios={n_scenarios}",
    )
    _LOG.info("[mip-dataset] saved %s rows → %s", n, path)


def main() -> None:
    args = parse_args()
    _configure_logging(verbose=bool(args.verbose))
    _reset_shutdown()
    prev_int = signal.signal(signal.SIGINT, _request_shutdown)
    prev_term = signal.signal(signal.SIGTERM, _request_shutdown)

    try:
        try:
            import cvxpy as _cp  # noqa: F401
        except ImportError:
            _LOG.error(
                "[mip-dataset] cvxpy not installed — run: uv sync --extra mip-dataset "
                "(or uv run --extra mip-dataset gen-mip-dataset …)"
            )
            raise SystemExit(1) from None

        rng = np.random.default_rng(int(args.seed))
        out_path = args.output.expanduser().resolve()

        batches_b: list[np.ndarray] = []
        batches_m: list[np.ndarray] = []
        batches_a: list[np.ndarray] = []

        games_done = 0
        t0 = time.perf_counter()

        while True:
            if shutdown_requested():
                break
            if args.games > 0 and games_done >= args.games:
                break

            bl, ml, al = collect_one_game_transitions(
                n_horizon=int(args.n_stage),
                n_scenarios=int(args.n_scenarios),
                rng=rng,
            )
            games_done += 1

            if bl:
                batches_b.append(np.stack(bl, axis=0))
                batches_m.append(np.stack(ml, axis=0))
                batches_a.append(np.asarray(al, dtype=np.int64))

            if games_done % max(1, int(args.autosave_every_games)) == 0:
                bb, mm, aa = _concat_batches(batches_b, batches_m, batches_a)
                _persist(
                    out_path,
                    boards=bb,
                    masks=mm,
                    actions=aa,
                    n_stage=int(args.n_stage),
                    n_scenarios=int(args.n_scenarios),
                    seed=int(args.seed),
                )

            if games_done % 5 == 0 or shutdown_requested():
                elapsed = time.perf_counter() - t0
                rows = sum(b.shape[0] for b in batches_b)
                _LOG.info(
                    "[mip-dataset] games=%s rows=%s elapsed=%.1fs",
                    games_done,
                    rows,
                    elapsed,
                )

        bb, mm, aa = _concat_batches(batches_b, batches_m, batches_a)
        _persist(
            out_path,
            boards=bb,
            masks=mm,
            actions=aa,
            n_stage=int(args.n_stage),
            n_scenarios=int(args.n_scenarios),
            seed=int(args.seed),
        )
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        _reset_shutdown()


if __name__ == "__main__":
    main()
