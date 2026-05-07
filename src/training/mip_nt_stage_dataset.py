#!/usr/bin/env pkgx uv run
"""Generate MCTS-labeled 2048 datasets as schema-v2 shard runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from game2048.game import GameLogic
from training.env import Game2048Env
from training.imitation import boards_face_values_to_log2
from training.label_schema import DEFAULT_INVALID_Q_THRESHOLD, load_labels_npz_any, save_labels_npz_v2

_LOG = logging.getLogger("game2048.mip_nt_stage_dataset")
_shutdown_requested = False
MANIFEST_VERSION = 1


def _request_shutdown(_signum: int, _frame: object | None) -> None:
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    _LOG.warning(
        "[mcts-dataset] shutdown requested — will finish the current game, save, then exit"
    )


def shutdown_requested() -> bool:
    return bool(_shutdown_requested)


def _reset_shutdown() -> None:
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = False


@dataclass
class MctsDatasetManifest:
    format_version: int = MANIFEST_VERSION
    output_dir: str = ""
    stages: int = 2
    scenarios: int = 10
    seed: int = 42
    games_target: int = 0
    shard_rows: int = 8192
    next_game_id: int = 0
    next_source_index: int = 0
    rows: int = 0
    shard_files: list[str] = field(default_factory=list)
    complete: bool = False
    interrupted: bool = False

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, object]) -> MctsDatasetManifest:
        return cls(
            format_version=int(payload.get("format_version", MANIFEST_VERSION)),
            output_dir=str(payload.get("output_dir", "")),
            stages=int(payload.get("stages", 2)),
            scenarios=int(payload.get("scenarios", 10)),
            seed=int(payload.get("seed", 42)),
            games_target=int(payload.get("games_target", 0)),
            shard_rows=int(payload.get("shard_rows", 8192)),
            next_game_id=int(payload.get("next_game_id", 0)),
            next_source_index=int(payload.get("next_source_index", 0)),
            rows=int(payload.get("rows", 0)),
            shard_files=[str(x) for x in payload.get("shard_files", [])],
            complete=bool(payload.get("complete", False)),
            interrupted=bool(payload.get("interrupted", False)),
        )


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def _save_manifest_atomic(output_dir: Path, manifest: MctsDatasetManifest) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _manifest_path(output_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _load_manifest(output_dir: Path) -> MctsDatasetManifest:
    path = _manifest_path(output_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return MctsDatasetManifest.from_json_dict(payload)


def _manifest_matches(
    wanted: MctsDatasetManifest,
    existing: MctsDatasetManifest,
) -> bool:
    return (
        wanted.output_dir == existing.output_dir
        and wanted.stages == existing.stages
        and wanted.scenarios == existing.scenarios
        and wanted.seed == existing.seed
        and wanted.games_target == existing.games_target
        and wanted.shard_rows == existing.shard_rows
    )


# --- 2048 engine (face tile values 2, 4, …), copied from the notebook ----------------


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


def _suffix_sequences(stages: int) -> list[tuple[int, ...]]:
    if stages < 1:
        raise ValueError("stages must be >= 1")
    if stages == 1:
        return [()]
    import itertools

    return list(itertools.product(range(4), repeat=stages - 1))


def _simulate_action_suffix_stochastic(
    board: np.ndarray,
    *,
    first_action: int,
    suffix: tuple[int, ...],
    n_scenarios: int,
    rng: np.random.Generator,
) -> float:
    totals = np.empty(n_scenarios, dtype=np.float64)
    for idx in range(n_scenarios):
        b = board.copy()
        total = 0.0

        moved, gain = apply_move_face(b, first_action)
        if np.array_equal(moved, b):
            totals[idx] = DEFAULT_INVALID_Q_THRESHOLD
            continue
        total += float(gain)
        b = spawn_tile_face(moved, rng)

        for action in suffix:
            moved, gain = apply_move_face(b, action)
            if np.array_equal(moved, b):
                break
            total += float(gain)
            b = spawn_tile_face(moved, rng)

        totals[idx] = total
    return float(np.mean(totals))


def evaluate_legal_first_action_q(
    board: np.ndarray,
    *,
    n_horizon: int,
    n_scenarios: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, tuple[int, ...], float]:
    suffixes = _suffix_sequences(n_horizon)
    legal_first = legal_action_mask_log2(face_board_to_log2_row(board))
    q = np.full(4, DEFAULT_INVALID_Q_THRESHOLD, dtype=np.float32)
    best_suffix_by_action: dict[int, tuple[int, ...]] = {}

    for action in np.flatnonzero(legal_first):
        best_score = float("-inf")
        best_suffix: tuple[int, ...] = ()
        for suffix in suffixes:
            score = _simulate_action_suffix_stochastic(
                board,
                first_action=int(action),
                suffix=suffix,
                n_scenarios=n_scenarios,
                rng=rng,
            )
            if score > best_score:
                best_score = score
                best_suffix = suffix
        q[int(action)] = np.float32(best_score)
        best_suffix_by_action[int(action)] = best_suffix

    pick_scores = q.astype(np.float64, copy=True)
    pick_scores[~legal_first] = -np.inf
    best_action = int(np.argmax(pick_scores))
    best_suffix = best_suffix_by_action.get(best_action, ())
    best_seq = (best_action, *best_suffix)
    return q, best_action, best_seq, float(q[best_action])


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
    game_id: int,
    n_horizon: int,
    n_scenarios: int,
    rng: np.random.Generator,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[int],
    list[np.ndarray],
    list[int],
    list[int],
]:
    """Return per-transition board/mask/action/q/provenance rows for one game."""
    board = np.zeros((4, 4), dtype=np.int32)
    board = spawn_tile_face(board, rng)
    board = spawn_tile_face(board, rng)

    boards_log2: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    actions: list[int] = []
    teacher_q_rows: list[np.ndarray] = []
    episode_ids: list[int] = []
    move_indexes: list[int] = []
    move_idx = 0

    while True:
        log2 = face_board_to_log2_row(board)
        mask = legal_action_mask_log2(log2)
        if not mask.any():
            break

        q_row, direction, _seq, _ev = evaluate_legal_first_action_q(
            board, n_horizon=n_horizon, n_scenarios=n_scenarios, rng=rng
        )
        if not mask[direction]:
            _LOG.warning(
                "[mcts-dataset] planner picked illegal move %s — dropping remainder of game",
                direction,
            )
            break

        new_board, _gained = apply_move_face(board, direction)
        if np.array_equal(new_board, board):
            break

        boards_log2.append(log2.astype(np.int64, copy=False))
        masks.append(mask.copy())
        actions.append(int(direction))
        teacher_q_rows.append(q_row.astype(np.float32, copy=False))
        episode_ids.append(int(game_id))
        move_indexes.append(move_idx)
        move_idx += 1

        board = spawn_tile_face(new_board, rng)

    return boards_log2, masks, actions, teacher_q_rows, episode_ids, move_indexes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Roll out MCTS-labeled 2048 games and save schema-v2 shard datasets "
            "(manifest + shard_*.npz). Ctrl+C saves progress and exits."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mcts_dataset"),
        help="Output run directory for manifest.json + shard_*.npz labels.",
    )
    p.add_argument(
        "--games",
        type=int,
        default=0,
        help="Stop after this many completed games (0 = run until Ctrl+C).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stages", type=int, default=2, metavar="N")
    p.add_argument("--scenarios", type=int, default=10)
    p.add_argument("--n-stage", type=int, default=None, metavar="N")
    p.add_argument("--n-scenarios", type=int, default=None)
    p.add_argument(
        "--shard-rows",
        type=int,
        default=8192,
        metavar="ROWS",
        help="Rows per shard_*.npz file in the output directory.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _concat_batches(
    parts_boards: list[np.ndarray],
    parts_masks: list[np.ndarray],
    parts_actions: list[np.ndarray],
    parts_q: list[np.ndarray],
    parts_episode_id: list[np.ndarray],
    parts_move_idx: list[np.ndarray],
    parts_source_indexes: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not parts_boards:
        zero_b = np.zeros((0, 4, 4), dtype=np.int64)
        zero_m = np.zeros((0, 4), dtype=np.bool_)
        zero_a = np.zeros(0, dtype=np.int64)
        zero_q = np.zeros((0, 4), dtype=np.float32)
        zero_i = np.zeros(0, dtype=np.int64)
        return zero_b, zero_m, zero_a, zero_q, zero_i, zero_i, zero_i
    return (
        np.concatenate(parts_boards, axis=0),
        np.concatenate(parts_masks, axis=0),
        np.concatenate(parts_actions, axis=0),
        np.concatenate(parts_q, axis=0),
        np.concatenate(parts_episode_id, axis=0),
        np.concatenate(parts_move_idx, axis=0),
        np.concatenate(parts_source_indexes, axis=0),
    )


def _validate_shard_rows(
    *,
    masks: np.ndarray,
    actions: np.ndarray,
    teacher_q: np.ndarray,
    invalid_q_threshold: float = DEFAULT_INVALID_Q_THRESHOLD,
) -> dict[str, int]:
    counters = {
        "rows": int(masks.shape[0]),
        "zero_legal_q_rows": 0,
        "nonfinite_rows": 0,
        "argmax_mismatch_rows": 0,
        "illegal_q_rows": 0,
    }
    rows = int(masks.shape[0])
    if rows == 0:
        return counters

    for row in range(rows):
        legal = masks[row]
        q = teacher_q[row]
        action = int(actions[row])

        if not legal.any():
            raise ValueError(f"row {row}: missing legal actions")
        if action < 0 or action >= 4:
            raise ValueError(f"row {row}: teacher action out of range")

        illegal = ~legal
        if not np.allclose(
            q[illegal],
            float(invalid_q_threshold),
            rtol=0.0,
            atol=1.0e-6,
        ):
            counters["illegal_q_rows"] += 1
            raise ValueError(
                f"row {row}: illegal actions must use invalid threshold == {invalid_q_threshold}"
            )
        legal_q = q[legal]
        if not np.isfinite(legal_q).all():
            counters["nonfinite_rows"] += 1
            raise ValueError(f"row {row}: non-finite legal teacher_q values")
        if np.all(legal_q == 0.0):
            counters["zero_legal_q_rows"] += 1

        pick_scores = q.astype(np.float64, copy=True)
        pick_scores[illegal] = -np.inf
        expected = int(np.argmax(pick_scores))
        if action != expected:
            counters["argmax_mismatch_rows"] += 1
            raise ValueError(
                f"row {row}: teacher action {action} != masked argmax {expected}"
            )

    if counters["rows"] > 0 and counters["zero_legal_q_rows"] == counters["rows"]:
        raise ValueError(
            "shard rejected: all rows have all legal teacher_q values equal to zero"
        )
    return counters


def _write_shard(
    *,
    output_dir: Path,
    manifest: MctsDatasetManifest,
    boards: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    teacher_q: np.ndarray,
    episode_id: np.ndarray,
    move_idx: np.ndarray,
    source_indexes: np.ndarray,
) -> None:
    rows = int(boards.shape[0])
    if rows == 0:
        return
    counters = _validate_shard_rows(
        masks=masks,
        actions=actions,
        teacher_q=teacher_q,
    )
    _LOG.info(
        "[mcts-dataset][validate] rows=%s zero_legal_q_rows=%s nonfinite_rows=%s argmax_mismatch_rows=%s illegal_q_rows=%s",
        counters["rows"],
        counters["zero_legal_q_rows"],
        counters["nonfinite_rows"],
        counters["argmax_mismatch_rows"],
        counters["illegal_q_rows"],
    )
    shard_name = f"shard_{len(manifest.shard_files) + 1:06d}.npz"
    shard_path = output_dir / shard_name
    save_labels_npz_v2(
        path=shard_path,
        boards=boards,
        action_masks=masks,
        teacher_actions=actions,
        teacher_q=teacher_q,
        source_indexes=source_indexes,
        stages=manifest.stages,
        scenarios=manifest.scenarios,
        seed=manifest.seed,
        dataset_path=f"mcts_dataset|stages={manifest.stages}|scenarios={manifest.scenarios}",
        episode_id=episode_id,
        move_idx=move_idx,
    )
    manifest.shard_files.append(shard_name)
    manifest.rows += rows
    manifest.next_source_index += rows
    _save_manifest_atomic(output_dir, manifest)
    _LOG.info(
        "[mcts-dataset] wrote %s rows -> %s (rows_total=%s shards=%s)",
        rows,
        shard_path,
        manifest.rows,
        len(manifest.shard_files),
    )


def _flush_shard_buffer(
    *,
    output_dir: Path,
    manifest: MctsDatasetManifest,
    parts_boards: list[np.ndarray],
    parts_masks: list[np.ndarray],
    parts_actions: list[np.ndarray],
    parts_q: list[np.ndarray],
    parts_episode_id: list[np.ndarray],
    parts_move_idx: list[np.ndarray],
    parts_source_indexes: list[np.ndarray],
    max_rows: int,
) -> int:
    boards, masks, actions, teacher_q, episode_id, move_idx, source_indexes = _concat_batches(
        parts_boards,
        parts_masks,
        parts_actions,
        parts_q,
        parts_episode_id,
        parts_move_idx,
        parts_source_indexes,
    )
    rows = int(boards.shape[0])
    if rows == 0:
        return 0

    if max_rows <= 0 or rows <= max_rows:
        _write_shard(
            output_dir=output_dir,
            manifest=manifest,
            boards=boards,
            masks=masks,
            actions=actions,
            teacher_q=teacher_q,
            episode_id=episode_id,
            move_idx=move_idx,
            source_indexes=source_indexes,
        )
        parts_boards.clear()
        parts_masks.clear()
        parts_actions.clear()
        parts_q.clear()
        parts_episode_id.clear()
        parts_move_idx.clear()
        parts_source_indexes.clear()
        return rows

    _write_shard(
        output_dir=output_dir,
        manifest=manifest,
        boards=boards[:max_rows],
        masks=masks[:max_rows],
        actions=actions[:max_rows],
        teacher_q=teacher_q[:max_rows],
        episode_id=episode_id[:max_rows],
        move_idx=move_idx[:max_rows],
        source_indexes=source_indexes[:max_rows],
    )
    parts_boards[:] = [boards[max_rows:]]
    parts_masks[:] = [masks[max_rows:]]
    parts_actions[:] = [actions[max_rows:]]
    parts_q[:] = [teacher_q[max_rows:]]
    parts_episode_id[:] = [episode_id[max_rows:]]
    parts_move_idx[:] = [move_idx[max_rows:]]
    parts_source_indexes[:] = [source_indexes[max_rows:]]
    return max_rows


def _run_benchmark(
    *,
    stages: int,
    scenarios: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    board = np.zeros((4, 4), dtype=np.int32)
    board = spawn_tile_face(board, rng)
    board = spawn_tile_face(board, rng)
    loops = 12
    start = time.perf_counter()
    for _ in range(loops):
        evaluate_legal_first_action_q(
            board,
            n_horizon=stages,
            n_scenarios=scenarios,
            rng=rng,
        )
    elapsed = max(time.perf_counter() - start, 1.0e-9)
    _LOG.info(
        "[mcts-dataset][bench] loops=%s stages=%s scenarios=%s planner_calls_per_sec=%.2f",
        loops,
        stages,
        scenarios,
        float(loops / elapsed),
    )


def main() -> None:
    args = parse_args()
    _configure_logging(verbose=bool(args.verbose))
    _reset_shutdown()
    prev_int = signal.signal(signal.SIGINT, _request_shutdown)
    prev_term = signal.signal(signal.SIGTERM, _request_shutdown)

    stages = int(args.n_stage) if args.n_stage is not None else int(args.stages)
    scenarios = int(args.n_scenarios) if args.n_scenarios is not None else int(args.scenarios)
    if stages < 1:
        raise SystemExit("--stages must be >= 1")
    if scenarios < 1:
        raise SystemExit("--scenarios must be >= 1")
    if int(args.shard_rows) < 1:
        raise SystemExit("--shard-rows must be >= 1")

    if args.n_stage is not None or args.n_scenarios is not None:
        _LOG.warning(
            "[mcts-dataset] --n-stage/--n-scenarios are deprecated; prefer --stages/--scenarios"
        )

    output_dir = args.output_dir.expanduser().resolve()
    desired_manifest = MctsDatasetManifest(
        output_dir=str(output_dir),
        stages=stages,
        scenarios=scenarios,
        seed=int(args.seed),
        games_target=int(args.games),
        shard_rows=int(args.shard_rows),
    )

    try:
        if args.resume:
            path = _manifest_path(output_dir)
            if not path.is_file():
                raise SystemExit(f"--resume requires existing manifest at {path}")
            manifest = _load_manifest(output_dir)
            if not args.force and not _manifest_matches(desired_manifest, manifest):
                raise SystemExit(
                    "Existing manifest does not match options. Use --force to override."
                )
            if manifest.complete:
                _LOG.info("[mcts-dataset] manifest is complete; nothing to do")
                return
            manifest.interrupted = False
            _save_manifest_atomic(output_dir, manifest)
        else:
            if _manifest_path(output_dir).exists() and not args.force:
                raise SystemExit(
                    f"{_manifest_path(output_dir)} already exists; use --resume or --force"
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = desired_manifest
            _save_manifest_atomic(output_dir, manifest)

        rng = np.random.default_rng(manifest.seed)

        if args.benchmark:
            _run_benchmark(stages=manifest.stages, scenarios=manifest.scenarios, seed=manifest.seed)

        batches_b: list[np.ndarray] = []
        batches_m: list[np.ndarray] = []
        batches_a: list[np.ndarray] = []
        batches_q: list[np.ndarray] = []
        batches_eid: list[np.ndarray] = []
        batches_mid: list[np.ndarray] = []
        batches_src: list[np.ndarray] = []

        planner_seconds = 0.0
        t0 = time.perf_counter()

        while True:
            if shutdown_requested():
                break
            if manifest.games_target > 0 and manifest.next_game_id >= manifest.games_target:
                break

            planner_start = time.perf_counter()
            bl, ml, al, ql, eids, mids = collect_one_game_transitions(
                game_id=manifest.next_game_id,
                n_horizon=manifest.stages,
                n_scenarios=manifest.scenarios,
                rng=rng,
            )
            planner_seconds += time.perf_counter() - planner_start

            manifest.next_game_id += 1
            _save_manifest_atomic(output_dir, manifest)

            if bl:
                n = len(bl)
                current_buffer_rows = sum(x.shape[0] for x in batches_b)
                src = np.arange(
                    manifest.next_source_index + current_buffer_rows,
                    manifest.next_source_index + current_buffer_rows + n,
                    dtype=np.int64,
                )
                batches_b.append(np.stack(bl, axis=0))
                batches_m.append(np.stack(ml, axis=0))
                batches_a.append(np.asarray(al, dtype=np.int64))
                batches_q.append(np.stack(ql, axis=0))
                batches_eid.append(np.asarray(eids, dtype=np.int64))
                batches_mid.append(np.asarray(mids, dtype=np.int64))
                batches_src.append(src)

            while sum(x.shape[0] for x in batches_b) >= manifest.shard_rows:
                _flush_shard_buffer(
                    output_dir=output_dir,
                    manifest=manifest,
                    parts_boards=batches_b,
                    parts_masks=batches_m,
                    parts_actions=batches_a,
                    parts_q=batches_q,
                    parts_episode_id=batches_eid,
                    parts_move_idx=batches_mid,
                    parts_source_indexes=batches_src,
                    max_rows=manifest.shard_rows,
                )

            if manifest.next_game_id % 5 == 0 or shutdown_requested():
                elapsed = max(time.perf_counter() - t0, 1.0e-9)
                rows_total = manifest.rows + sum(x.shape[0] for x in batches_b)
                rows_per_sec = rows_total / elapsed
                games_per_sec = manifest.next_game_id / elapsed
                avg_moves = rows_total / max(manifest.next_game_id, 1)
                planner_ms = 1000.0 * planner_seconds / max(manifest.next_game_id, 1)
                _LOG.info(
                    "[mcts-dataset] games=%s rows=%s elapsed=%.1fs rows/s=%.1f games/s=%.2f avg_moves=%.2f shard=%s stages=%s scenarios=%s planner_ms/game=%.1f",
                    manifest.next_game_id,
                    rows_total,
                    elapsed,
                    rows_per_sec,
                    games_per_sec,
                    avg_moves,
                    len(manifest.shard_files) + 1,
                    manifest.stages,
                    manifest.scenarios,
                    planner_ms,
                )

        _flush_shard_buffer(
            output_dir=output_dir,
            manifest=manifest,
            parts_boards=batches_b,
            parts_masks=batches_m,
            parts_actions=batches_a,
            parts_q=batches_q,
            parts_episode_id=batches_eid,
            parts_move_idx=batches_mid,
            parts_source_indexes=batches_src,
            max_rows=0,
        )

        manifest.complete = (manifest.games_target > 0) and (
            manifest.next_game_id >= manifest.games_target
        )
        manifest.interrupted = shutdown_requested()
        _save_manifest_atomic(output_dir, manifest)

        _LOG.info(
            "[mcts-dataset] done complete=%s interrupted=%s games=%s rows=%s shards=%s dir=%s",
            manifest.complete,
            manifest.interrupted,
            manifest.next_game_id,
            manifest.rows,
            len(manifest.shard_files),
            output_dir,
        )
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        _reset_shutdown()


def load_mcts_dataset_shards(run_dir: Path) -> dict[str, object]:
    manifest = _load_manifest(run_dir)
    if not manifest.shard_files:
        raise ValueError(f"No shards listed in {_manifest_path(run_dir)}")
    payloads = [load_labels_npz_any(run_dir / name) for name in manifest.shard_files]
    keys = [
        "boards",
        "action_masks",
        "teacher_actions",
        "teacher_q",
        "teacher_policy",
        "board_hash",
        "episode_id",
        "move_idx",
    ]
    out: dict[str, object] = {}
    for key in keys:
        parts = [p[key] for p in payloads]  # type: ignore[index]
        out[key] = np.concatenate(parts, axis=0)  # type: ignore[arg-type]
    out["source_indexes"] = np.concatenate(
        [p["source_indexes"] for p in payloads], axis=0  # type: ignore[index]
    )
    out["stages"] = manifest.stages
    out["scenarios"] = manifest.scenarios
    out["seed"] = manifest.seed
    out["rows"] = manifest.rows
    return out


if __name__ == "__main__":
    main()
