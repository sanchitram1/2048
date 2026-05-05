from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch import nn

from game2048.game import GameLogic
from training.config import TrainConfig, train_config_from_dict
from training.dqn import build_value_network, mask_illegal_actions
from training.env import Game2048Env
from training.planning import choose_n_step_mc
from training.train import resolve_device, seed_everything

_LOG = logging.getLogger("game2048.imitation")

ACTION_DIM = Game2048Env.action_space_n()

MANIFEST_VERSION = 1
_shutdown_requested = False


def _set_shutdown_flag(signum: int, frame: object | None) -> None:
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name
    _LOG.warning(
        "[imitation] received %s — will stop after the current chunk finishes "
        "(use --resume to continue)",
        sig_name,
    )


def _reset_shutdown_flag() -> None:
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = False


def shutdown_requested() -> bool:
    return bool(_shutdown_requested)


@dataclass
class ShardLabelManifest:
    """Persisted next to shard_*.npz for resume and provenance."""

    format_version: int = MANIFEST_VERSION
    dataset_path: str = ""
    dataset_encoding: str = "log2"
    stages: int = 3
    scenarios: int = 10
    seed: int = 7
    chunk_rows: int = 4096
    shard_usable_rows: int = 8192
    next_raw_row: int = 0
    global_usable_labeled: int = 0
    shard_files: list[str] = field(default_factory=list)
    complete: bool = False
    interrupted: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "dataset_path": self.dataset_path,
            "dataset_encoding": self.dataset_encoding,
            "stages": self.stages,
            "scenarios": self.scenarios,
            "seed": self.seed,
            "chunk_rows": self.chunk_rows,
            "shard_usable_rows": self.shard_usable_rows,
            "next_raw_row": self.next_raw_row,
            "global_usable_labeled": self.global_usable_labeled,
            "shard_files": list(self.shard_files),
            "complete": self.complete,
            "interrupted": self.interrupted,
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> ShardLabelManifest:
        return cls(
            format_version=int(d.get("format_version", MANIFEST_VERSION)),
            dataset_path=str(d["dataset_path"]),
            dataset_encoding=str(d.get("dataset_encoding", "log2")),
            stages=int(d["stages"]),
            scenarios=int(d["scenarios"]),
            seed=int(d["seed"]),
            chunk_rows=int(d["chunk_rows"]),
            shard_usable_rows=int(d["shard_usable_rows"]),
            next_raw_row=int(d["next_raw_row"]),
            global_usable_labeled=int(d["global_usable_labeled"]),
            shard_files=list(d.get("shard_files", [])),
            complete=bool(d.get("complete", False)),
            interrupted=bool(d.get("interrupted", False)),
        )


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "manifest.json"


def load_shard_manifest(run_dir: Path) -> ShardLabelManifest:
    mp = manifest_path(run_dir)
    if not mp.is_file():
        raise FileNotFoundError(f"Shard manifest not found: {mp}")
    data = json.loads(mp.read_text(encoding="utf-8"))
    return ShardLabelManifest.from_json_dict(data)


def save_shard_manifest_atomic(run_dir: Path, manifest: ShardLabelManifest) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    mp = manifest_path(run_dir)
    tmp = mp.with_suffix(".json.tmp")
    payload = json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True)
    tmp.write_text(payload + "\n", encoding="utf-8")
    os.replace(tmp, mp)


def _teacher_config_matches(a: ShardLabelManifest, b: ShardLabelManifest) -> bool:
    return (
        a.dataset_path == b.dataset_path
        and a.dataset_encoding == b.dataset_encoding
        and a.stages == b.stages
        and a.scenarios == b.scenarios
        and a.seed == b.seed
        and a.chunk_rows == b.chunk_rows
        and a.shard_usable_rows == b.shard_usable_rows
    )


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.partial{path.suffix}")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class _ShardBuffer:
    """Concatenate labeled chunks until a shard-sized flush."""

    def __init__(self) -> None:
        self._boards: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._tq: list[np.ndarray] = []
        self._src: list[np.ndarray] = []

    def __len__(self) -> int:
        return sum(b.shape[0] for b in self._boards)

    def extend(
        self,
        boards: np.ndarray,
        masks: np.ndarray,
        actions: np.ndarray,
        tq: np.ndarray,
        src: np.ndarray,
    ) -> None:
        if boards.shape[0] == 0:
            return
        self._boards.append(boards)
        self._masks.append(masks)
        self._actions.append(actions)
        self._tq.append(tq)
        self._src.append(src)

    def _merged(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        def cat(xs: list[np.ndarray]) -> np.ndarray:
            if not xs:
                msg = "empty buffer"
                raise RuntimeError(msg)
            return xs[0] if len(xs) == 1 else np.concatenate(xs, axis=0)

        return (
            cat(self._boards),
            cat(self._masks),
            cat(self._actions),
            cat(self._tq),
            cat(self._src),
        )

    def clear(self) -> None:
        self._boards.clear()
        self._masks.clear()
        self._actions.clear()
        self._tq.clear()
        self._src.clear()

    def take_first_rows(
        self, n: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if n <= 0:
            msg = "n must be positive"
            raise ValueError(msg)
        boards, masks, actions, tq, src = self._merged()
        if boards.shape[0] < n:
            msg = "buffer smaller than take_first_rows"
            raise RuntimeError(msg)
        head = (boards[:n], masks[:n], actions[:n], tq[:n], src[:n])
        tail_b = boards[n:]
        if tail_b.shape[0] == 0:
            self.clear()
        else:
            self._boards = [tail_b]
            self._masks = [masks[n:]]
            self._actions = [actions[n:]]
            self._tq = [tq[n:]]
            self._src = [src[n:]]
        return head

    def take_all(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        if len(self) == 0:
            return None
        out = self._merged()
        self.clear()
        return out


def _format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h{m:02d}m"


BoardDatasetEncoding = Literal["log2", "face"]


def boards_face_values_to_log2(boards: np.ndarray) -> np.ndarray:
    """Convert cells from human tile values (0, 2, 4, 8, …) to log2 exponents."""
    b = np.asarray(boards, dtype=np.int64)
    if b.ndim != 3 or b.shape[1:] != (4, 4):
        msg = f"Expected board array shaped (N, 4, 4); got shape {b.shape}"
        raise ValueError(msg)
    if np.any(b < 0):
        raise ValueError("Face-value encoding: cells must be non-negative")
    mask = b != 0
    if not mask.any():
        return np.zeros_like(b, dtype=np.int64)
    vals = b[mask]
    if np.any(vals < 2):
        raise ValueError(
            "Face-value encoding: non-zero cells must be tile values ≥ 2 (powers of two)."
        )
    if not np.all((vals & (vals - 1)) == 0):
        raise ValueError(
            "Face-value encoding requires powers of 2 in non-zero cells "
            "(got a non-power-of-two tile)."
        )
    out = np.zeros_like(b, dtype=np.int64)
    out[mask] = np.log2(vals).astype(np.int64)
    if not np.all(2 ** out[mask] == vals):
        raise ValueError("Face-value → log2 conversion failed (check tile values).")
    return out


def load_board_dataset(
    path: str | Path, *, encoding: BoardDatasetEncoding = "log2"
) -> np.ndarray:
    """Load corpus shaped (N, 4, 4).

    - ``log2`` (default): cells match ``GameLogic`` / env encoding (0 empty, 1→2, 2→4, …).
    - ``face``: cells hold displayed tile values (0, 2, 4, 8, …); converted to log2 internally.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Board dataset not found: {resolved}")
    boards = np.load(resolved, allow_pickle=False)
    if boards.ndim != 3 or boards.shape[1:] != (4, 4):
        msg = f"Expected board array shaped (N, 4, 4); got shape {boards.shape}"
        raise ValueError(msg)
    if not np.issubdtype(boards.dtype, np.integer):
        raise ValueError(f"Expected integer dtype for boards; got {boards.dtype}")
    arr = np.asarray(boards, dtype=np.int64)
    if encoding == "face":
        _LOG.info(
            "[imitation] dataset encoding: face values → log2 exponents (GameLogic format)"
        )
        return boards_face_values_to_log2(arr)
    return arr


def game_from_board(grid: np.ndarray) -> GameLogic:
    """Rebuild rules state so legal moves reflect the frozen board snapshot."""
    logic = GameLogic()
    logical = np.asarray(grid, dtype=np.int16)
    if logical.shape != (logic.grid_size, logic.grid_size):
        raise ValueError(f"Board must be 4x4; got {logical.shape}")
    logic.grid = logical.astype(np.int16, copy=True)
    logic.score = 0
    logic.done = False
    return logic


def filter_usable_boards(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep boards with ≥1 slide that changes the grid."""
    usable: list[np.ndarray] = []
    row_idx: list[int] = []
    for i in range(boards.shape[0]):
        game = game_from_board(boards[i])
        if game.available_moves():
            usable.append(boards[i])
            row_idx.append(i)
    if not usable:
        n_raw = int(boards.shape[0])
        _LOG.warning(
            "[imitation] usable boards: 0 / %d (all rows dropped — no legal changing move)",
            n_raw,
        )
        return boards[:0].copy(), np.zeros(0, dtype=np.int64)
    stacked = np.stack(usable, axis=0).astype(np.int64, copy=False)
    indexes = np.asarray(row_idx, dtype=np.int64)
    n_raw = int(boards.shape[0])
    n_ok = int(stacked.shape[0])
    _LOG.info(
        "[imitation] usable boards: %d / %d (dropped %d with no legal changing move)",
        n_ok,
        n_raw,
        n_raw - n_ok,
    )
    return stacked, indexes


def label_board_states(
    boards: np.ndarray,
    *,
    stages: int,
    scenarios: int,
    seed: int,
    max_boards: int | None,
    log_every: int = 250,
    usable_rng_offset: int = 0,
    usable_prefiltered: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ``choose_n_step_mc`` teacher on usable boards.

    RNG for row ``row`` uses ``seed + usable_rng_offset + row`` so chunked labeling
    matches a single-pass run when chunks are processed in dataset order.

    When ``usable_prefiltered`` is set, it must be ``(usable_boards, source_indexes)``
    relative to the same logical slice as ``boards``; caller caps row count (e.g.
    ``max_boards`` in sharded mode). ``max_boards`` is ignored in that case.
    """
    if usable_prefiltered is not None:
        usable, src_indexes = usable_prefiltered
        usable = np.asarray(usable, dtype=np.int64, copy=False)
        src_indexes = np.asarray(src_indexes, dtype=np.int64, copy=False)
    else:
        usable, src_indexes = filter_usable_boards(boards)
        if max_boards is not None:
            cap = max(0, int(max_boards))
            usable = usable[:cap]
            src_indexes = src_indexes[: usable.shape[0]]
            _LOG.info(
                "[imitation] capped to --max-boards=%s → labeling %d rows",
                cap,
                usable.shape[0],
            )

    action_dim = ACTION_DIM
    n = usable.shape[0]
    _LOG.info(
        "[imitation] teacher labeling: N=%d Monte-Carlo boards, stages=%d scenarios=%d "
        "(each row runs N-step MC teacher on one frozen board snapshot)",
        n,
        stages,
        scenarios,
    )
    masks = np.zeros((n, action_dim), dtype=np.bool_)
    teacher_actions = np.zeros(n, dtype=np.int64)
    teacher_q = np.zeros((n, action_dim), dtype=np.float32)

    t0 = time.perf_counter()
    for row in range(n):
        game = game_from_board(usable[row])
        planned = choose_n_step_mc(
            game,
            stages=stages,
            scenarios=scenarios,
            rng=random.Random(seed + int(usable_rng_offset) + row),
        )
        for action_id in planned.legal_actions:
            masks[row, action_id] = True
        ta = planned.action
        if not masks[row, ta]:
            raise RuntimeError("Teacher chose illegal action")
        teacher_actions[row] = ta
        teacher_q[row] = np.array(planned.q_values, dtype=np.float32)

        if log_every > 0 and n > 0:
            done = row + 1
            if done % log_every == 0 or done == n:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (n - done) / rate if rate > 0 else float("nan")
                _LOG.info(
                    "[imitation] labeled %d/%d (%.2f boards/s, ETA %s)",
                    done,
                    n,
                    rate,
                    _format_eta(remaining),
                )

    return usable, masks, teacher_actions, teacher_q, src_indexes


def save_labels_npz(
    *,
    path: Path,
    boards: np.ndarray,
    action_masks: np.ndarray,
    teacher_actions: np.ndarray,
    teacher_q: np.ndarray,
    source_indexes: np.ndarray | None,
    stages: int,
    scenarios: int,
    seed: int,
    dataset_path: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "boards": boards,
        "action_masks": action_masks,
        "teacher_actions": teacher_actions,
        "teacher_q": teacher_q,
        "source_indexes": source_indexes.astype(np.int64, copy=False)
        if source_indexes is not None
        else np.zeros(0, dtype=np.int64),
        "stages": np.array([stages], dtype=np.int64),
        "scenarios": np.array([scenarios], dtype=np.int64),
        "seed": np.array([seed], dtype=np.int64),
        "dataset_path": np.array(dataset_path, dtype=np.str_),
    }
    _write_npz_atomic(path, arrays)


def load_labels_npz(path: str | Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Labels file not found: {resolved}")
    data = np.load(resolved, allow_pickle=False)
    boards = data["boards"].astype(np.int64, copy=False)
    masks = data["action_masks"]
    actions = data["teacher_actions"].astype(np.int64, copy=False)
    tq = data["teacher_q"].astype(np.float32, copy=False)
    if masks.dtype != np.bool_:
        masks = masks.astype(np.bool_, copy=False)
    src_raw = data.get("source_indexes")
    unique_src_index = isinstance(src_raw, np.ndarray) and src_raw.ndim == 1
    idx_list = np.asarray(src_raw, dtype=np.int64) if unique_src_index else None
    if idx_list is not None and idx_list.size == 0:
        idx_list = None
    return {
        "boards": boards,
        "action_masks": masks,
        "teacher_actions": actions,
        "teacher_q": tq,
        "source_indexes": idx_list,
        "stages": int(data["stages"][0]) if "stages" in data else None,
        "scenarios": int(data["scenarios"][0]) if "scenarios" in data else None,
        "seed": int(data["seed"][0]) if "seed" in data else None,
    }


def load_board_chunk_copy(
    mmap: np.ndarray,
    start: int,
    end: int,
    *,
    encoding: BoardDatasetEncoding,
) -> np.ndarray:
    """Copy ``mmap[start:end]`` into RAM and apply dataset encoding."""
    slab = np.asarray(mmap[start:end], dtype=np.int64)
    if encoding == "face":
        _LOG.debug(
            "[imitation] chunk [%s:%s] face→log2 (materialized from mmap)", start, end
        )
        return boards_face_values_to_log2(slab)
    return slab


def load_labels_merged(run_dir: str | Path) -> dict[str, object]:
    """Load all ``shard_*.npz`` listed in ``manifest.json`` and concatenate."""
    rd = Path(run_dir).expanduser().resolve()
    manifest = load_shard_manifest(rd)
    if not manifest.shard_files:
        raise ValueError(f"No shards recorded in manifest under {rd}")

    boards_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    tq_list: list[np.ndarray] = []
    src_list: list[np.ndarray] = []

    for name in manifest.shard_files:
        sp = rd / name
        if not sp.is_file():
            msg = f"Manifest lists missing shard {sp}"
            raise FileNotFoundError(msg)
        payload = load_labels_npz(sp)
        boards_list.append(payload["boards"])  # type: ignore[arg-type]
        masks_list.append(payload["action_masks"])  # type: ignore[arg-type]
        actions_list.append(payload["teacher_actions"])  # type: ignore[arg-type]
        tq_list.append(payload["teacher_q"])  # type: ignore[arg-type]
        si = payload["source_indexes"]
        if si is None:
            msg = f"Shard {sp} missing source_indexes"
            raise ValueError(msg)
        src_list.append(si)

    boards = np.concatenate(boards_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    tq = np.concatenate(tq_list, axis=0)
    src = np.concatenate(src_list, axis=0)

    return {
        "boards": boards.astype(np.int64, copy=False),
        "action_masks": masks.astype(np.bool_, copy=False),
        "teacher_actions": actions.astype(np.int64, copy=False),
        "teacher_q": tq.astype(np.float32, copy=False),
        "source_indexes": src.astype(np.int64, copy=False),
        "stages": manifest.stages,
        "scenarios": manifest.scenarios,
        "seed": manifest.seed,
        "dataset_path_manifest": manifest.dataset_path,
    }


def load_labels_for_training(path: str | Path) -> dict[str, object]:
    """Single ``.npz`` or a shard run directory with ``manifest.json``."""
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        return load_labels_merged(resolved)
    return load_labels_npz(resolved)


def run_sharded_labeling(
    *,
    dataset_path: Path,
    dataset_encoding: BoardDatasetEncoding,
    run_dir: Path,
    stages: int,
    scenarios: int,
    seed: int,
    chunk_rows: int,
    shard_usable_rows: int,
    resume: bool,
    force: bool,
    max_boards: int | None,
    log_every: int,
) -> tuple[bool, ShardLabelManifest]:
    """Chunked labeling with manifest + shard npz files.

    Returns ``(clean_finish, manifest)``. ``clean_finish`` is False if SIGINT/SIGTERM
    arrived between chunks (after the current chunk finished).

    Resume advances along **raw dataset indices**; when ``--max-boards`` stops
    mid-chunk, the manifest cursor rewinds to the next unlabeled usable row so
    ``--resume`` cannot skip positions.
    """
    if chunk_rows < 1 or shard_usable_rows < 1:
        raise ValueError("chunk_rows and shard_usable_rows must be >= 1")

    resolved_ds = dataset_path.expanduser().resolve()
    if not resolved_ds.is_file():
        raise FileNotFoundError(f"Board dataset not found: {resolved_ds}")

    rd = run_dir.expanduser().resolve()
    mp = manifest_path(rd)

    desired_meta = ShardLabelManifest(
        format_version=MANIFEST_VERSION,
        dataset_path=str(resolved_ds),
        dataset_encoding=dataset_encoding,
        stages=stages,
        scenarios=scenarios,
        seed=seed,
        chunk_rows=chunk_rows,
        shard_usable_rows=shard_usable_rows,
        next_raw_row=0,
        global_usable_labeled=0,
        shard_files=[],
        complete=False,
        interrupted=False,
    )

    if resume:
        if not mp.is_file():
            raise SystemExit(f"--resume requires existing manifest at {mp}")
        manifest = load_shard_manifest(rd)
        if not force and not _teacher_config_matches(desired_meta, manifest):
            raise SystemExit(
                "Existing manifest does not match teacher/layout settings "
                "(dataset path/encoding, stages, scenarios, seed, chunk/shard sizes). "
                "Align CLI flags or pass --force (unsafe)."
            )
        if manifest.complete:
            _LOG.info("[imitation] shard run already complete — nothing to label")
            return True, manifest
        _LOG.info(
            "[imitation] resume shard labeling | next_raw_row=%s usable_so_far=%s shards=%s",
            manifest.next_raw_row,
            manifest.global_usable_labeled,
            len(manifest.shard_files),
        )
    else:
        if mp.is_file():
            raise SystemExit(
                f"{mp} already exists — choose a new --labels-run-dir or pass --resume"
            )
        manifest = desired_meta
        rd.mkdir(parents=True, exist_ok=True)
        save_shard_manifest_atomic(rd, manifest)

    mmap_boards = np.load(resolved_ds, mmap_mode="r")
    if mmap_boards.ndim != 3 or mmap_boards.shape[1:] != (4, 4):
        msg = f"Expected board array shaped (N, 4, 4); got shape {mmap_boards.shape}"
        raise ValueError(msg)
    if not np.issubdtype(mmap_boards.dtype, np.integer):
        raise ValueError(f"Expected integer dtype for boards; got {mmap_boards.dtype}")

    total_rows = int(mmap_boards.shape[0])
    buffer = _ShardBuffer()
    global_usable = int(manifest.global_usable_labeled)
    row_cursor = int(manifest.next_raw_row)
    clean_finish = True

    prev_sigint = signal.signal(signal.SIGINT, _set_shutdown_flag)
    prev_sigterm = signal.signal(signal.SIGTERM, _set_shutdown_flag)
    _reset_shutdown_flag()

    def write_shard_rows(
        boards: np.ndarray,
        masks: np.ndarray,
        tac: np.ndarray,
        tq_b: np.ndarray,
        src: np.ndarray,
    ) -> None:
        idx = len(manifest.shard_files) + 1
        name = f"shard_{idx:06d}.npz"
        shard_p = rd / name
        save_labels_npz(
            path=shard_p,
            boards=boards,
            action_masks=masks,
            teacher_actions=tac,
            teacher_q=tq_b,
            source_indexes=src,
            stages=stages,
            scenarios=scenarios,
            seed=seed,
            dataset_path=str(resolved_ds),
        )
        manifest.shard_files.append(name)
        _LOG.info(
            "[imitation] wrote %s (%d usable rows, total shards=%d)",
            name,
            boards.shape[0],
            len(manifest.shard_files),
        )

    try:
        while row_cursor < total_rows:
            if shutdown_requested():
                clean_finish = False
                break
            if max_boards is not None and global_usable >= max_boards:
                break

            chunk_end = min(row_cursor + chunk_rows, total_rows)
            slab = load_board_chunk_copy(
                mmap_boards,
                row_cursor,
                chunk_end,
                encoding=dataset_encoding,
            )
            remaining_cap: int | None = None
            if max_boards is not None:
                remaining_cap = max(0, int(max_boards) - global_usable)

            usable_full, src_rel_full = filter_usable_boards(slab)
            if usable_full.shape[0] == 0:
                next_raw_boundary = chunk_end
            else:
                take_n = int(usable_full.shape[0])
                if remaining_cap is not None:
                    take_n = min(take_n, int(remaining_cap))

                if take_n <= 0:
                    next_raw_boundary = chunk_end
                else:
                    usable_sub = usable_full[:take_n]
                    src_sub_rel = src_rel_full[:take_n]

                    usable, masks, tac, tq_arr, src_rel = label_board_states(
                        slab,
                        stages=stages,
                        scenarios=scenarios,
                        seed=seed,
                        max_boards=None,
                        log_every=log_every,
                        usable_rng_offset=global_usable,
                        usable_prefiltered=(usable_sub, src_sub_rel),
                    )
                    src_abs = src_rel.astype(np.int64, copy=False) + int(row_cursor)

                    buffer.extend(usable, masks, tac, tq_arr, src_abs)
                    global_usable += int(usable.shape[0])

                    while len(buffer) >= shard_usable_rows:
                        chunk_shard = buffer.take_first_rows(shard_usable_rows)
                        write_shard_rows(*chunk_shard)

                    if take_n < usable_full.shape[0]:
                        next_raw_boundary = int(row_cursor + int(src_rel_full[take_n]))
                    else:
                        next_raw_boundary = chunk_end

            manifest.next_raw_row = next_raw_boundary
            manifest.global_usable_labeled = global_usable
            manifest.interrupted = False
            save_shard_manifest_atomic(rd, manifest)

            row_cursor = next_raw_boundary

            if shutdown_requested():
                clean_finish = False
                break

        tail = buffer.take_all()
        if tail is not None:
            write_shard_rows(*tail)

        manifest.global_usable_labeled = global_usable
        manifest.next_raw_row = row_cursor
        manifest.complete = row_cursor >= total_rows
        manifest.interrupted = not clean_finish
        save_shard_manifest_atomic(rd, manifest)

    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        _reset_shutdown_flag()

    _LOG.info(
        "[imitation] shard labeling finished | clean=%s complete=%s usable=%d shards=%d next_raw_row=%d/%d",
        clean_finish,
        manifest.complete,
        manifest.global_usable_labeled,
        len(manifest.shard_files),
        manifest.next_raw_row,
        total_rows,
    )
    return clean_finish, manifest


def _teacher_probs_from_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax restricted to legal actions (row vectors)."""
    masked = torch.where(mask, q, torch.full_like(q, -1.0e9))
    probs = torch.softmax(masked, dim=-1) * mask.float()
    return probs


@dataclass(frozen=True)
class TeacherAgreementMetrics:
    """Student vs frozen teacher on a row subset."""

    n: int
    exact_match_rate: float
    teacher_action_prob_mean: float


def compute_train_val_indices_row_shuffle(
    n: int,
    *,
    val_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """IID row shuffle: ``floor(val_fraction * n)`` rows for validation."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty.copy(), empty.copy()
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n)
    n_val = int(math.floor(float(val_fraction) * n))
    val_idx = np.sort(perm[:n_val].astype(np.int64, copy=False))
    train_idx = np.sort(perm[n_val:].astype(np.int64, copy=False))
    return train_idx, val_idx


def compute_train_val_indices_by_source(
    source_indexes: np.ndarray,
    *,
    val_fraction: float,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split **groups** of rows sharing the same ``source_indexes`` value (no row leakage)."""
    src = np.asarray(source_indexes, dtype=np.int64).reshape(-1)
    n = int(src.shape[0])
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty.copy(), empty.copy()
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    uniques, inverse = np.unique(src, return_inverse=True)
    n_groups = int(uniques.shape[0])
    rng = np.random.default_rng(int(split_seed))
    group_perm = rng.permutation(n_groups)
    n_val_g = int(math.floor(float(val_fraction) * n_groups))
    val_group_ids = set(group_perm[:n_val_g].tolist())
    val_gid_arr = np.array(sorted(val_group_ids), dtype=np.int64)
    is_val_group = np.isin(inverse, val_gid_arr)
    val_idx = np.nonzero(is_val_group)[0].astype(np.int64, copy=False)
    train_idx = np.nonzero(~is_val_group)[0].astype(np.int64, copy=False)
    return train_idx, val_idx


def compute_train_val_split_arrays(
    *,
    n_rows: int,
    source_indexes: np.ndarray | None,
    val_fraction: float,
    split_seed: int,
    split_by_source: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Train index array + validation index array (possibly empty)."""
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative")
    if val_fraction < 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if val_fraction == 0.0:
        all_idx = np.arange(n_rows, dtype=np.int64)
        empty = np.zeros(0, dtype=np.int64)
        return all_idx, empty
    if split_by_source:
        if source_indexes is None:
            raise ValueError("--split-by-source requires source_indexes in the labels artifact")
        si = np.asarray(source_indexes, dtype=np.int64).reshape(-1)
        if si.shape[0] != n_rows:
            msg = "source_indexes length must match number of labeled rows"
            raise ValueError(msg)
        return compute_train_val_indices_by_source(
            si, val_fraction=val_fraction, split_seed=split_seed
        )
    return compute_train_val_indices_row_shuffle(
        n_rows, val_fraction=val_fraction, split_seed=split_seed
    )


def evaluate_teacher_agreement(
    q_network: nn.Module,
    boards: np.ndarray,
    action_masks: np.ndarray,
    teacher_actions: np.ndarray,
    row_indices: np.ndarray | Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
    include_soft_prob: bool = True,
) -> TeacherAgreementMetrics:
    """Masked argmax(student Q) vs ``teacher_actions``; optional mean softmax prob on teacher move."""
    idx = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    n = int(idx.size)
    if n == 0:
        return TeacherAgreementMetrics(
            n=0, exact_match_rate=float("nan"), teacher_action_prob_mean=float("nan")
        )
    bs = max(1, int(batch_size))
    q_network.eval()
    exact = 0
    prob_acc = 0.0
    with torch.no_grad():
        for start in range(0, n, bs):
            sel = idx[start : start + bs]
            states_b = torch.as_tensor(boards[sel], dtype=torch.long, device=device)
            masks_b = torch.as_tensor(action_masks[sel], dtype=torch.bool, device=device)
            ta_b = torch.as_tensor(teacher_actions[sel], dtype=torch.long, device=device)
            logits = q_network(states_b)
            masked = mask_illegal_actions(logits, masks_b)
            pred = masked.argmax(dim=-1)
            exact += int((pred == ta_b).sum().item())
            if include_soft_prob:
                log_p = torch.log_softmax(masked, dim=-1)
                prob_acc += float(
                    log_p.gather(1, ta_b.unsqueeze(1)).squeeze(1).exp().sum().item()
                )
    rate = exact / n if n else float("nan")
    mean_p = prob_acc / n if (n and include_soft_prob) else float("nan")
    return TeacherAgreementMetrics(
        n=n, exact_match_rate=rate, teacher_action_prob_mean=mean_p
    )


def run_teacher_agreement_report(
    *,
    checkpoint: Path,
    labels_path: Path,
    val_fraction: float,
    split_seed: int,
    split_by_source: bool,
    device: torch.device,
    batch_size: int,
    log_train_metrics: bool,
    verbose: bool,
) -> None:
    """Load checkpoint + labels and INFO-log masked-argmax agreement vs teacher_actions."""
    configure_logging(verbose=verbose)
    ck_path = checkpoint.expanduser().resolve()
    if not ck_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")
    payload = load_labels_for_training(labels_path)
    boards = payload["boards"]  # type: ignore[assignment]
    masks = payload["action_masks"]  # type: ignore[assignment]
    tac = payload["teacher_actions"]  # type: ignore[assignment]
    boards = boards.astype(np.int64, copy=False)
    masks = masks.astype(np.bool_, copy=False)
    tac = tac.astype(np.int64, copy=False)
    n = int(boards.shape[0])
    raw_src = payload.get("source_indexes")
    src_arr = (
        None if raw_src is None else np.asarray(raw_src, dtype=np.int64).reshape(-1)
    )

    payload_ck = torch.load(ck_path, map_location=device, weights_only=False)
    ck_cfg = payload_ck.get("config")
    if not isinstance(ck_cfg, dict):
        raise ValueError("Checkpoint missing config dict")
    train_cfg = train_config_from_dict(ck_cfg)

    q_network = build_value_network(
        train_cfg.value_network,
        ACTION_DIM,
        max_exponent=train_cfg.max_exponent,
        embedding_dim=train_cfg.embedding_dim,
        hidden_dim=train_cfg.hidden_dim,
    ).to(device)
    q_sd = payload_ck.get("q_network_state_dict")
    if not isinstance(q_sd, dict):
        raise ValueError("Checkpoint missing q_network_state_dict")
    q_network.load_state_dict(q_sd)

    train_idx, val_idx = compute_train_val_split_arrays(
        n_rows=n,
        source_indexes=src_arr,
        val_fraction=val_fraction,
        split_seed=split_seed,
        split_by_source=split_by_source,
    )

    _LOG.info(
        "[imitation] teacher agreement | checkpoint=%s rows=%d train=%d val=%d "
        "split_seed=%s by_source=%s val_fraction=%s",
        ck_path,
        n,
        train_idx.shape[0],
        val_idx.shape[0],
        split_seed,
        split_by_source,
        val_fraction,
    )

    want_train = bool(log_train_metrics or val_idx.shape[0] == 0)
    if want_train and train_idx.shape[0] > 0:
        tm = evaluate_teacher_agreement(
            q_network,
            boards,
            masks,
            tac,
            train_idx,
            device=device,
            batch_size=batch_size,
        )
        _LOG.info(
            "[imitation] teacher_agreement train exact_match=%.6f mean_p_teacher=%.6f n=%d",
            tm.exact_match_rate,
            tm.teacher_action_prob_mean,
            tm.n,
        )
    if val_idx.shape[0] > 0:
        vm = evaluate_teacher_agreement(
            q_network,
            boards,
            masks,
            tac,
            val_idx,
            device=device,
            batch_size=batch_size,
        )
        _LOG.info(
            "[imitation] teacher_agreement val exact_match=%.6f mean_p_teacher=%.6f n=%d",
            vm.exact_match_rate,
            vm.teacher_action_prob_mean,
            vm.n,
        )


def imitation_loss_batch(
    *,
    logits: torch.Tensor,
    action_masks: torch.Tensor,
    teacher_actions: torch.Tensor,
    teacher_q: torch.Tensor | None,
    soft_target_weight: float,
) -> torch.Tensor:
    if soft_target_weight < 0.0 or soft_target_weight > 1.0:
        raise ValueError("soft_target_weight must be in [0, 1]")

    masked_logits = mask_illegal_actions(logits, action_masks)
    log_probs = torch.log_softmax(masked_logits, dim=-1)

    illegal_target = (~action_masks.gather(1, teacher_actions.unsqueeze(1))).squeeze(1)
    if illegal_target.any():
        raise RuntimeError("Teacher indices must be legal")

    nll_hard = -log_probs.gather(1, teacher_actions.unsqueeze(1)).squeeze(1)

    if soft_target_weight == 0.0 or teacher_q is None:
        return nll_hard.mean()

    tgt = _teacher_probs_from_q(teacher_q, action_masks).clamp(min=1e-8)
    kl = (tgt * (torch.log(tgt) - log_probs)).sum(dim=-1)
    return ((1.0 - soft_target_weight) * nll_hard + soft_target_weight * kl).mean()


class BoardLabelDataset(torch.utils.data.Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        boards: np.ndarray,
        action_masks: np.ndarray,
        teacher_actions: np.ndarray,
        teacher_q: np.ndarray | None,
    ) -> None:
        self.boards = boards
        self.action_masks = action_masks
        self.teacher_actions = teacher_actions
        self.teacher_q = teacher_q

    def __len__(self) -> int:
        return int(self.boards.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        bs = torch.as_tensor(self.boards[index], dtype=torch.long)
        mask = torch.as_tensor(self.action_masks[index], dtype=torch.bool)
        act = torch.as_tensor(self.teacher_actions[index], dtype=torch.long)
        if self.teacher_q is None:
            return bs, mask, act
        tq = torch.as_tensor(self.teacher_q[index], dtype=torch.float32)
        return bs, mask, act, tq


def merge_train_config_with_init(
    base: TrainConfig,
    *,
    init_checkpoint_path: Path | None,
    learning_rate_override: float | None,
    value_network_override: str | None,
) -> TrainConfig:
    merged_dict = asdict(train_config_from_dict(asdict(base)))
    if init_checkpoint_path is not None and init_checkpoint_path.is_file():
        payload = torch.load(
            init_checkpoint_path, map_location="cpu", weights_only=False
        )
        ck_cfg = payload.get("config")
        if isinstance(ck_cfg, dict):
            merged_dict.update(asdict(train_config_from_dict(ck_cfg)))
    replacements: dict[str, object] = {
        "seed": base.seed,
        "model_dir": base.model_dir,
        "device": base.device,
    }
    if learning_rate_override is not None:
        replacements["learning_rate"] = learning_rate_override
    if value_network_override is not None:
        replacements["value_network"] = value_network_override

    merged_dict.update(replacements)
    field_names = tuple(TrainConfig.__dataclass_fields__.keys())
    return TrainConfig(**{name: merged_dict[name] for name in field_names})


def train_imitation(
    *,
    boards: np.ndarray,
    action_masks: np.ndarray,
    teacher_actions: np.ndarray,
    teacher_q: np.ndarray,
    train_cfg: TrainConfig,
    init_checkpoint_path: Path | None,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    device: torch.device,
    soft_target_weight: float,
    save_step: str | int,
    val_boards: np.ndarray | None = None,
    val_masks: np.ndarray | None = None,
    val_teacher_actions: np.ndarray | None = None,
    log_agreement_every_epoch: bool = False,
    log_train_agreement_every_epoch: bool = False,
    agreement_batch_size: int = 512,
) -> Path:
    if boards.shape[0] == 0:
        raise ValueError("No labeled boards to train on")

    tq_np = teacher_q if soft_target_weight > 0.0 else None
    ds = BoardLabelDataset(boards, action_masks, teacher_actions, tq_np)
    sampler = torch.utils.data.RandomSampler(
        ds,
        generator=torch.Generator().manual_seed(train_cfg.seed),
    )
    drop_last = len(ds) >= batch_size and batch_size > 1 and len(ds) % batch_size == 1
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=0,
    )

    action_dim = ACTION_DIM
    q_network = build_value_network(
        train_cfg.value_network,
        action_dim,
        max_exponent=train_cfg.max_exponent,
        embedding_dim=train_cfg.embedding_dim,
        hidden_dim=train_cfg.hidden_dim,
    ).to(device)

    target_network = build_value_network(
        train_cfg.value_network,
        action_dim,
        max_exponent=train_cfg.max_exponent,
        embedding_dim=train_cfg.embedding_dim,
        hidden_dim=train_cfg.hidden_dim,
    ).to(device)

    if init_checkpoint_path is not None and init_checkpoint_path.is_file():
        full_ckpt = torch.load(
            init_checkpoint_path, map_location=device, weights_only=False
        )
        q_sd = full_ckpt.get("q_network_state_dict")
        if isinstance(q_sd, dict):
            q_network.load_state_dict(q_sd)

    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=train_cfg.learning_rate)
    if init_checkpoint_path is not None and init_checkpoint_path.is_file():
        ck = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        opt_sd = ck.get("optimizer_state_dict")
        if isinstance(opt_sd, dict):
            try:
                optimizer.load_state_dict(opt_sd)
            except (RuntimeError, ValueError):
                pass

    q_network.train()
    target_network.eval()

    _LOG.info(
        "[imitation] supervised training: n=%s arch=%s device=%s lr=%s epochs=%s batch=%s "
        "soft_target=%s save_step=%s",
        boards.shape[0],
        train_cfg.value_network,
        device,
        train_cfg.learning_rate,
        epochs,
        batch_size,
        soft_target_weight,
        save_step,
    )
    if init_checkpoint_path is not None:
        _LOG.info("[imitation] init weights from %s", init_checkpoint_path)

    for epoch in range(epochs):
        losses: list[float] = []
        t_epoch = time.perf_counter()
        for batch_idx, batch_tuple in enumerate(loader):
            if len(batch_tuple) == 4:
                states_b, masks_b, ta_b, tq_b = batch_tuple
                tq_blob = tq_b.to(device)
            else:
                states_b, masks_b, ta_b = batch_tuple
                tq_blob = None

            states_b = states_b.to(device)
            masks_b = masks_b.to(device)
            ta_b = ta_b.to(device)

            logits = q_network(states_b)
            loss = imitation_loss_batch(
                logits=logits,
                action_masks=masks_b,
                teacher_actions=ta_b,
                teacher_q=tq_blob,
                soft_target_weight=soft_target_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), train_cfg.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
            if _LOG.isEnabledFor(logging.DEBUG) and (
                batch_idx % 100 == 0 or batch_idx + 1 == len(loader)
            ):
                _LOG.debug(
                    "[imitation] epoch %s batch %s/%s loss=%.5f",
                    epoch + 1,
                    batch_idx + 1,
                    len(loader),
                    losses[-1],
                )

        epoch_mean = float(np.mean(losses)) if losses else 0.0
        epoch_s = time.perf_counter() - t_epoch
        _LOG.info(
            "[imitation] epoch=%s/%s mean_batch_loss=%.4f batches=%s (%.1fs)",
            epoch + 1,
            epochs,
            epoch_mean,
            len(losses),
            epoch_s,
        )

        if (
            log_agreement_every_epoch
            and val_boards is not None
            and val_boards.shape[0] > 0
            and val_masks is not None
            and val_teacher_actions is not None
        ):
            v_idx = np.arange(val_boards.shape[0], dtype=np.int64)
            vm = evaluate_teacher_agreement(
                q_network,
                val_boards,
                val_masks,
                val_teacher_actions,
                v_idx,
                device=device,
                batch_size=agreement_batch_size,
            )
            _LOG.info(
                "[imitation] epoch=%s/%s val_teacher_exact=%.6f val_teacher_mean_p=%.6f n_val=%d",
                epoch + 1,
                epochs,
                vm.exact_match_rate,
                vm.teacher_action_prob_mean,
                vm.n,
            )
        if log_train_agreement_every_epoch:
            t_idx = np.arange(boards.shape[0], dtype=np.int64)
            tm = evaluate_teacher_agreement(
                q_network,
                boards,
                action_masks,
                teacher_actions,
                t_idx,
                device=device,
                batch_size=agreement_batch_size,
            )
            _LOG.info(
                "[imitation] epoch=%s/%s train_teacher_exact=%.6f train_teacher_mean_p=%.6f n_train=%d",
                epoch + 1,
                epochs,
                tm.exact_match_rate,
                tm.teacher_action_prob_mean,
                tm.n,
            )

    ckpt_dir = Path(model_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_{save_step}.pt"
    torch.save(
        {
            "step": save_step,
            "episodes_completed": 0,
            "config": asdict(train_cfg),
            "q_network_state_dict": q_network.cpu().state_dict(),
            "target_network_state_dict": target_network.cpu().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Imitation: (1) load .npy boards, (2) optional face→log2 conversion, "
            "(3) drop stuck boards, (4) label each usable board with N-step MC teacher "
            "into .npz (slow; use --log-every), (5) train QCNN with masked CE / soft KL.\n"
            "Logging: default INFO milestones; -v adds DEBUG batch losses; --log-every N "
            "prints labeling throughput/ETA every N boards."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="NX4x4 boards .npy (required unless --train-only)",
    )
    p.add_argument(
        "--dataset-encoding",
        choices=("log2", "face"),
        default="log2",
        help=(
            "log2: cells are env exponents (0 empty, 1→tile 2, …). "
            "face: cells are displayed tile values (0, 2, 4, 8, …)."
        ),
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("scripts/boards_mcts_labels.npz"),
        metavar="PATH",
        help="Teacher labels artifact (.npz); written when labeling, read when --train-only.",
    )
    p.add_argument(
        "--label-only",
        action="store_true",
        help="Compute labels then exit.",
    )
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Train from labels only (loads --labels: single .npz or shard run directory).",
    )
    p.add_argument(
        "--labels-run-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Write sharded teacher labels here as manifest.json + shard_*.npz "
            "(mmap chunks; Ctrl+C stops between chunks). Resume with --resume. "
            "After a full label+train run without --label-only, training loads merged shards "
            "from this directory automatically."
        ),
    )
    p.add_argument(
        "--chunk-rows",
        type=int,
        default=4096,
        metavar="N",
        help="Raw dataset rows loaded per chunk when using --labels-run-dir.",
    )
    p.add_argument(
        "--shard-usable-rows",
        type=int,
        default=8192,
        metavar="N",
        help="Target usable labeled rows per shard .npz when using --labels-run-dir.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume sharded labeling using manifest.json in --labels-run-dir.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "With --resume, allow continuing even if CLI teacher/layout settings "
            "do not match the manifest (unsafe)."
        ),
    )
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--scenarios", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-boards", type=int, default=None)
    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--learning-rate",
        type=float,
        default=TrainConfig.learning_rate,
    )
    p.add_argument(
        "--soft-target-weight",
        type=float,
        default=0.0,
    )
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--save-step", type=str, default="you_better_name_me")
    p.add_argument(
        "--value-network",
        choices=("qcnn", "qnetwork"),
        default="qcnn",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging (per-batch loss during supervised training).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=250,
        metavar="N",
        help="Log teacher-labeling progress every N boards (0 = no intermediate progress).",
    )
    p.add_argument(
        "--agreement-only",
        action="store_true",
        help=(
            "Load --checkpoint and teacher labels from --labels (.npz or shard dir); "
            "log train/val teacher agreement and exit (no optimizer steps)."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="PATH",
        help="With --agreement-only: supervised QCNN checkpoint (.pt) with config + q_network_state_dict.",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        metavar="F",
        help=(
            "Optional train/val split before supervised training: validation fraction in [0,1). "
            "0 keeps all rows for training. Uses deterministic shuffle (see --split-seed)."
        ),
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        metavar="INT",
        help="RNG seed for train/val assignment; defaults to --seed.",
    )
    p.add_argument(
        "--split-by-source",
        action="store_true",
        help="Assign entire source_indexes groups to train or val (reduces trajectory leakage).",
    )
    p.add_argument(
        "--log-agreement-every-epoch",
        action="store_true",
        help="After each epoch, one batched forward pass on the val holdout (needs --val-fraction > 0).",
    )
    p.add_argument(
        "--log-train-agreement-every-epoch",
        action="store_true",
        help="After each epoch, batched teacher agreement on the train slice.",
    )
    p.add_argument(
        "--agreement-batch-size",
        type=int,
        default=512,
        metavar="N",
        help="Batch size for teacher-agreement eval forwards.",
    )
    p.add_argument(
        "--agreement-train-metrics-too",
        action="store_true",
        help="With --agreement-only: when val_fraction > 0, also log train-split metrics.",
    )
    return p.parse_args()


def configure_logging(*, verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(levelname)s %(name)s: %(message)s" if verbose else "%(message)s"
    if not _LOG.handlers:
        h = logging.StreamHandler()
        _LOG.addHandler(h)
        _LOG.propagate = False
    for handler in _LOG.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(logging.Formatter(fmt))
    _LOG.setLevel(level)


def resolve_labels_arg(args: argparse.Namespace) -> Path:
    return Path(args.labels)


def main() -> None:
    args = parse_args()
    configure_logging(verbose=bool(args.verbose))
    seed_everything(int(args.seed))
    device = resolve_device(args.device)

    labels_path = resolve_labels_arg(args)

    if args.label_only and args.train_only:
        raise SystemExit("Cannot combine --label-only with --train-only")

    if args.resume and args.labels_run_dir is None:
        raise SystemExit("--resume requires --labels-run-dir")
    if (
        args.labels_run_dir is not None
        and args.train_only
        and not args.label_only
    ):
        raise SystemExit(
            "Use --labels <shard_run_dir> with --train-only (not --labels-run-dir)"
        )

    if args.agreement_only:
        if args.label_only or args.train_only:
            raise SystemExit(
                "--agreement-only cannot combine with --label-only / --train-only"
            )
        if args.labels_run_dir is not None:
            raise SystemExit(
                "--agreement-only reads labels from --labels only (omit --labels-run-dir)"
            )
        if args.checkpoint is None:
            raise SystemExit("--agreement-only requires --checkpoint")
        ck_ag = args.checkpoint.expanduser().resolve()
        if not ck_ag.is_file():
            raise SystemExit(f"--checkpoint not found: {ck_ag}")
        if args.log_agreement_every_epoch or args.log_train_agreement_every_epoch:
            raise SystemExit(
                "Agreement epoch hooks apply to supervised training; omit them with "
                "--agreement-only"
            )
        split_seed_eff = (
            int(args.split_seed) if args.split_seed is not None else int(args.seed)
        )
        run_teacher_agreement_report(
            checkpoint=args.checkpoint,
            labels_path=labels_path,
            val_fraction=float(args.val_fraction),
            split_seed=split_seed_eff,
            split_by_source=bool(args.split_by_source),
            device=device,
            batch_size=int(args.agreement_batch_size),
            log_train_metrics=bool(args.agreement_train_metrics_too),
            verbose=bool(args.verbose),
        )
        return

    if args.log_agreement_every_epoch and args.val_fraction <= 0:
        raise SystemExit("--log-agreement-every-epoch requires --val-fraction > 0")
    if args.split_by_source and args.val_fraction <= 0:
        raise SystemExit("--split-by-source requires --val-fraction > 0")

    usable: np.ndarray
    masks: np.ndarray
    tac: np.ndarray
    tq: np.ndarray

    run_label = not args.train_only or args.label_only

    base_cfg = TrainConfig(
        seed=int(args.seed),
        learning_rate=float(args.learning_rate),
        batch_size=int(args.batch_size),
        value_network=args.value_network,  # type: ignore[arg-type]
        device=args.device,
        model_dir=str(args.model_dir),
    )
    train_cfg = merge_train_config_with_init(
        base_cfg,
        init_checkpoint_path=args.init_checkpoint,
        learning_rate_override=float(args.learning_rate),
        value_network_override=args.value_network,
    )

    interrupted_label = False

    if run_label:
        if args.dataset is None:
            raise SystemExit("--dataset required when labeling")
        if args.labels_run_dir is not None:
            _LOG.info(
                "[imitation] phase=label(sharded) | dataset=%s encoding=%s | "
                "MC stages=%s scenarios=%s | run_dir=%s chunk_rows=%s shard_usable=%s resume=%s",
                args.dataset,
                args.dataset_encoding,
                args.stages,
                args.scenarios,
                args.labels_run_dir,
                args.chunk_rows,
                args.shard_usable_rows,
                args.resume,
            )
            clean, _manifest = run_sharded_labeling(
                dataset_path=args.dataset,
                dataset_encoding=args.dataset_encoding,
                run_dir=args.labels_run_dir,
                stages=int(args.stages),
                scenarios=int(args.scenarios),
                seed=int(args.seed),
                chunk_rows=int(args.chunk_rows),
                shard_usable_rows=int(args.shard_usable_rows),
                resume=bool(args.resume),
                force=bool(args.force),
                max_boards=args.max_boards,
                log_every=int(args.log_every),
            )
            interrupted_label = not clean
        else:
            _LOG.info(
                "[imitation] phase=label | dataset=%s encoding=%s | MC stages=%s scenarios=%s | "
                "labels_out=%s",
                args.dataset,
                args.dataset_encoding,
                args.stages,
                args.scenarios,
                labels_path,
            )
            boards = load_board_dataset(args.dataset, encoding=args.dataset_encoding)
            _LOG.info("[imitation] loaded board tensor shape=%s", boards.shape)
            usable, masks, tac, tq, src_indexes = label_board_states(
                boards,
                stages=int(args.stages),
                scenarios=int(args.scenarios),
                seed=int(args.seed),
                max_boards=args.max_boards,
                log_every=int(args.log_every),
            )
            save_labels_npz(
                path=labels_path,
                boards=usable,
                action_masks=masks,
                teacher_actions=tac,
                teacher_q=tq,
                source_indexes=src_indexes,
                stages=int(args.stages),
                scenarios=int(args.scenarios),
                seed=int(args.seed),
                dataset_path=str(args.dataset.resolve()),
            )
            _LOG.info("saved %s usable boards → %s", usable.shape[0], labels_path)

    if args.label_only:
        if interrupted_label:
            sys.exit(130)
        return

    if interrupted_label:
        _LOG.error(
            "[imitation] sharded labeling interrupted — fix manifest/run_dir or pass "
            "--resume before supervised training"
        )
        sys.exit(130)

    if not run_label:
        _LOG.info(
            "[imitation] train-only: loading teacher labels from %s (no MC replanning)",
            labels_path,
        )
        payload_l = load_labels_for_training(labels_path)
        usable = payload_l["boards"]  # type: ignore[assignment]
        masks = payload_l["action_masks"]  # type: ignore[assignment]
        tac = payload_l["teacher_actions"]  # type: ignore[assignment]
        tq = payload_l["teacher_q"]  # type: ignore[assignment]
    else:
        train_src = (
            args.labels_run_dir.expanduser().resolve()
            if args.labels_run_dir is not None
            else labels_path.expanduser().resolve()
        )
        _LOG.info(
            "[imitation] loading labels for training from %s",
            train_src,
        )
        payload_l = load_labels_for_training(train_src)
        usable = payload_l["boards"]  # type: ignore[assignment]
        masks = payload_l["action_masks"]  # type: ignore[assignment]
        tac = payload_l["teacher_actions"]  # type: ignore[assignment]
        tq = payload_l["teacher_q"]  # type: ignore[assignment]

    split_seed_eff = (
        int(args.split_seed) if args.split_seed is not None else int(args.seed)
    )
    raw_si = payload_l.get("source_indexes")
    src_np = (
        None if raw_si is None else np.asarray(raw_si, dtype=np.int64).reshape(-1)
    )

    if args.val_fraction > 0:
        if args.split_by_source and src_np is None:
            raise SystemExit("--split-by-source requires source_indexes in labels")
        tr_idx, va_idx = compute_train_val_split_arrays(
            n_rows=int(usable.shape[0]),
            source_indexes=src_np,
            val_fraction=float(args.val_fraction),
            split_seed=split_seed_eff,
            split_by_source=bool(args.split_by_source),
        )
        train_boards = usable[tr_idx]
        train_masks = masks[tr_idx]
        train_tac = tac[tr_idx]
        train_tq = tq[tr_idx]
        if va_idx.shape[0] > 0:
            val_b = usable[va_idx]
            val_m = masks[va_idx]
            val_ta = tac[va_idx]
        else:
            val_b = None
            val_m = None
            val_ta = None
    else:
        train_boards = usable
        train_masks = masks
        train_tac = tac
        train_tq = tq
        val_b = None
        val_m = None
        val_ta = None

    _LOG.info(
        "[imitation] supervised train | device=%s train_rows=%s val_holdout=%s epochs=%s batch=%s",
        device,
        train_boards.shape[0],
        int(val_b.shape[0]) if val_b is not None else 0,
        args.epochs,
        args.batch_size,
    )

    ck_out = train_imitation(
        boards=train_boards.astype(np.int64, copy=False),
        action_masks=train_masks.astype(np.bool_, copy=False),
        teacher_actions=train_tac.astype(np.int64, copy=False),
        teacher_q=train_tq.astype(np.float32, copy=False),
        train_cfg=train_cfg,
        init_checkpoint_path=args.init_checkpoint,
        model_dir=args.model_dir,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        device=device,
        soft_target_weight=float(args.soft_target_weight),
        save_step=args.save_step,
        val_boards=val_b.astype(np.int64, copy=False) if val_b is not None else None,
        val_masks=val_m.astype(np.bool_, copy=False) if val_m is not None else None,
        val_teacher_actions=val_ta.astype(np.int64, copy=False)
        if val_ta is not None
        else None,
        log_agreement_every_epoch=bool(args.log_agreement_every_epoch),
        log_train_agreement_every_epoch=bool(args.log_train_agreement_every_epoch),
        agreement_batch_size=int(args.agreement_batch_size),
    )
    _LOG.info("saved checkpoint %s", ck_out)


def parse_agreement_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Masked-argmax student Q vs teacher_actions on a labeled artifact (.npz or shard dir)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-by-source", action="store_true")
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--train-metrics-too",
        action="store_true",
        help="Also log train-split metrics when val_fraction > 0.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def agreement_main() -> None:
    args = parse_agreement_args()
    configure_logging(verbose=bool(args.verbose))
    seed_everything(int(args.split_seed))
    device = resolve_device(args.device)
    run_teacher_agreement_report(
        checkpoint=args.checkpoint,
        labels_path=args.labels,
        val_fraction=float(args.val_fraction),
        split_seed=int(args.split_seed),
        split_by_source=bool(args.split_by_source),
        device=device,
        batch_size=int(args.batch_size),
        log_train_metrics=bool(args.train_metrics_too),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
