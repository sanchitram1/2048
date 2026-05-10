from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from game2048.game import GameLogic
from training.env import Game2048Env
from training.imitation import label_board_states
from training.inference import choose_greedy_action, load_q_network
from training.label_schema import board_hashes, load_labels_npz_any, save_labels_npz_v2
from training.planning import choose_n_step_mc
from training.dqn import legal_actions_to_mask

_LOG = logging.getLogger("game2048.expert")

MANIFEST_VERSION = 1
DEFAULT_CHECKPOINT = Path("experiments/current_best_imitation_checkpoint.pt")
DEFAULT_OUTPUT_DIR = Path("data/expert-latest")
_shutdown_requested = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_shutdown_flag(signum: int, frame: object | None) -> None:
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    sig_name = signal.Signals(signum).name
    _LOG.warning(
        "[expert] received %s; will save the current batch and stop",
        sig_name,
    )


def _reset_shutdown_flag() -> None:
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = False


def shutdown_requested() -> bool:
    return bool(_shutdown_requested)


@dataclass
class ExpertManifest:
    """Manifest for policy-collected, MCTS-labeled expert-iteration shards.

    The first fields intentionally match ``training.imitation.ShardLabelManifest``
    so ``uv run imitate --train-only --labels <run_dir>`` can load this directory.
    """

    format_version: int = MANIFEST_VERSION
    dataset_path: str = ""
    dataset_encoding: str = "log2"
    stages: int = 2
    scenarios: int = 10
    seed: int = 7
    chunk_rows: int = 1024
    shard_usable_rows: int = 1024
    next_raw_row: int = 0
    global_usable_labeled: int = 0
    shard_files: list[str] = field(default_factory=list)
    complete: bool = False
    interrupted: bool = False
    checkpoint: str = ""
    device: str = "cpu"
    epsilon: float = 0.0
    next_episode_id: int = 0
    snapshot_every_rows: int = 1024
    snapshot_every_minutes: float = 10.0
    dedupe: bool = True
    extreme_disagreement: bool = False
    min_teacher_gap: float = 0.0
    min_model_margin: float = 0.0
    teacher_q_cutoff: float = -1.0e5
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["updated_at"] = _utc_now()
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> ExpertManifest:
        return cls(
            format_version=int(data.get("format_version", MANIFEST_VERSION)),
            dataset_path=str(data.get("dataset_path", "")),
            dataset_encoding=str(data.get("dataset_encoding", "log2")),
            stages=int(data.get("stages", 2)),
            scenarios=int(data.get("scenarios", 10)),
            seed=int(data.get("seed", 7)),
            chunk_rows=int(
                data.get("chunk_rows", data.get("snapshot_every_rows", 1024))
            ),
            shard_usable_rows=int(
                data.get("shard_usable_rows", data.get("snapshot_every_rows", 1024))
            ),
            next_raw_row=int(data.get("next_raw_row", 0)),
            global_usable_labeled=int(data.get("global_usable_labeled", 0)),
            shard_files=[str(x) for x in data.get("shard_files", [])],
            complete=bool(data.get("complete", False)),
            interrupted=bool(data.get("interrupted", False)),
            checkpoint=str(data.get("checkpoint", "")),
            device=str(data.get("device", "cpu")),
            epsilon=float(data.get("epsilon", 0.0)),
            next_episode_id=int(data.get("next_episode_id", 0)),
            snapshot_every_rows=int(data.get("snapshot_every_rows", 1024)),
            snapshot_every_minutes=float(data.get("snapshot_every_minutes", 10.0)),
            dedupe=bool(data.get("dedupe", True)),
            extreme_disagreement=bool(data.get("extreme_disagreement", False)),
            min_teacher_gap=float(data.get("min_teacher_gap", 0.0)),
            min_model_margin=float(data.get("min_model_margin", 0.0)),
            teacher_q_cutoff=float(data.get("teacher_q_cutoff", -1.0e5)),
            created_at=str(data.get("created_at", "")) or _utc_now(),
            updated_at=str(data.get("updated_at", "")) or _utc_now(),
        )


class PendingRows:
    def __init__(self) -> None:
        self.boards: list[np.ndarray] = []
        self.source_indexes: list[int] = []
        self.episode_ids: list[int] = []
        self.move_indexes: list[int] = []

    def __len__(self) -> int:
        return len(self.boards)

    def append(
        self,
        *,
        board: np.ndarray,
        source_index: int,
        episode_id: int,
        move_idx: int,
    ) -> None:
        self.boards.append(np.asarray(board, dtype=np.int64).copy())
        self.source_indexes.append(int(source_index))
        self.episode_ids.append(int(episode_id))
        self.move_indexes.append(int(move_idx))

    def clear(self) -> None:
        self.boards.clear()
        self.source_indexes.clear()
        self.episode_ids.clear()
        self.move_indexes.clear()

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.boards:
            raise RuntimeError("Cannot materialize an empty pending buffer")
        return (
            np.stack(self.boards, axis=0).astype(np.int64, copy=False),
            np.asarray(self.source_indexes, dtype=np.int64),
            np.asarray(self.episode_ids, dtype=np.int64),
            np.asarray(self.move_indexes, dtype=np.int64),
        )


class PendingDisagreementRows:
    def __init__(self) -> None:
        self.boards: list[np.ndarray] = []
        self.action_masks: list[np.ndarray] = []
        self.teacher_actions: list[int] = []
        self.teacher_q: list[np.ndarray] = []
        self.source_indexes: list[int] = []
        self.episode_ids: list[int] = []
        self.move_indexes: list[int] = []
        self.model_actions: list[int] = []
        self.model_q: list[np.ndarray] = []
        self.model_masked_q: list[np.ndarray] = []
        self.model_margins: list[float] = []
        self.teacher_gaps: list[float] = []

    def __len__(self) -> int:
        return len(self.boards)

    def append(
        self,
        *,
        board: np.ndarray,
        action_mask: np.ndarray,
        teacher_action: int,
        teacher_q: np.ndarray,
        source_index: int,
        episode_id: int,
        move_idx: int,
        model_action: int,
        model_q: np.ndarray,
        model_masked_q: np.ndarray,
        model_margin: float,
        teacher_gap: float,
    ) -> None:
        self.boards.append(np.asarray(board, dtype=np.int64).copy())
        self.action_masks.append(np.asarray(action_mask, dtype=np.bool_).copy())
        self.teacher_actions.append(int(teacher_action))
        self.teacher_q.append(np.asarray(teacher_q, dtype=np.float32).copy())
        self.source_indexes.append(int(source_index))
        self.episode_ids.append(int(episode_id))
        self.move_indexes.append(int(move_idx))
        self.model_actions.append(int(model_action))
        self.model_q.append(np.asarray(model_q, dtype=np.float32).copy())
        self.model_masked_q.append(np.asarray(model_masked_q, dtype=np.float32).copy())
        self.model_margins.append(float(model_margin))
        self.teacher_gaps.append(float(teacher_gap))

    def clear(self) -> None:
        self.__init__()

    def arrays(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, np.ndarray],
    ]:
        if not self.boards:
            raise RuntimeError("Cannot materialize an empty pending buffer")
        extras = {
            "model_actions": np.asarray(self.model_actions, dtype=np.int64),
            "model_q": np.stack(self.model_q, axis=0).astype(np.float32, copy=False),
            "model_masked_q": np.stack(self.model_masked_q, axis=0).astype(
                np.float32, copy=False
            ),
            "model_margin": np.asarray(self.model_margins, dtype=np.float32),
            "teacher_gap": np.asarray(self.teacher_gaps, dtype=np.float32),
        }
        return (
            np.stack(self.boards, axis=0).astype(np.int64, copy=False),
            np.stack(self.action_masks, axis=0).astype(np.bool_, copy=False),
            np.asarray(self.teacher_actions, dtype=np.int64),
            np.stack(self.teacher_q, axis=0).astype(np.float32, copy=False),
            np.asarray(self.source_indexes, dtype=np.int64),
            np.asarray(self.episode_ids, dtype=np.int64),
            np.asarray(self.move_indexes, dtype=np.int64),
            extras,
        )


def manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def load_expert_manifest(output_dir: Path) -> ExpertManifest:
    path = manifest_path(output_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Expert manifest not found: {path}")
    return ExpertManifest.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


def save_expert_manifest_atomic(output_dir: Path, manifest: ExpertManifest) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(output_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _metadata_matches(a: ExpertManifest, b: ExpertManifest) -> bool:
    return (
        a.dataset_path == b.dataset_path
        and a.stages == b.stages
        and a.scenarios == b.scenarios
        and a.seed == b.seed
        and a.checkpoint == b.checkpoint
        and a.epsilon == b.epsilon
        and a.dedupe == b.dedupe
        and a.extreme_disagreement == b.extreme_disagreement
        and a.min_teacher_gap == b.min_teacher_gap
        and a.min_model_margin == b.min_model_margin
        and a.teacher_q_cutoff == b.teacher_q_cutoff
    )


def _seen_hashes_from_shards(output_dir: Path, shard_files: list[str]) -> set[str]:
    seen: set[str] = set()
    for shard in shard_files:
        payload = load_labels_npz_any(output_dir / shard)
        hashes = payload["board_hash"]
        if isinstance(hashes, np.ndarray):
            seen.update(str(x) for x in hashes.tolist())
    return seen


def _next_shard_name(manifest: ExpertManifest) -> str:
    return f"shard_{len(manifest.shard_files) + 1:06d}.npz"


def _choose_policy_action(
    *,
    q_network: object,
    state: np.ndarray,
    legal_actions: list[int],
    device: object,
    rng: random.Random,
    epsilon: float,
) -> int:
    if epsilon > 0 and rng.random() < epsilon:
        return int(rng.choice(legal_actions))
    model_action = choose_greedy_action(
        q_network=q_network,  # type: ignore[arg-type]
        state=state,
        legal_actions=legal_actions,
        device=device,  # type: ignore[arg-type]
    )
    return int(model_action.action)


def _flush_pending(
    *,
    pending: PendingRows,
    output_dir: Path,
    manifest: ExpertManifest,
    log_every: int,
) -> int:
    if len(pending) == 0:
        return 0

    boards, source_indexes, episode_ids, move_indexes = pending.arrays()
    _LOG.info(
        "[expert] MCTS labeling %d policy-collected boards (stages=%d scenarios=%d)",
        boards.shape[0],
        manifest.stages,
        manifest.scenarios,
    )
    usable, masks, actions, teacher_q, src = label_board_states(
        boards,
        stages=manifest.stages,
        scenarios=manifest.scenarios,
        seed=manifest.seed,
        max_boards=None,
        log_every=log_every,
        usable_rng_offset=int(source_indexes[0]) if source_indexes.size else 0,
        usable_prefiltered=(boards, source_indexes),
    )
    if usable.shape[0] == 0:
        pending.clear()
        return 0

    shard_name = _next_shard_name(manifest)
    shard_path = output_dir / shard_name
    # label_board_states returns the source index array supplied above, so build a
    # lookup in case future filtering is added to the prefiltered path.
    row_by_source = {int(s): i for i, s in enumerate(source_indexes.tolist())}
    selected = np.asarray([row_by_source[int(s)] for s in src.tolist()], dtype=np.int64)
    save_labels_npz_v2(
        path=shard_path,
        boards=usable,
        action_masks=masks,
        teacher_actions=actions,
        teacher_q=teacher_q,
        source_indexes=src,
        stages=manifest.stages,
        scenarios=manifest.scenarios,
        seed=manifest.seed,
        dataset_path=manifest.dataset_path,
        policy_checkpoint=manifest.checkpoint,
        episode_id=episode_ids[selected],
        move_idx=move_indexes[selected],
    )

    rows = int(usable.shape[0])
    manifest.shard_files.append(shard_name)
    manifest.global_usable_labeled += rows
    manifest.next_raw_row += int(boards.shape[0])
    save_expert_manifest_atomic(output_dir, manifest)
    pending.clear()
    _LOG.info(
        "[expert] saved %s rows=%d total_rows=%d",
        shard_path,
        rows,
        manifest.global_usable_labeled,
    )
    return rows


def _flush_disagreement_pending(
    *,
    pending: PendingDisagreementRows,
    output_dir: Path,
    manifest: ExpertManifest,
) -> int:
    if len(pending) == 0:
        return 0

    (
        boards,
        action_masks,
        teacher_actions,
        teacher_q,
        source_indexes,
        episode_ids,
        move_indexes,
        extra_arrays,
    ) = pending.arrays()
    shard_name = _next_shard_name(manifest)
    shard_path = output_dir / shard_name
    save_labels_npz_v2(
        path=shard_path,
        boards=boards,
        action_masks=action_masks,
        teacher_actions=teacher_actions,
        teacher_q=teacher_q,
        source_indexes=source_indexes,
        stages=manifest.stages,
        scenarios=manifest.scenarios,
        seed=manifest.seed,
        dataset_path=manifest.dataset_path,
        policy_checkpoint=manifest.checkpoint,
        episode_id=episode_ids,
        move_idx=move_indexes,
        extra_arrays=extra_arrays,
    )

    rows = int(boards.shape[0])
    manifest.shard_files.append(shard_name)
    manifest.global_usable_labeled += rows
    manifest.next_raw_row += rows
    save_expert_manifest_atomic(output_dir, manifest)
    pending.clear()
    _LOG.info(
        "[expert] saved disagreement %s rows=%d total_rows=%d",
        shard_path,
        rows,
        manifest.global_usable_labeled,
    )
    return rows


def _best_and_gap(
    *,
    values: np.ndarray,
    legal_actions: list[int],
    cutoff: float | None = None,
) -> tuple[int, float] | None:
    candidates: list[tuple[int, float]] = []
    for action in legal_actions:
        value = float(values[action])
        if not np.isfinite(value):
            continue
        if cutoff is not None and value <= cutoff:
            continue
        candidates.append((int(action), value))
    if len(candidates) < 2:
        return None
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0], candidates[0][1] - candidates[1][1]


def _disagreement_label_for_state(
    *,
    state: np.ndarray,
    legal_actions: list[int],
    q_network: object,
    device: object,
    stages: int,
    scenarios: int,
    rng: random.Random,
    min_teacher_gap: float,
    min_model_margin: float,
    teacher_q_cutoff: float,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float] | None:
    model_action = choose_greedy_action(
        q_network=q_network,  # type: ignore[arg-type]
        state=state,
        legal_actions=legal_actions,
        device=device,  # type: ignore[arg-type]
    )
    model_q = np.asarray(model_action.q_values, dtype=np.float32)
    model_best = _best_and_gap(values=model_q, legal_actions=legal_actions)
    if model_best is None:
        return None
    _model_best_action, model_margin = model_best
    if model_margin < min_model_margin:
        return None

    game = GameLogic(skip_initial_spawn=True)
    game.grid = np.asarray(state, dtype=np.int16).copy()
    planned = choose_n_step_mc(game, stages=stages, scenarios=scenarios, rng=rng)
    teacher_q = np.asarray(planned.q_values, dtype=np.float32)
    teacher_best = _best_and_gap(
        values=teacher_q,
        legal_actions=[int(action) for action in planned.legal_actions],
        cutoff=teacher_q_cutoff,
    )
    if teacher_best is None:
        return None
    teacher_action, teacher_gap = teacher_best
    if teacher_gap < min_teacher_gap:
        return None
    if int(model_action.action) == int(teacher_action):
        return None

    action_mask = legal_actions_to_mask(4, legal_actions)
    model_masked_q = model_q.copy()
    model_masked_q[~action_mask] = -1.0e9
    return (
        int(teacher_action),
        teacher_q,
        action_mask,
        model_q,
        model_masked_q,
        int(model_action.action),
        float(model_margin),
        float(teacher_gap),
    )


def run_expert_iteration(
    *,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    stages: int,
    scenarios: int,
    seed: int,
    epsilon: float,
    snapshot_every_rows: int,
    snapshot_every_minutes: float,
    max_episodes: int | None,
    max_rows: int | None,
    resume: bool,
    force: bool,
    dedupe: bool,
    extreme_disagreement: bool = False,
    min_teacher_gap: float = 8.0,
    min_model_margin: float = 0.0,
    teacher_q_cutoff: float = -1.0e5,
    log_every: int = 100,
) -> ExpertManifest:
    if snapshot_every_rows < 1:
        raise ValueError("snapshot_every_rows must be >= 1")
    if snapshot_every_minutes <= 0:
        raise ValueError("snapshot_every_minutes must be > 0")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if min_teacher_gap < 0:
        raise ValueError("min_teacher_gap must be >= 0")
    if min_model_margin < 0:
        raise ValueError("min_model_margin must be >= 0")

    resolved_checkpoint = checkpoint.expanduser().resolve()
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint}")

    rd = output_dir.expanduser().resolve()
    dataset_path = f"expert://{resolved_checkpoint}"
    desired = ExpertManifest(
        dataset_path=dataset_path,
        stages=stages,
        scenarios=scenarios,
        seed=seed,
        chunk_rows=snapshot_every_rows,
        shard_usable_rows=snapshot_every_rows,
        checkpoint=str(resolved_checkpoint),
        device=device,
        epsilon=epsilon,
        snapshot_every_rows=snapshot_every_rows,
        snapshot_every_minutes=snapshot_every_minutes,
        dedupe=dedupe,
        extreme_disagreement=extreme_disagreement,
        min_teacher_gap=min_teacher_gap,
        min_model_margin=min_model_margin,
        teacher_q_cutoff=teacher_q_cutoff,
    )

    if resume:
        manifest = load_expert_manifest(rd)
        if not force and not _metadata_matches(desired, manifest):
            raise SystemExit(
                "Existing expert manifest does not match requested options; "
                "rerun with matching options or pass --force"
            )
        manifest.complete = False
        manifest.interrupted = False
        manifest.snapshot_every_rows = snapshot_every_rows
        manifest.snapshot_every_minutes = snapshot_every_minutes
    else:
        if manifest_path(rd).exists() and not force:
            raise SystemExit(
                f"{manifest_path(rd)} already exists; use --resume or --force"
            )
        rd.mkdir(parents=True, exist_ok=True)
        manifest = desired
        save_expert_manifest_atomic(rd, manifest)

    q_network, _config, torch_device = load_q_network(
        resolved_checkpoint, device_name=device
    )
    env = Game2048Env()
    rng = random.Random(seed + manifest.next_episode_id)
    pending = PendingRows()
    disagreement_pending = PendingDisagreementRows()
    seen_hashes = (
        _seen_hashes_from_shards(rd, manifest.shard_files) if dedupe else set()
    )
    last_flush = time.monotonic()
    episodes_done = 0

    _LOG.info(
        "[expert] starting checkpoint=%s output=%s rows=%d shards=%d resume=%s mode=%s",
        resolved_checkpoint,
        rd,
        manifest.global_usable_labeled,
        len(manifest.shard_files),
        resume,
        "extreme_disagreement" if manifest.extreme_disagreement else "policy_states",
    )

    while not shutdown_requested():
        if max_episodes is not None and episodes_done >= max_episodes:
            break
        if (
            max_rows is not None
            and manifest.global_usable_labeled
            + (len(disagreement_pending) if manifest.extreme_disagreement else len(pending))
            >= max_rows
        ):
            break

        episode_id = manifest.next_episode_id
        env.seed(seed + episode_id)
        state, _info = env.reset()
        move_idx = 0
        while not shutdown_requested():
            legal_actions = env.legal_actions()
            if not legal_actions:
                break
            if (
                max_rows is not None
                and manifest.global_usable_labeled
                + (
                    len(disagreement_pending)
                    if manifest.extreme_disagreement
                    else len(pending)
                )
                >= max_rows
            ):
                break

            board = np.asarray(state, dtype=np.int64)
            board_hash = str(board_hashes(board.reshape(1, 4, 4))[0])
            if manifest.extreme_disagreement:
                if not dedupe or board_hash not in seen_hashes:
                    planned_label = _disagreement_label_for_state(
                        state=state,
                        legal_actions=legal_actions,
                        q_network=q_network,
                        device=torch_device,
                        stages=manifest.stages,
                        scenarios=manifest.scenarios,
                        rng=rng,
                        min_teacher_gap=manifest.min_teacher_gap,
                        min_model_margin=manifest.min_model_margin,
                        teacher_q_cutoff=manifest.teacher_q_cutoff,
                    )
                    if planned_label is not None:
                        (
                            teacher_action,
                            teacher_q,
                            action_mask,
                            model_q,
                            model_masked_q,
                            model_action,
                            model_margin,
                            teacher_gap,
                        ) = planned_label
                        disagreement_pending.append(
                            board=board,
                            action_mask=action_mask,
                            teacher_action=teacher_action,
                            teacher_q=teacher_q,
                            source_index=manifest.next_raw_row
                            + len(disagreement_pending),
                            episode_id=episode_id,
                            move_idx=move_idx,
                            model_action=model_action,
                            model_q=model_q,
                            model_masked_q=model_masked_q,
                            model_margin=model_margin,
                            teacher_gap=teacher_gap,
                        )
                        seen_hashes.add(board_hash)
            elif not dedupe or board_hash not in seen_hashes:
                pending.append(
                    board=board,
                    source_index=manifest.next_raw_row + len(pending),
                    episode_id=episode_id,
                    move_idx=move_idx,
                )
                seen_hashes.add(board_hash)

            action = _choose_policy_action(
                q_network=q_network,
                state=state,
                legal_actions=legal_actions,
                device=torch_device,
                rng=rng,
                epsilon=epsilon,
            )
            state, _reward, done, truncated, _info = env.step(action)
            move_idx += 1

            elapsed_since_flush = (time.monotonic() - last_flush) / 60.0
            active_pending_len = (
                len(disagreement_pending) if manifest.extreme_disagreement else len(pending)
            )
            if active_pending_len >= snapshot_every_rows or (
                active_pending_len > 0 and elapsed_since_flush >= snapshot_every_minutes
            ):
                if manifest.extreme_disagreement:
                    _flush_disagreement_pending(
                        pending=disagreement_pending,
                        output_dir=rd,
                        manifest=manifest,
                    )
                else:
                    _flush_pending(
                        pending=pending,
                        output_dir=rd,
                        manifest=manifest,
                        log_every=log_every,
                    )
                last_flush = time.monotonic()

            if done or truncated:
                break

        episodes_done += 1
        manifest.next_episode_id = episode_id + 1
        save_expert_manifest_atomic(rd, manifest)

    if manifest.extreme_disagreement and len(disagreement_pending) > 0:
        _flush_disagreement_pending(
            pending=disagreement_pending,
            output_dir=rd,
            manifest=manifest,
        )
    elif len(pending) > 0:
        _flush_pending(
            pending=pending,
            output_dir=rd,
            manifest=manifest,
            log_every=log_every,
        )

    manifest.complete = not shutdown_requested() and (
        max_episodes is not None or max_rows is not None
    )
    manifest.interrupted = shutdown_requested()
    save_expert_manifest_atomic(rd, manifest)
    _LOG.info(
        "[expert] finished rows=%d shards=%d episodes_next=%d interrupted=%s",
        manifest.global_usable_labeled,
        len(manifest.shard_files),
        manifest.next_episode_id,
        manifest.interrupted,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect board states from the current best policy, relabel them with "
            "the MCTS teacher, and write schema-v2 expert-iteration shards."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--scenarios", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Probability of taking a random legal policy action while collecting states.",
    )
    parser.add_argument("--snapshot-every-rows", type=int, default=1024)
    parser.add_argument("--snapshot-every-minutes", type=float, default=10.0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite or continue despite manifest option mismatches.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate board positions within the expert run.",
    )
    parser.add_argument(
        "--extreme-disagreement",
        action="store_true",
        help=(
            "Only save states where greedy model action disagrees with MCTS and "
            "both model/planner confidence thresholds pass. Uses the same MCTS "
            "call for filtering and saved teacher_q labels."
        ),
    )
    parser.add_argument(
        "--min-teacher-gap",
        type=float,
        default=8.0,
        help="Minimum MCTS best-vs-second legal Q gap for --extreme-disagreement.",
    )
    parser.add_argument(
        "--min-model-margin",
        type=float,
        default=0.0,
        help="Minimum model best-vs-second legal Q gap for --extreme-disagreement.",
    )
    parser.add_argument(
        "--teacher-q-cutoff",
        type=float,
        default=-1.0e5,
        help="Planner Q values at or below this are ignored for gap filtering.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    _reset_shutdown_flag()
    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _set_shutdown_flag)
    signal.signal(signal.SIGTERM, _set_shutdown_flag)
    try:
        run_expert_iteration(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            device=args.device,
            stages=args.stages,
            scenarios=args.scenarios,
            seed=args.seed,
            epsilon=args.epsilon,
            snapshot_every_rows=args.snapshot_every_rows,
            snapshot_every_minutes=args.snapshot_every_minutes,
            max_episodes=args.max_episodes,
            max_rows=args.max_rows,
            resume=args.resume,
            force=args.force,
            dedupe=not args.no_dedupe,
            extreme_disagreement=args.extreme_disagreement,
            min_teacher_gap=args.min_teacher_gap,
            min_model_margin=args.min_model_margin,
            teacher_q_cutoff=args.teacher_q_cutoff,
            log_every=args.log_every,
        )
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


if __name__ == "__main__":
    main()
