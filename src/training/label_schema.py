from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ACTION_DIM = 4
SCHEMA_VERSION_V1 = 1
SCHEMA_VERSION_V2 = 2
DEFAULT_POLICY_TEMPERATURE = 1.0
DEFAULT_INVALID_Q_THRESHOLD = -1.0e8


@dataclass(frozen=True)
class TeacherLabelManifest:
    schema_version: int = SCHEMA_VERSION_V2
    dataset_path: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    shard_files: list[str] = field(default_factory=list)
    rows: int = 0
    stages: int | None = None
    scenarios: int | None = None
    seed: int | None = None
    policy_temperature: float = DEFAULT_POLICY_TEMPERATURE
    invalid_q_threshold: float = DEFAULT_INVALID_Q_THRESHOLD
    policy_checkpoint: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> TeacherLabelManifest:
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION_V2)),
            dataset_path=str(data.get("dataset_path", "")),
            created_at=str(data.get("created_at", "")),
            shard_files=[str(x) for x in data.get("shard_files", [])],
            rows=int(data.get("rows", 0)),
            stages=_optional_int(data.get("stages")),
            scenarios=_optional_int(data.get("scenarios")),
            seed=_optional_int(data.get("seed")),
            policy_temperature=float(
                data.get("policy_temperature", DEFAULT_POLICY_TEMPERATURE)
            ),
            invalid_q_threshold=float(
                data.get("invalid_q_threshold", DEFAULT_INVALID_Q_THRESHOLD)
            ),
            policy_checkpoint=_optional_str(data.get("policy_checkpoint")),
        )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.partial{path.suffix}")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def write_teacher_manifest(path: Path, manifest: TeacherLabelManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_teacher_manifest(path: Path) -> TeacherLabelManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return TeacherLabelManifest.from_json_dict(data)


def board_hashes(boards: np.ndarray) -> np.ndarray:
    arr = np.asarray(boards)
    if arr.ndim != 3 or arr.shape[1:] != (4, 4):
        raise ValueError(f"boards must have shape (N, 4, 4); got {arr.shape}")
    out = np.empty(arr.shape[0], dtype="<U32")
    stable = np.ascontiguousarray(arr.astype("<i2", copy=False))
    for i, board in enumerate(stable):
        digest = hashlib.blake2b(board.tobytes(), digest_size=16)
        out[i] = digest.hexdigest()
    return out


def teacher_policy_from_q(
    teacher_q: np.ndarray,
    action_masks: np.ndarray,
    *,
    temperature: float = DEFAULT_POLICY_TEMPERATURE,
    invalid_q_threshold: float | None = DEFAULT_INVALID_Q_THRESHOLD,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    q = np.asarray(teacher_q, dtype=np.float32)
    masks = np.asarray(action_masks, dtype=np.bool_)
    if q.ndim != 2 or q.shape[1] != ACTION_DIM:
        raise ValueError(f"teacher_q must have shape (N, {ACTION_DIM}); got {q.shape}")
    if masks.shape != q.shape:
        raise ValueError(f"action_masks must match teacher_q shape; got {masks.shape}")

    probs = np.zeros_like(q, dtype=np.float32)
    for row in range(q.shape[0]):
        legal = masks[row]
        if not legal.any():
            raise ValueError(f"row {row} has no legal actions")
        valid = legal & np.isfinite(q[row])
        if invalid_q_threshold is not None:
            valid &= q[row] > float(invalid_q_threshold)

        if not valid.any():
            probs[row, legal] = 1.0 / float(legal.sum())
            continue

        scaled = q[row, valid].astype(np.float64) / float(temperature)
        scaled -= float(np.max(scaled))
        weights = np.exp(scaled)
        denom = float(weights.sum())
        if denom <= 0 or not np.isfinite(denom):
            probs[row, legal] = 1.0 / float(legal.sum())
            continue
        probs[row, valid] = (weights / denom).astype(np.float32)

    return probs


def validate_teacher_labels(
    *,
    boards: np.ndarray,
    action_masks: np.ndarray,
    teacher_actions: np.ndarray,
    teacher_q: np.ndarray,
) -> None:
    n = int(np.asarray(boards).shape[0])
    if np.asarray(boards).shape != (n, 4, 4):
        raise ValueError(f"boards must have shape (N, 4, 4); got {boards.shape}")
    if np.asarray(action_masks).shape != (n, ACTION_DIM):
        raise ValueError(
            f"action_masks must have shape (N, {ACTION_DIM}); got {action_masks.shape}"
        )
    if np.asarray(teacher_q).shape != (n, ACTION_DIM):
        raise ValueError(
            f"teacher_q must have shape (N, {ACTION_DIM}); got {teacher_q.shape}"
        )
    actions = np.asarray(teacher_actions, dtype=np.int64).reshape(-1)
    if actions.shape != (n,):
        raise ValueError(f"teacher_actions must have shape (N,); got {actions.shape}")
    if n == 0:
        return
    if np.any((actions < 0) | (actions >= ACTION_DIM)):
        raise ValueError("teacher_actions contains out-of-range actions")
    masks = np.asarray(action_masks, dtype=np.bool_)
    if not masks[np.arange(n), actions].all():
        raise ValueError("teacher_actions must be legal under action_masks")


def save_labels_npz_v2(
    *,
    path: Path,
    boards: np.ndarray,
    action_masks: np.ndarray,
    teacher_actions: np.ndarray,
    teacher_q: np.ndarray,
    source_indexes: np.ndarray | None,
    stages: int | None,
    scenarios: int | None,
    seed: int | None,
    dataset_path: str,
    teacher_policy: np.ndarray | None = None,
    teacher_value: np.ndarray | None = None,
    teacher_score_gain: np.ndarray | None = None,
    teacher_max_tile: np.ndarray | None = None,
    policy_checkpoint: str | None = None,
    episode_id: np.ndarray | None = None,
    move_idx: np.ndarray | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
    policy_temperature: float = DEFAULT_POLICY_TEMPERATURE,
    invalid_q_threshold: float = DEFAULT_INVALID_Q_THRESHOLD,
) -> None:
    boards_arr = np.asarray(boards, dtype=np.int64)
    masks_arr = np.asarray(action_masks, dtype=np.bool_)
    actions_arr = np.asarray(teacher_actions, dtype=np.int64).reshape(-1)
    q_arr = np.asarray(teacher_q, dtype=np.float32)
    validate_teacher_labels(
        boards=boards_arr,
        action_masks=masks_arr,
        teacher_actions=actions_arr,
        teacher_q=q_arr,
    )
    n = boards_arr.shape[0]

    if source_indexes is None:
        source_arr = np.zeros(0, dtype=np.int64)
    else:
        source_arr = np.asarray(source_indexes, dtype=np.int64).reshape(-1)
        if source_arr.shape != (n,):
            raise ValueError("source_indexes length must match boards")

    if teacher_policy is None:
        policy_arr = teacher_policy_from_q(
            q_arr,
            masks_arr,
            temperature=policy_temperature,
            invalid_q_threshold=invalid_q_threshold,
        )
    else:
        policy_arr = np.asarray(teacher_policy, dtype=np.float32)
        if policy_arr.shape != (n, ACTION_DIM):
            raise ValueError("teacher_policy must have shape (N, 4)")
        validate_teacher_policy(policy_arr, masks_arr)

    episode_arr = _default_optional_row_array(episode_id, n)
    move_arr = _default_optional_row_array(move_idx, n)

    arrays: dict[str, np.ndarray] = {
        "schema_version": np.array([SCHEMA_VERSION_V2], dtype=np.int64),
        "boards": boards_arr,
        "action_masks": masks_arr,
        "teacher_actions": actions_arr,
        "teacher_q": q_arr,
        "teacher_policy": policy_arr,
        "source_indexes": source_arr,
        "board_hash": board_hashes(boards_arr),
        "episode_id": episode_arr,
        "move_idx": move_arr,
        "stages": _scalar_int_array(stages),
        "scenarios": _scalar_int_array(scenarios),
        "seed": _scalar_int_array(seed),
        "dataset_path": np.array(dataset_path, dtype=np.str_),
        "policy_temperature": np.array([policy_temperature], dtype=np.float32),
        "invalid_q_threshold": np.array([invalid_q_threshold], dtype=np.float32),
    }
    if policy_checkpoint is not None:
        arrays["policy_checkpoint"] = np.array(policy_checkpoint, dtype=np.str_)
    _add_optional_float_array(arrays, "teacher_value", teacher_value, n)
    _add_optional_float_array(arrays, "teacher_score_gain", teacher_score_gain, n)
    _add_optional_float_array(arrays, "teacher_max_tile", teacher_max_tile, n)
    if extra_arrays:
        for name, values in extra_arrays.items():
            if name in arrays:
                raise ValueError(f"extra array would overwrite schema key: {name}")
            arr = np.asarray(values)
            if arr.shape[0] != n:
                raise ValueError(f"{name} first dimension must match boards")
            arrays[name] = arr
    _write_npz_atomic(path, arrays)


def validate_teacher_policy(
    teacher_policy: np.ndarray,
    action_masks: np.ndarray,
    *,
    atol: float = 1.0e-5,
) -> None:
    policy = np.asarray(teacher_policy, dtype=np.float32)
    masks = np.asarray(action_masks, dtype=np.bool_)
    if policy.shape != masks.shape:
        raise ValueError("teacher_policy shape must match action_masks")
    if np.any(policy < -atol):
        raise ValueError("teacher_policy must be non-negative")
    if np.any(np.abs(policy[~masks]) > atol):
        raise ValueError("teacher_policy must be zero on illegal actions")
    row_sums = policy.sum(axis=1)
    if not np.allclose(row_sums, np.ones_like(row_sums), atol=atol, rtol=0):
        raise ValueError("teacher_policy rows must sum to 1")


def _scalar_int_array(value: int | None) -> np.ndarray:
    return np.array([-1 if value is None else int(value)], dtype=np.int64)


def _none_if_negative(value: int) -> int | None:
    return None if value < 0 else int(value)


def _default_optional_row_array(values: np.ndarray | None, n: int) -> np.ndarray:
    if values is None:
        return np.full(n, -1, dtype=np.int64)
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.shape != (n,):
        raise ValueError("optional row array length must match boards")
    return arr


def _add_optional_float_array(
    arrays: dict[str, np.ndarray],
    name: str,
    values: np.ndarray | None,
    n: int,
) -> None:
    if values is None:
        return
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} length must match boards")
    arrays[name] = arr


def load_labels_npz_any(path: str | Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Labels file not found: {resolved}")
    with np.load(resolved, allow_pickle=False) as data:
        boards = data["boards"].astype(np.int64, copy=False)
        masks = data["action_masks"]
        if masks.dtype != np.bool_:
            masks = masks.astype(np.bool_, copy=False)
        actions = data["teacher_actions"].astype(np.int64, copy=False)
        teacher_q = data["teacher_q"].astype(np.float32, copy=False)
        validate_teacher_labels(
            boards=boards,
            action_masks=masks,
            teacher_actions=actions,
            teacher_q=teacher_q,
        )

        schema_version = (
            int(data["schema_version"][0])
            if "schema_version" in data
            else SCHEMA_VERSION_V1
        )
        teacher_policy = (
            data["teacher_policy"].astype(np.float32, copy=False)
            if "teacher_policy" in data
            else teacher_policy_from_q(teacher_q, masks)
        )
        src = _load_optional_source_indexes(data, boards.shape[0])
        stages = _load_optional_scalar_int(data, "stages")
        scenarios = _load_optional_scalar_int(data, "scenarios")
        seed = _load_optional_scalar_int(data, "seed")

        payload: dict[str, object] = {
            "schema_version": schema_version,
            "boards": boards,
            "action_masks": masks,
            "teacher_actions": actions,
            "teacher_q": teacher_q,
            "teacher_policy": teacher_policy,
            "source_indexes": src,
            "board_hash": data["board_hash"].astype(np.str_, copy=False)
            if "board_hash" in data
            else board_hashes(boards),
            "episode_id": data["episode_id"].astype(np.int64, copy=False)
            if "episode_id" in data
            else np.full(boards.shape[0], -1, dtype=np.int64),
            "move_idx": data["move_idx"].astype(np.int64, copy=False)
            if "move_idx" in data
            else np.full(boards.shape[0], -1, dtype=np.int64),
            "stages": stages,
            "scenarios": scenarios,
            "seed": seed,
            "dataset_path": _load_optional_scalar_str(data, "dataset_path"),
            "policy_temperature": float(data["policy_temperature"][0])
            if "policy_temperature" in data
            else DEFAULT_POLICY_TEMPERATURE,
            "invalid_q_threshold": float(data["invalid_q_threshold"][0])
            if "invalid_q_threshold" in data
            else DEFAULT_INVALID_Q_THRESHOLD,
        }
        for name in ("teacher_value", "teacher_score_gain", "teacher_max_tile"):
            if name in data:
                payload[name] = data[name].astype(np.float32, copy=False)
        if "policy_checkpoint" in data:
            payload["policy_checkpoint"] = _load_optional_scalar_str(
                data, "policy_checkpoint"
            )
    return payload


def _load_optional_source_indexes(
    data: np.lib.npyio.NpzFile, n: int
) -> np.ndarray | None:
    src_raw = data.get("source_indexes")
    if not isinstance(src_raw, np.ndarray) or src_raw.ndim != 1:
        return None
    if src_raw.size == 0:
        return None
    src = np.asarray(src_raw, dtype=np.int64)
    if src.shape != (n,):
        raise ValueError("source_indexes length must match boards")
    return src


def _load_optional_scalar_int(
    data: np.lib.npyio.NpzFile,
    key: str,
) -> int | None:
    if key not in data:
        return None
    arr = np.asarray(data[key], dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return None
    return _none_if_negative(int(arr[0]))


def _load_optional_scalar_str(
    data: np.lib.npyio.NpzFile,
    key: str,
) -> str | None:
    if key not in data:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return None
    text = str(arr[0])
    return text if text else None
