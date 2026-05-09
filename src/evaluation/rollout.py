"""Rollout NPZ I/O for diagnostics.

Centralizes the on-disk format for collected rollout boards so that
``compare`` and any future tools share the same loader/saver.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from game2048.compare import BoardTransition, RolloutConfig


def save_rollout_npz(
    path: Path,
    transitions: list[BoardTransition],
    cfg: RolloutConfig,
) -> None:
    """Save rollout board set and config to .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    boards_data = [t.to_dict() for t in transitions]
    cfg_data = asdict(cfg)
    np.savez(
        path,
        boards=np.array(boards_data, dtype=object),
        config=np.array([cfg_data], dtype=object),
    )


def load_rollout_npz(path: Path) -> tuple[list[BoardTransition], RolloutConfig]:
    """Load rollout board set and config from .npz."""
    data = np.load(path, allow_pickle=True)

    cfg_array = data["config"]
    cfg_dict = cfg_array[0] if len(cfg_array) > 0 else {}
    cfg = RolloutConfig(
        source_checkpoint=str(cfg_dict.get("source_checkpoint", "")),
        episodes=int(cfg_dict.get("episodes", 0)),
        tail_moves=int(cfg_dict.get("tail_moves", 0)),
        seed=int(cfg_dict.get("seed", 0)),
        created_at=str(cfg_dict.get("created_at", "")),
        total_transitions=int(cfg_dict.get("total_transitions", 0)),
    )

    boards = data["boards"]
    transitions: list[BoardTransition] = []
    for board in boards:
        board_data = board["board"]
        if isinstance(board_data, np.ndarray):
            board_data = board_data.tolist()
        transitions.append(
            BoardTransition(
                game_id=int(board["game_id"]),
                move_index=int(board["move_index"]),
                score=float(board["score"]),
                max_tile=int(board["max_tile"]),
                observed_action=int(board["observed_action"]),
                observed_move=str(board["observed_move"]),
                observed_reward=float(board["observed_reward"]),
                done=bool(board["done"]),
                board=board_data,
                moves_from_episode_end=int(board.get("moves_from_episode_end", 0)),
            )
        )

    return transitions, cfg
