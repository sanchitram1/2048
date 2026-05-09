"""Round-trip test for rollout NPZ I/O."""

from __future__ import annotations

from pathlib import Path

from evaluation.rollout import load_rollout_npz, save_rollout_npz
from game2048.compare import BoardTransition, RolloutConfig


def test_round_trip_npz(tmp_path: Path) -> None:
    cfg = RolloutConfig(
        source_checkpoint="models/dummy.pt",
        episodes=2,
        tail_moves=3,
        seed=42,
        created_at="2026-05-09T00:00:00+00:00",
        total_transitions=2,
    )
    transitions = [
        BoardTransition(
            game_id=0,
            move_index=4,
            score=120.0,
            max_tile=64,
            observed_action=1,
            observed_move="r",
            observed_reward=4.0,
            done=False,
            board=[[0] * 4 for _ in range(4)],
            moves_from_episode_end=2,
        ),
        BoardTransition(
            game_id=0,
            move_index=5,
            score=124.0,
            max_tile=64,
            observed_action=2,
            observed_move="u",
            observed_reward=0.0,
            done=True,
            board=[[1, 0, 0, 0] for _ in range(4)],
            moves_from_episode_end=0,
        ),
    ]

    path = tmp_path / "rollout.npz"
    save_rollout_npz(path, transitions, cfg)
    assert path.exists()

    loaded_transitions, loaded_cfg = load_rollout_npz(path)
    assert loaded_cfg == cfg
    assert loaded_transitions == transitions
