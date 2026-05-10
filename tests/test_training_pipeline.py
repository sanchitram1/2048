from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from unittest.mock import patch

from training.config import TrainConfig
from training.dqn import ReplayBuffer, Transition, legal_actions_to_mask
from training.train import ModelDirNotEmptyError, planner_distillation_loss, train


def test_replay_buffer_sample_shapes() -> None:
    buffer = ReplayBuffer(capacity=8)
    action_dim = 4

    for action in range(4):
        state = np.full((4, 4), action, dtype=np.int16)
        next_state = state + 1
        buffer.add(
            state=state,
            action=action,
            reward=float(action),
            next_state=next_state,
            done=action == 3,
            next_action_mask=legal_actions_to_mask(action_dim, [0, 2]),
        )

    batch = buffer.sample(batch_size=4, device=torch.device("cpu"))

    assert batch.states.shape == (4, 4, 4)
    assert batch.actions.shape == (4,)
    assert batch.rewards.shape == (4,)
    assert batch.next_states.shape == (4, 4, 4)
    assert batch.next_action_masks.shape == (4, action_dim)
    assert batch.states.dtype == torch.long
    assert batch.next_action_masks.dtype == torch.bool


def test_train_writes_checkpoints(tmp_path) -> None:
    model_dir = tmp_path / "models"
    config = TrainConfig(
        steps=8,
        batch_size=2,
        replay_capacity=16,
        learning_starts=2,
        train_frequency=1,
        target_update_interval=4,
        checkpoint_interval=4,
        eval_interval=0,
        log_interval=0,
        model_dir=str(model_dir),
        device="cpu",
    )

    train(config, argv=["train", "--model-dir", str(model_dir)])

    assert (model_dir / "checkpoint_4.pt").exists()
    assert (model_dir / "checkpoint_8.pt").exists()
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_config"]["steps"] == 8
    assert not (model_dir / "metrics.jsonl").exists()


def test_train_writes_manifest_and_metrics_jsonl(tmp_path) -> None:
    model_dir = tmp_path / "models"
    config = TrainConfig(
        steps=25,
        batch_size=2,
        replay_capacity=32,
        learning_starts=2,
        train_frequency=1,
        target_update_interval=100,
        checkpoint_interval=0,
        eval_interval=10,
        eval_episodes=1,
        log_interval=10,
        model_dir=str(model_dir),
        device="cpu",
        exploration="epsilon",
    )
    argv = ["train", "--model-dir", str(model_dir)]

    train(config, argv=argv, explicitly_provided={"model_dir"})

    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["argv"] == argv
    assert manifest["device"] == "cpu"
    assert manifest["train_config"] == asdict(config)
    assert manifest["explicitly_provided_cli_keys"] == ["model_dir"]

    lines = (model_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["schema_version"] == 1
    assert row["step"] == 10
    assert "epsilon" in row
    assert "replay_buffer_size" in row
    assert row["episodes_completed"] >= 0
    assert row["eval"] is not None
    for key in (
        "mean_score",
        "median_score",
        "score_variance",
        "mean_max_tile",
        "max_score",
        "reach_256",
        "reach_512",
        "reach_1024",
        "reach_2048",
    ):
        assert key in row["eval"]

    assert "dqn_loss_mean_last_100" in row
    assert "planner_loss_mean_last_100" in row
    assert "total_loss_mean_last_100" in row
    assert row["total_loss_mean_last_100"] == row["train_loss_mean_last_100"]
    assert "planner_rows_attempted_since_last_log" in row
    assert "planner_rows_used_since_last_log" in row
    assert "planner_calls_since_last_log" in row
    assert "planner_seconds_since_last_log" in row


def test_sample_transitions_and_batch_matches_stacked_tensors() -> None:
    buffer = ReplayBuffer(capacity=8)
    action_dim = 4
    for action in range(4):
        state = np.full((4, 4), action, dtype=np.int16)
        next_state = state + 1
        buffer.add(
            state=state,
            action=action,
            reward=float(action),
            next_state=next_state,
            done=action == 3,
            next_action_mask=legal_actions_to_mask(action_dim, [0, 2]),
        )
    transitions, batch = buffer.sample_transitions_and_batch(4, torch.device("cpu"))
    stacked = torch.as_tensor(
        np.stack([t.state for t in transitions]), dtype=torch.long, device="cpu"
    )
    assert torch.equal(batch.states, stacked)


def test_train_planner_disabled_does_not_call_mc(tmp_path) -> None:
    model_dir = tmp_path / "models"
    config = TrainConfig(
        steps=6,
        batch_size=2,
        replay_capacity=16,
        learning_starts=2,
        train_frequency=1,
        target_update_interval=100,
        checkpoint_interval=0,
        eval_interval=0,
        log_interval=0,
        model_dir=str(model_dir),
        device="cpu",
        exploration="epsilon",
        planner_samples_per_update=2,
        planner_loss_weight=0.0,
    )
    with patch("training.train.choose_n_step_mc") as mock_mc:
        train(config, argv=["train", "--model-dir", str(model_dir)])
    mock_mc.assert_not_called()


def test_train_rejects_negative_planner_samples(tmp_path) -> None:
    model_dir = tmp_path / "models"
    config = TrainConfig(
        steps=2,
        batch_size=2,
        replay_capacity=8,
        learning_starts=1,
        train_frequency=1,
        model_dir=str(model_dir),
        device="cpu",
        exploration="epsilon",
        planner_samples_per_update=-1,
    )
    with pytest.raises(ValueError, match="planner_samples_per_update"):
        train(config, argv=["train", "--model-dir", str(model_dir)])


def test_train_rejects_negative_planner_loss_weight(tmp_path) -> None:
    model_dir = tmp_path / "models"
    config = TrainConfig(
        steps=2,
        batch_size=2,
        replay_capacity=8,
        learning_starts=1,
        train_frequency=1,
        model_dir=str(model_dir),
        device="cpu",
        exploration="epsilon",
        planner_samples_per_update=1,
        planner_loss_weight=-0.01,
    )
    with pytest.raises(ValueError, match="planner_loss_weight"):
        train(config, argv=["train", "--model-dir", str(model_dir)])


def test_train_rejects_non_positive_planner_temperature_when_planner_on(
    tmp_path,
) -> None:
    model_dir = tmp_path / "models"
    config = TrainConfig(
        steps=2,
        batch_size=2,
        replay_capacity=8,
        learning_starts=1,
        train_frequency=1,
        model_dir=str(model_dir),
        device="cpu",
        exploration="epsilon",
        planner_samples_per_update=1,
        planner_loss_weight=0.01,
        planner_temperature=0.0,
    )
    with pytest.raises(ValueError, match="planner_temperature"):
        train(config, argv=["train", "--model-dir", str(model_dir)])


def test_train_rejects_non_positive_planner_stages_when_planner_on(tmp_path) -> None:
    model_dir = tmp_path / "models"
    config = TrainConfig(
        steps=2,
        batch_size=2,
        replay_capacity=8,
        learning_starts=1,
        train_frequency=1,
        model_dir=str(model_dir),
        device="cpu",
        exploration="epsilon",
        planner_samples_per_update=1,
        planner_loss_weight=0.01,
        planner_stages=0,
    )
    with pytest.raises(ValueError, match="planner_stages"):
        train(config, argv=["train", "--model-dir", str(model_dir)])


def test_train_rejects_non_positive_planner_scenarios_when_planner_on(
    tmp_path,
) -> None:
    model_dir = tmp_path / "models"
    config = TrainConfig(
        steps=2,
        batch_size=2,
        replay_capacity=8,
        learning_starts=1,
        train_frequency=1,
        model_dir=str(model_dir),
        device="cpu",
        exploration="epsilon",
        planner_samples_per_update=1,
        planner_loss_weight=0.01,
        planner_scenarios=0,
    )
    with pytest.raises(ValueError, match="planner_scenarios"):
        train(config, argv=["train", "--model-dir", str(model_dir)])


def test_train_planner_smoke_finishes(tmp_path) -> None:
    model_dir = tmp_path / "models"
    config = TrainConfig(
        steps=8,
        batch_size=2,
        replay_capacity=16,
        learning_starts=2,
        train_frequency=1,
        target_update_interval=100,
        checkpoint_interval=0,
        eval_interval=0,
        log_interval=0,
        model_dir=str(model_dir),
        device="cpu",
        exploration="epsilon",
        planner_samples_per_update=1,
        planner_stages=1,
        planner_scenarios=1,
        planner_temperature=1.0,
        planner_loss_weight=0.01,
    )
    train(config, argv=["train", "--model-dir", str(model_dir)])
    assert (model_dir / "checkpoint_8.pt").exists()


def test_planner_distillation_skips_no_legal_replay_row() -> None:
    dead_board = np.array(
        [
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [1, 2, 1, 2],
            [2, 1, 2, 1],
        ],
        dtype=np.int16,
    )
    transition = Transition(
        state=dead_board,
        action=0,
        reward=0.0,
        next_state=dead_board,
        done=True,
        next_action_mask=np.zeros(4, dtype=np.bool_),
    )
    stats = {
        "planner_rows_attempted": 0.0,
        "planner_rows_used": 0.0,
        "planner_calls": 0.0,
        "planner_seconds": 0.0,
    }
    config = TrainConfig(planner_samples_per_update=1, planner_loss_weight=0.01)
    student_q = torch.zeros((1, 4), requires_grad=True)

    with patch("training.train.choose_n_step_mc") as mock_mc:
        loss = planner_distillation_loss(
            transitions=[transition],
            student_q_all=student_q,
            config=config,
            device=torch.device("cpu"),
            train_update_index=0,
            stats=stats,
        )

    mock_mc.assert_not_called()
    assert loss.item() == 0.0
    assert stats["planner_rows_attempted"] == 1.0
    assert stats["planner_rows_used"] == 0.0
    assert stats["planner_calls"] == 0.0


def test_train_rejects_nonempty_model_dir(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "stale.txt").write_text("x", encoding="utf-8")
    config = TrainConfig(
        steps=2,
        batch_size=2,
        replay_capacity=8,
        learning_starts=1,
        train_frequency=1,
        target_update_interval=10,
        checkpoint_interval=0,
        eval_interval=0,
        log_interval=0,
        model_dir=str(model_dir),
        device="cpu",
    )
    with pytest.raises(ModelDirNotEmptyError, match="not empty"):
        train(config)
