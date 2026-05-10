from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unittest.mock import patch

from training.config import TrainConfig
from training.dqn import ReplayBatch, ReplayBuffer, Transition, legal_actions_to_mask
from training.train import (
    ModelDirNotEmptyError,
    apply_q_output_affine,
    compute_td_loss,
    planner_distillation_loss,
    train,
)


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


def test_compute_td_loss_applies_reward_bootstrap_and_loss_scales() -> None:
    class ConstantQ(torch.nn.Module):
        def __init__(self, values: list[float]) -> None:
            super().__init__()
            self.register_buffer("values", torch.tensor([values], dtype=torch.float32))

        def forward(self, boards: torch.Tensor) -> torch.Tensor:
            return self.values.repeat(boards.shape[0], 1)

    batch = ReplayBatch(
        states=torch.zeros((1, 4, 4), dtype=torch.long),
        actions=torch.tensor([0], dtype=torch.long),
        rewards=torch.tensor([2.0], dtype=torch.float32),
        next_states=torch.zeros((1, 4, 4), dtype=torch.long),
        dones=torch.tensor([False]),
        next_action_masks=torch.tensor([[True, True, False, False]]),
    )

    loss = compute_td_loss(
        batch=batch,
        online_network=ConstantQ([5.0, 10.0, 0.0, 0.0]),
        target_network=ConstantQ([0.0, 10.0, 0.0, 0.0]),
        gamma=0.5,
        td_reward_scale=3.0,
        td_bootstrap_scale=0.2,
        td_loss_scale=2.0,
    )

    # target = 2*3 + 0.5*0.2*10 = 7; current = 5; normalized diff = 1.
    assert loss.item() == pytest.approx(0.5)


def test_apply_q_output_affine_updates_final_linear_only() -> None:
    network = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 4),
    )
    first_weight_before = network[0].weight.detach().clone()
    with torch.no_grad():
        network[2].weight.fill_(2.0)
        network[2].bias.fill_(3.0)

    apply_q_output_affine(network, scale=0.5, shift=-1.0)

    torch.testing.assert_close(network[0].weight, first_weight_before)
    torch.testing.assert_close(network[2].weight, torch.full_like(network[2].weight, 1.0))
    torch.testing.assert_close(network[2].bias, torch.full_like(network[2].bias, 0.5))


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
    assert "planner_rows_skipped_sentinel_saturation_since_last_log" in row
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


def test_planner_distillation_skips_saturated_sentinel_like_teacher() -> None:
    board = np.zeros((4, 4), dtype=np.int16)
    board[0, 0] = 1
    board[0, 1] = 1
    transition = Transition(
        state=board,
        action=0,
        reward=0.0,
        next_state=board,
        done=False,
        next_action_mask=np.ones(4, dtype=np.bool_),
    )
    stats = {
        "planner_rows_attempted": 0.0,
        "planner_rows_used": 0.0,
        "planner_rows_skipped_sentinel_saturation": 0.0,
        "planner_calls": 0.0,
        "planner_seconds": 0.0,
    }
    config = TrainConfig(
        planner_samples_per_update=1,
        planner_loss_weight=0.01,
        planner_temperature=1.0,
    )
    student_q = torch.zeros((1, 4), requires_grad=True)
    planned = SimpleNamespace(
        q_values=(-1.0e6, -8.0e5, -1.0e9, -1.0e9),
        legal_actions=(0, 1),
    )

    with (
        patch("training.train.choose_n_step_mc", return_value=planned),
        patch("training.train._LOG.warning") as mock_warning,
    ):
        loss = planner_distillation_loss(
            transitions=[transition],
            student_q_all=student_q,
            config=config,
            device=torch.device("cpu"),
            train_update_index=0,
            stats=stats,
        )

    assert loss.item() == 0.0
    assert stats["planner_rows_attempted"] == 1.0
    assert stats["planner_rows_used"] == 0.0
    assert stats["planner_rows_skipped_sentinel_saturation"] == 1.0
    assert "[WARNING] skipping planner row" in mock_warning.call_args.args[0]


def test_planner_distillation_best_action_margin_loss() -> None:
    board = np.zeros((4, 4), dtype=np.int16)
    board[0, 0] = 1
    board[0, 1] = 1
    transition = Transition(
        state=board,
        action=0,
        reward=0.0,
        next_state=board,
        done=False,
        next_action_mask=np.ones(4, dtype=np.bool_),
    )
    stats = {
        "planner_rows_attempted": 0.0,
        "planner_rows_used": 0.0,
        "planner_rows_skipped_sentinel_saturation": 0.0,
        "planner_calls": 0.0,
        "planner_seconds": 0.0,
    }
    config = TrainConfig(planner_samples_per_update=1, planner_loss_weight=0.01)
    student_q = torch.tensor([[0.0, 0.25, 0.0, 0.0]], requires_grad=True)
    planned = SimpleNamespace(
        q_values=(10.0, 5.0, -1.0e9, -1.0e9),
        legal_actions=(0, 1),
    )

    with patch("training.train.choose_n_step_mc", return_value=planned):
        loss = planner_distillation_loss(
            transitions=[transition],
            student_q_all=student_q,
            config=config,
            device=torch.device("cpu"),
            train_update_index=0,
            stats=stats,
        )

    assert loss.item() == pytest.approx(1.25)
    assert stats["planner_rows_used"] == 1.0


def test_planner_distillation_skips_low_teacher_gap() -> None:
    board = np.zeros((4, 4), dtype=np.int16)
    board[0, 0] = 1
    board[0, 1] = 1
    transition = Transition(
        state=board,
        action=0,
        reward=0.0,
        next_state=board,
        done=False,
        next_action_mask=np.ones(4, dtype=np.bool_),
    )
    stats = {
        "planner_rows_attempted": 0.0,
        "planner_rows_used": 0.0,
        "planner_rows_skipped_sentinel_saturation": 0.0,
        "planner_rows_skipped_teacher_gap": 0.0,
        "planner_calls": 0.0,
        "planner_seconds": 0.0,
    }
    config = TrainConfig(
        planner_samples_per_update=1,
        planner_loss_weight=0.01,
        planner_min_teacher_gap=8.0,
    )
    student_q = torch.tensor([[0.0, 0.25, 0.0, 0.0]], requires_grad=True)
    planned = SimpleNamespace(
        q_values=(10.0, 5.0, -1.0e9, -1.0e9),
        legal_actions=(0, 1),
    )

    with patch("training.train.choose_n_step_mc", return_value=planned):
        loss = planner_distillation_loss(
            transitions=[transition],
            student_q_all=student_q,
            config=config,
            device=torch.device("cpu"),
            train_update_index=0,
            stats=stats,
        )

    assert loss.item() == 0.0
    assert stats["planner_rows_used"] == 0.0
    assert stats["planner_rows_skipped_teacher_gap"] == 1.0


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
