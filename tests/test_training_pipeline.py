from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

from training.config import TrainConfig
from training.dqn import ReplayBuffer, legal_actions_to_mask
from training.train import ModelDirNotEmptyError, train


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
