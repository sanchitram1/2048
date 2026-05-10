"""Optimizer state handling for ``--init-from-checkpoint``."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from training.config import TrainConfig
from training.dqn import build_value_network
from training.env import Game2048Env
from training.train import merge_config_from_init_checkpoint, train


def test_init_from_checkpoint_starts_fresh_optimizer(tmp_path: Path) -> None:
    """Init checkpoints provide weights/config only; optimizer state is not resumed."""
    ckpt_path = tmp_path / "init.pt"
    checkpoint_config = TrainConfig(
        value_network="qcnn",
        max_exponent=15,
        embedding_dim=32,
        hidden_dim=256,
        learning_rate=1e-3,
        device="cpu",
    )
    env = Game2048Env()
    action_dim = env.action_space_n()
    q_net = build_value_network(
        checkpoint_config.value_network,
        action_dim,
        max_exponent=checkpoint_config.max_exponent,
        embedding_dim=checkpoint_config.embedding_dim,
        hidden_dim=checkpoint_config.hidden_dim,
    )
    optimizer = torch.optim.Adam(q_net.parameters(), lr=checkpoint_config.learning_rate)
    torch.save(
        {
            "step": 1,
            "episodes_completed": 0,
            "config": asdict(checkpoint_config),
            "q_network_state_dict": q_net.state_dict(),
            "target_network_state_dict": q_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )

    model_dir = tmp_path / "models"
    cli_config = TrainConfig(
        steps=4,
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
        learning_rate=5e-4,
    )
    explicitly_provided = {
        "steps",
        "batch_size",
        "replay_capacity",
        "learning_starts",
        "train_frequency",
        "target_update_interval",
        "checkpoint_interval",
        "eval_interval",
        "log_interval",
        "model_dir",
        "device",
        "exploration",
        "learning_rate",
    }
    merged = merge_config_from_init_checkpoint(
        cli_config,
        ckpt_path,
        explicitly_provided=explicitly_provided,
    )

    real_adam = torch.optim.Adam
    created_optimizers: list[torch.optim.Adam] = []

    def adam_factory(*args, **kwargs):
        fresh_optimizer = real_adam(*args, **kwargs)

        def fail_load_state_dict(_state_dict):
            raise AssertionError(
                "--init-from-checkpoint must not load optimizer_state_dict"
            )

        fresh_optimizer.load_state_dict = fail_load_state_dict  # type: ignore[method-assign]
        created_optimizers.append(fresh_optimizer)
        return fresh_optimizer

    with patch("training.train.torch.optim.Adam", side_effect=adam_factory):
        train(
            cli_config,
            init_checkpoint=ckpt_path,
            explicitly_provided=explicitly_provided,
            argv=["train", "--model-dir", str(model_dir)],
        )

    assert created_optimizers
    assert created_optimizers[0].param_groups[0]["lr"] == merged.learning_rate
