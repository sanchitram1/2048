"""Tests for checkpoint CLI override logic."""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from training.config import TrainConfig
from training.dqn import build_value_network
from training.env import Game2048Env
from training.train import merge_config_from_init_checkpoint, train


@pytest.fixture
def checkpoint_path() -> Path:
    """Create a temporary checkpoint file with a known config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoint_test.pt"
        
        # Checkpoint config: custom architecture and experiment settings
        checkpoint_config = TrainConfig(
            # Architecture (checkpoint-owned)
            value_network="qcnn",
            max_exponent=14,
            embedding_dim=48,
            hidden_dim=512,
            # Experiment (should be overridable)
            learning_rate=5e-4,
            gamma=0.95,
            batch_size=256,
            target_update_interval=500,
            epsilon_start=0.8,
            epsilon_end=0.02,
            epsilon_decay_steps=15_000,
            exploration="epsilon",
            grad_clip=5.0,
            # Run-control (should be overridable)
            seed=42,
            steps=100_000,
            model_dir="checkpoint_models",
            device="cpu",
            replay_capacity=50_000,
            checkpoint_interval=5_000,
            eval_interval=5_000,
            learning_starts=1_000,
            train_frequency=2,
        )
        
        payload = {
            "step": 50000,
            "episodes_completed": 100,
            "config": asdict(checkpoint_config),
            "q_network_state_dict": {},
            "target_network_state_dict": {},
            "optimizer_state_dict": {},
        }
        torch.save(payload, ckpt_path)
        yield ckpt_path


def test_checkpoint_owned_fields_preserved(checkpoint_path: Path) -> None:
    """Architecture fields always come from checkpoint, never overridden."""
    # CLI config with different architecture values
    cli_config = TrainConfig(
        value_network="qnetwork",
        max_exponent=15,
        embedding_dim=32,
        hidden_dim=256,
    )
    explicitly_provided = {
        "value_network",
        "max_exponent",
        "embedding_dim",
        "hidden_dim",
    }
    
    merged = merge_config_from_init_checkpoint(
        cli_config,
        checkpoint_path,
        explicitly_provided=explicitly_provided,
    )
    
    # Architecture fields should come from checkpoint, not CLI
    assert merged.value_network == "qcnn"
    assert merged.max_exponent == 14
    assert merged.embedding_dim == 48
    assert merged.hidden_dim == 512


def test_experiment_fields_overridden_when_explicit(checkpoint_path: Path) -> None:
    """Experiment knobs override checkpoint only when explicitly supplied."""
    # CLI config with different experiment values
    cli_config = TrainConfig(
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=128,
        target_update_interval=1_000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20_000,
        exploration="ucb",
        grad_clip=10.0,
    )
    
    # Only learning_rate and gamma explicitly provided
    explicitly_provided = {"learning_rate", "gamma"}
    
    merged = merge_config_from_init_checkpoint(
        cli_config,
        checkpoint_path,
        explicitly_provided=explicitly_provided,
    )
    
    # Explicitly provided should override
    assert merged.learning_rate == 1e-3
    assert merged.gamma == 0.99
    
    # Not explicitly provided should stay from checkpoint
    assert merged.batch_size == 256
    assert merged.target_update_interval == 500
    assert merged.epsilon_start == 0.8
    assert merged.epsilon_end == 0.02
    assert merged.epsilon_decay_steps == 15_000
    assert merged.exploration == "epsilon"
    assert merged.grad_clip == 5.0


def test_run_control_fields_overridden_when_explicit(checkpoint_path: Path) -> None:
    """Run-control fields override checkpoint only when explicitly supplied."""
    cli_config = TrainConfig(
        seed=123,
        steps=50_000,
        model_dir="new_models",
        device="mps",
        replay_capacity=100_000,
        checkpoint_interval=10_000,
        eval_interval=10_000,
        learning_starts=2_000,
        train_frequency=4,
    )
    
    # Only seed and steps explicitly provided
    explicitly_provided = {"seed", "steps"}
    
    merged = merge_config_from_init_checkpoint(
        cli_config,
        checkpoint_path,
        explicitly_provided=explicitly_provided,
    )
    
    # Explicitly provided should override
    assert merged.seed == 123
    assert merged.steps == 50_000
    
    # Not explicitly provided should stay from checkpoint
    assert merged.model_dir == "checkpoint_models"
    assert merged.device == "cpu"
    assert merged.replay_capacity == 50_000
    assert merged.checkpoint_interval == 5_000
    assert merged.eval_interval == 5_000
    assert merged.learning_starts == 1_000
    assert merged.train_frequency == 2


def test_no_explicitly_provided_all_from_checkpoint(checkpoint_path: Path) -> None:
    """When no args explicitly provided, all overridable fields come from checkpoint."""
    # CLI config (defaults)
    cli_config = TrainConfig()
    
    # Empty explicitly_provided
    explicitly_provided: set[str] = set()
    
    merged = merge_config_from_init_checkpoint(
        cli_config,
        checkpoint_path,
        explicitly_provided=explicitly_provided,
    )
    
    # All checkpoint values should be preserved
    assert merged.learning_rate == 5e-4
    assert merged.gamma == 0.95
    assert merged.batch_size == 256
    assert merged.target_update_interval == 500
    assert merged.epsilon_start == 0.8
    assert merged.epsilon_end == 0.02
    assert merged.epsilon_decay_steps == 15_000
    assert merged.exploration == "epsilon"
    assert merged.grad_clip == 5.0
    assert merged.seed == 42
    assert merged.steps == 100_000
    assert merged.model_dir == "checkpoint_models"
    assert merged.device == "cpu"
    assert merged.replay_capacity == 50_000
    assert merged.checkpoint_interval == 5_000
    assert merged.eval_interval == 5_000
    assert merged.learning_starts == 1_000
    assert merged.train_frequency == 2


def test_none_explicitly_provided_backwards_compatible(checkpoint_path: Path) -> None:
    """Passing None for explicitly_provided should work (backward compatibility)."""
    cli_config = TrainConfig(learning_rate=1e-3)
    
    # None should be treated as empty set (all from checkpoint)
    merged = merge_config_from_init_checkpoint(
        cli_config,
        checkpoint_path,
        explicitly_provided=None,
    )
    
    # Should get checkpoint values, not CLI values
    assert merged.learning_rate == 5e-4


def test_epsilon_schedule_override(checkpoint_path: Path) -> None:
    """Epsilon schedule fields (epsilon_start, epsilon_end, epsilon_decay_steps) can be overridden."""
    cli_config = TrainConfig(
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay_steps=10_000,
    )
    
    explicitly_provided = {
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_steps",
    }
    
    merged = merge_config_from_init_checkpoint(
        cli_config,
        checkpoint_path,
        explicitly_provided=explicitly_provided,
    )
    
    # Should be overridden
    assert merged.epsilon_start == 0.5
    assert merged.epsilon_end == 0.01
    assert merged.epsilon_decay_steps == 10_000


def test_exploration_strategy_override(checkpoint_path: Path) -> None:
    """Exploration strategy can be overridden."""
    # Checkpoint has epsilon, CLI specifies UCB
    cli_config = TrainConfig(exploration="ucb")
    explicitly_provided = {"exploration"}

    merged = merge_config_from_init_checkpoint(
        cli_config,
        checkpoint_path,
        explicitly_provided=explicitly_provided,
    )

    assert merged.exploration == "ucb"


def test_planner_fields_overridden_when_explicit(tmp_path: Path) -> None:
    """Planner knobs are CLI-overridable when init-from-checkpoint merges."""
    ckpt_path = tmp_path / "planner_ckpt.pt"
    checkpoint_config = TrainConfig(
        value_network="qnetwork",
        max_exponent=15,
        embedding_dim=32,
        hidden_dim=256,
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=4,
        target_update_interval=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20_000,
        exploration="epsilon",
        grad_clip=10.0,
        seed=1,
        steps=100,
        model_dir="m",
        device="cpu",
        replay_capacity=100,
        checkpoint_interval=0,
        eval_interval=0,
        learning_starts=2,
        train_frequency=1,
        planner_stages=9,
        planner_scenarios=9,
        planner_samples_per_update=9,
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
    torch.save(
        {
            "step": 1,
            "episodes_completed": 0,
            "config": asdict(checkpoint_config),
            "q_network_state_dict": q_net.state_dict(),
            "target_network_state_dict": q_net.state_dict(),
            "optimizer_state_dict": {},
        },
        ckpt_path,
    )

    cli_config = TrainConfig(
        planner_stages=1,
        planner_scenarios=2,
        planner_samples_per_update=3,
    )
    merged = merge_config_from_init_checkpoint(
        cli_config,
        ckpt_path,
        explicitly_provided={
            "planner_stages",
            "planner_scenarios",
            "planner_samples_per_update",
        },
    )
    assert merged.planner_stages == 1
    assert merged.planner_scenarios == 2
    assert merged.planner_samples_per_update == 3
    assert merged.planner_loss_weight == TrainConfig.planner_loss_weight


def test_all_experiment_fields_override(checkpoint_path: Path) -> None:
    """All experiment fields can be overridden together."""
    cli_config = TrainConfig(
        learning_rate=1e-4,
        gamma=0.98,
        batch_size=64,
        target_update_interval=2_000,
        epsilon_start=0.9,
        epsilon_end=0.01,
        epsilon_decay_steps=25_000,
        exploration="epsilon",
        grad_clip=20.0,
    )
    
    explicitly_provided = {
        "learning_rate",
        "gamma",
        "batch_size",
        "target_update_interval",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_steps",
        "exploration",
        "grad_clip",
    }
    
    merged = merge_config_from_init_checkpoint(
        cli_config,
        checkpoint_path,
        explicitly_provided=explicitly_provided,
    )
    
    # All should be from CLI
    assert merged.learning_rate == 1e-4
    assert merged.gamma == 0.98
    assert merged.batch_size == 64
    assert merged.target_update_interval == 2_000
    assert merged.epsilon_start == 0.9
    assert merged.epsilon_end == 0.01
    assert merged.epsilon_decay_steps == 25_000
    assert merged.exploration == "epsilon"
    assert merged.grad_clip == 20.0
    
    # Architecture should still be from checkpoint
    assert merged.value_network == "qcnn"
    assert merged.max_exponent == 14
    assert merged.embedding_dim == 48
    assert merged.hidden_dim == 512


def test_train_manifest_matches_merged_config_with_init_checkpoint(tmp_path: Path) -> None:
    """manifest.json train_config must reflect post-merge TrainConfig, not pre-merge CLI."""
    ckpt_path = tmp_path / "init.pt"
    checkpoint_config = TrainConfig(
        value_network="qcnn",
        max_exponent=14,
        embedding_dim=48,
        hidden_dim=512,
        learning_rate=5e-4,
        gamma=0.95,
        batch_size=256,
        target_update_interval=500,
        epsilon_start=0.8,
        epsilon_end=0.02,
        epsilon_decay_steps=15_000,
        exploration="epsilon",
        grad_clip=5.0,
        seed=42,
        steps=100_000,
        model_dir="checkpoint_models",
        device="cpu",
        replay_capacity=50_000,
        checkpoint_interval=5_000,
        eval_interval=5_000,
        learning_starts=1_000,
        train_frequency=2,
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

    model_dir = tmp_path / "run_models"
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
        learning_rate=1e-3,
        gamma=0.99,
        value_network="qnetwork",
        max_exponent=15,
        embedding_dim=32,
        hidden_dim=256,
        exploration="epsilon",
    )
    explicitly_provided = {"steps", "model_dir", "learning_rate", "gamma"}

    merged = merge_config_from_init_checkpoint(
        cli_config,
        ckpt_path,
        explicitly_provided=explicitly_provided,
    )

    train(
        cli_config,
        init_checkpoint=ckpt_path,
        explicitly_provided=explicitly_provided,
        argv=["train", "--model-dir", str(model_dir)],
    )

    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_config"] == asdict(merged)


def test_train_seeds_global_rng_with_merged_seed_after_init_checkpoint(tmp_path: Path) -> None:
    """seed_everything must run after merge so Python/Torch RNG matches checkpoint seed."""
    ckpt_path = tmp_path / "seed_ckpt.pt"
    checkpoint_config = TrainConfig(
        value_network="qcnn",
        max_exponent=14,
        embedding_dim=48,
        hidden_dim=512,
        learning_rate=5e-4,
        gamma=0.95,
        batch_size=256,
        target_update_interval=500,
        epsilon_start=0.8,
        epsilon_end=0.02,
        epsilon_decay_steps=15_000,
        exploration="epsilon",
        grad_clip=5.0,
        seed=12345,
        steps=100_000,
        model_dir="checkpoint_models",
        device="cpu",
        replay_capacity=50_000,
        checkpoint_interval=5_000,
        eval_interval=5_000,
        learning_starts=1_000,
        train_frequency=2,
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

    model_dir = tmp_path / "seed_run_models"
    cli_config = TrainConfig(
        steps=3,
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
        seed=7,
        exploration="epsilon",
    )
    explicitly_provided = {"steps", "model_dir"}

    merged = merge_config_from_init_checkpoint(
        cli_config,
        ckpt_path,
        explicitly_provided=explicitly_provided,
    )
    assert merged.seed == 12345

    with patch("training.train.seed_everything") as mock_seed:
        train(
            cli_config,
            init_checkpoint=ckpt_path,
            explicitly_provided=explicitly_provided,
            argv=["train"],
        )

    mock_seed.assert_called_once_with(12345)
