"""Tests for checkpoint CLI override logic."""
from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from training.config import TrainConfig
from training.train import merge_config_from_init_checkpoint


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
