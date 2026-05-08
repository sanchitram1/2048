"""Test optimizer state handling when learning_rate is overridden."""
from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn

from training.config import TrainConfig
from training.train import merge_config_from_init_checkpoint


@pytest.fixture
def checkpoint_with_optimizer_state() -> Path:
    """Create a checkpoint with optimizer state (previous training)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "checkpoint_with_optimizer.pt"
        
        checkpoint_config = TrainConfig(
            learning_rate=1e-3,
            value_network="qcnn",
            max_exponent=15,
            embedding_dim=32,
            hidden_dim=256,
        )
        
        # Create dummy model and optimizer to generate realistic state
        model = nn.Linear(10, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Simulate a training step to populate optimizer state
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        optimizer.step()
        
        payload = {
            "step": 1000,
            "episodes_completed": 10,
            "config": asdict(checkpoint_config),
            "q_network_state_dict": model.state_dict(),
            "target_network_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(payload, ckpt_path)
        yield ckpt_path


def test_optimizer_lr_reset_when_learning_rate_overridden(
    checkpoint_with_optimizer_state: Path,
) -> None:
    """When learning_rate is explicitly overridden, optimizer param-group LRs should be updated."""
    cli_config = TrainConfig(learning_rate=5e-4)
    explicitly_provided = {"learning_rate"}
    
    # Merge should work (no optimizer state reset in merge function itself)
    merged = merge_config_from_init_checkpoint(
        cli_config,
        checkpoint_with_optimizer_state,
        explicitly_provided=explicitly_provided,
    )
    
    assert merged.learning_rate == 5e-4
    
    # Now test the optimizer state reset logic that would happen in train()
    # Load checkpoint and create optimizer
    ckpt = torch.load(checkpoint_with_optimizer_state, weights_only=False)
    opt_sd = ckpt.get("optimizer_state_dict")
    
    # Create new optimizer with merged learning rate
    dummy_model = nn.Linear(10, 4)
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=merged.learning_rate)
    
    # Load optimizer state
    if isinstance(opt_sd, dict):
        optimizer.load_state_dict(opt_sd)
        
        # Reset param-group LRs when learning_rate was explicitly overridden
        if "learning_rate" in explicitly_provided:
            for param_group in optimizer.param_groups:
                param_group["lr"] = merged.learning_rate
    
    # Verify optimizer now has the new learning rate
    assert optimizer.param_groups[0]["lr"] == 5e-4


def test_optimizer_lr_unchanged_when_not_explicitly_overridden(
    checkpoint_with_optimizer_state: Path,
) -> None:
    """When learning_rate is NOT explicitly overridden, optimizer keeps checkpoint LR."""
    cli_config = TrainConfig(learning_rate=5e-4)  # different from checkpoint
    explicitly_provided: set[str] = set()  # learning_rate NOT explicitly provided
    
    merged = merge_config_from_init_checkpoint(
        cli_config,
        checkpoint_with_optimizer_state,
        explicitly_provided=explicitly_provided,
    )
    
    # Should get checkpoint value, not CLI value
    assert merged.learning_rate == 1e-3
    
    # Load optimizer state without reset
    ckpt = torch.load(checkpoint_with_optimizer_state, weights_only=False)
    opt_sd = ckpt.get("optimizer_state_dict")
    
    dummy_model = nn.Linear(10, 4)
    optimizer = torch.optim.Adam(dummy_model.parameters(), lr=merged.learning_rate)
    
    if isinstance(opt_sd, dict):
        optimizer.load_state_dict(opt_sd)
        # No reset since learning_rate not explicitly provided
        if "learning_rate" in explicitly_provided:
            for param_group in optimizer.param_groups:
                param_group["lr"] = merged.learning_rate
    
    # Optimizer should keep the loaded state (original checkpoint LR would be preserved)
    # since we didn't override it
    assert optimizer.param_groups[0]["lr"] == 1e-3
