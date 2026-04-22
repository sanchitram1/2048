from __future__ import annotations

import numpy as np
import torch

from training.config import TrainConfig
from training.dqn import ReplayBuffer, legal_actions_to_mask
from training.train import train


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

    train(config)

    assert (model_dir / "checkpoint_4.pt").exists()
    assert (model_dir / "checkpoint_8.pt").exists()
