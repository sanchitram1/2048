from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 50_000
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_starts: int = 2_000
    train_frequency: int = 4
    target_update_interval: int = 1_000
    checkpoint_interval: int = 10_000
    eval_interval: int = 10_000
    eval_episodes: int = 5
    log_interval: int = 1_000
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    grad_clip: float = 10.0
    seed: int = 7
    max_exponent: int = 15
    embedding_dim: int = 32
    hidden_dim: int = 256
    model_dir: str = "models"
    device: str = "auto"
