from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Literal

ValueNetworkKind = Literal["qnetwork", "qcnn"]


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
    value_network: ValueNetworkKind = "qnetwork"
    model_dir: str = "models"
    device: str = "auto"
    exploration: str = "ucb"
    planner_samples_per_update: int = 0
    planner_stages: int = 1
    planner_scenarios: int = 5
    planner_temperature: float = 1.0
    planner_loss_weight: float = 0.0
    planner_q_sentinel_cutoff: float = -1e8


def train_config_from_dict(data: dict) -> TrainConfig:
    """Merge saved dict with current defaults (e.g. new fields missing in old checkpoints)."""
    defaults = asdict(TrainConfig())
    allowed_keys = {field.name for field in fields(TrainConfig)}
    filtered = {key: value for key, value in data.items() if key in allowed_keys}
    merged: dict = {**defaults, **filtered}
    return TrainConfig(**merged)
