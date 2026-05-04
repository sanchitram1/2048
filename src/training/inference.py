from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from training.config import TrainConfig, train_config_from_dict
from training.dqn import (
    build_value_network,
    legal_actions_to_mask,
    mask_illegal_actions,
)
from training.env import Game2048Env
from training.train import resolve_device


CHECKPOINT_PATTERN = re.compile(r"checkpoint_(\d+)\.pt$")
ACTION_NAMES = {0: "left", 1: "right", 2: "up", 3: "down"}


@dataclass(frozen=True)
class ModelAction:
    action: int
    move: str
    q_values: tuple[float, ...]
    legal_actions: tuple[int, ...]


def find_latest_checkpoint(model_dir: str | Path = "models") -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in Path(model_dir).glob("checkpoint_*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _config_from_checkpoint(checkpoint: dict[str, object]) -> TrainConfig:
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        return TrainConfig()
    return train_config_from_dict(raw_config)


def load_q_network(
    checkpoint_path: str | Path,
    *,
    device_name: str = "cpu",
) -> tuple[nn.Module, TrainConfig, torch.device]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = _config_from_checkpoint(checkpoint)

    action_dim = Game2048Env.action_space_n()
    q_network = build_value_network(
        config.value_network,
        action_dim,
        max_exponent=config.max_exponent,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    q_network.load_state_dict(checkpoint["q_network_state_dict"])
    q_network.eval()
    return q_network, config, device


def choose_greedy_action(
    *,
    q_network: nn.Module,
    state,
    legal_actions: list[int],
    device: torch.device,
) -> ModelAction:
    action_dim = Game2048Env.action_space_n()
    action_mask = torch.as_tensor(
        legal_actions_to_mask(action_dim, legal_actions),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    state_tensor = torch.as_tensor(state, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = q_network(state_tensor)
        masked_q_values = mask_illegal_actions(q_values, action_mask)
        action = int(masked_q_values.argmax(dim=1).item())

    return ModelAction(
        action=action,
        move=ACTION_NAMES[action],
        q_values=tuple(float(value) for value in q_values.squeeze(0).tolist()),
        legal_actions=tuple(int(action) for action in legal_actions),
    )


class GreedyAgentRunner:
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device_name: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.q_network, self.config, self.device = load_q_network(
            self.checkpoint_path,
            device_name=device_name,
        )
        self.env = Game2048Env()
        self.env.seed(self.config.seed)
        self.move_count = 0
        self.state, self.info = self.env.reset()

    def reset(self) -> dict[str, object]:
        self.move_count = 0
        self.state, self.info = self.env.reset()
        return self.payload(event="state", model_action=None)

    def step(self) -> dict[str, object]:
        legal_actions = self.env.legal_actions()
        if not legal_actions:
            return self.payload(event="game_over", model_action=None)

        model_action = choose_greedy_action(
            q_network=self.q_network,
            state=self.state,
            legal_actions=legal_actions,
            device=self.device,
        )
        self.state, _reward, done, truncated, self.info = self.env.step(
            model_action.action
        )
        self.move_count += 1
        event = "game_over" if done or truncated else "agent_move"
        return self.payload(event=event, model_action=model_action)

    def payload(
        self,
        *,
        event: str,
        model_action: ModelAction | None,
    ) -> dict[str, object]:
        max_tile = int(self.info["max_tile"])
        payload: dict[str, object] = {
            "event": event,
            "tiles": [int(tile) for tile in self.state.flatten()],
            "score": int(self.info["score"]),
            "move_count": self.move_count,
            "done": bool(self.info["done"]),
            "max_tile": max_tile,
            "checkpoint": str(self.checkpoint_path),
            "model_type": "dqn",
        }
        if model_action is not None:
            payload.update(
                {
                    "move": model_action.move,
                    "action": model_action.action,
                    "q_values": list(model_action.q_values),
                    "legal_actions": list(model_action.legal_actions),
                }
            )
        return payload
