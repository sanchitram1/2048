from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray


@dataclass(frozen=True)
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    next_action_masks: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(
        self,
        *,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        self._buffer.append(
            Transition(
                state=np.array(state, copy=True),
                action=int(action),
                reward=float(reward),
                next_state=np.array(next_state, copy=True),
                done=bool(done),
                next_action_mask=np.array(next_action_mask, copy=True, dtype=np.bool_),
            )
        )

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        transitions = random.sample(self._buffer, batch_size)

        states = torch.as_tensor(
            np.stack([transition.state for transition in transitions]),
            dtype=torch.long,
            device=device,
        )
        actions = torch.as_tensor(
            [transition.action for transition in transitions],
            dtype=torch.long,
            device=device,
        )
        rewards = torch.as_tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
            device=device,
        )
        next_states = torch.as_tensor(
            np.stack([transition.next_state for transition in transitions]),
            dtype=torch.long,
            device=device,
        )
        dones = torch.as_tensor(
            [transition.done for transition in transitions],
            dtype=torch.bool,
            device=device,
        )
        next_action_masks = torch.as_tensor(
            np.stack([transition.next_action_mask for transition in transitions]),
            dtype=torch.bool,
            device=device,
        )
        return ReplayBatch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            next_action_masks=next_action_masks,
        )


class QNetwork(nn.Module):
    def __init__(
        self,
        action_dim: int,
        *,
        max_exponent: int = 15,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.max_exponent = max_exponent
        self.embedding = nn.Embedding(max_exponent + 1, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(16 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        encoded = boards.long().clamp(min=0, max=self.max_exponent)
        embedded = self.embedding(encoded)
        return self.head(embedded.flatten(start_dim=1))


def legal_actions_to_mask(action_dim: int, legal_actions: list[int]) -> np.ndarray:
    mask = np.zeros(action_dim, dtype=np.bool_)
    mask[legal_actions] = True
    return mask


def mask_illegal_actions(
    q_values: torch.Tensor, action_masks: torch.Tensor
) -> torch.Tensor:
    fill_value = torch.finfo(q_values.dtype).min
    return q_values.masked_fill(~action_masks, fill_value)


def linear_epsilon(step: int, *, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    progress = min(step / decay_steps, 1.0)
    return start + progress * (end - start)
