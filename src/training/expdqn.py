from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn


# =========================
# ORIGINAL STRUCTURES (UNCHANGED)
# =========================
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


# =========================
# UTILITIES
# =========================
def state_key(s: np.ndarray):
    return tuple(s.flatten())


def board_to_exponent(board: np.ndarray):
    exp = np.zeros_like(board)
    nz = board > 0
    exp[nz] = np.log2(board[nz]).astype(int)
    return exp


def is_new_game(s_prev, s_curr):
    return np.count_nonzero(s_curr) <= 2


def infer_action(s, s_next):
    for a in range(4):
        moved, _ = apply_move(s, a)
        diff = s_next - moved
        if np.count_nonzero(diff) == 1:
            return a
    return None


# =========================
# DATASET BUILDER
# =========================
class ExpertDatasetBuilder:
    def __init__(self, gamma: float = 0.95):
        self.gamma = gamma

    def load_boards(self, path: str):
        return np.load(path)

    def build_episodes(self, boards):
        episodes = []
        current = []

        for i in range(len(boards) - 1):
            s = boards[i]
            s_next = boards[i + 1]

            if is_new_game(s, s_next):
                if current:
                    episodes.append(current)
                current = []
                continue

            a = infer_action(s, s_next)
            if a is None:
                continue

            _, r = apply_move(s, a)
            current.append((s, a, r, s_next))

        if current:
            episodes.append(current)

        return episodes

    def monte_carlo(self, episodes):
        mc = []
        for ep in episodes:
            G = 0
            for s, a, r, _ in reversed(ep):
                G = r + self.gamma * G
                mc.append((s, a, G))
        return mc

    def build_transitions(self, episodes):
        transitions = []
        for ep in episodes:
            for i in range(len(ep) - 1):
                s, a, r, s_next = ep[i]
                transitions.append((s, a, r, s_next))
        return transitions

    def build_q_table(self, mc_dataset, transitions, passes=5):
        Q = defaultdict(lambda: np.full(4, -1e6))

        # initialize
        for s, a, G in mc_dataset:
            Q[state_key(s)][a] = G

        # bootstrap
        for _ in range(passes):
            for s, a, r, s_next in transitions:
                key = state_key(s)
                next_key = state_key(s_next)
                Q[key][a] = r + self.gamma * np.max(Q[next_key])

        return Q

    def build_training_tensors(self, Q):
        X, Y = [], []

        for s_key, q_vals in Q.items():
            s = np.array(s_key).reshape(4, 4)
            X.append(board_to_exponent(s))
            Y.append(q_vals)

        X = torch.tensor(np.array(X), dtype=torch.long)
        Y = torch.tensor(np.array(Y), dtype=torch.float32)

        return X, Y


# =========================
# PRETRAINER
# =========================
class QPretrainer:
    def __init__(self, model: QNetwork, device=None):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def train(self, X, Y, epochs=10, batch_size=128, lr=1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            perm = torch.randperm(len(X))
            total_loss = 0

            for i in range(0, len(X), batch_size):
                idx = perm[i : i + batch_size]

                states = X[idx].to(self.device)
                target_q = Y[idx].to(self.device)

                pred_q = self.model(states)
                loss = loss_fn(pred_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()


        return self.model


# =========================
# PIPELINE FUNCTION
# =========================
def run_pretraining_pipeline(board_file="boards_dataset.npy"):
    builder = ExpertDatasetBuilder(gamma=0.95)

    boards = builder.load_boards(board_file)
    episodes = builder.build_episodes(boards)

    mc = builder.monte_carlo(episodes)
    transitions = builder.build_transitions(episodes)

    Q = builder.build_q_table(mc, transitions)

    X, Y = builder.build_training_tensors(Q)

    net = QNetwork(action_dim=4)
    trainer = QPretrainer(net)

    trained_model = trainer.train(X, Y)

    return trained_model, Q