from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

import numpy as np

from game2048.game import GameLogic


ACTION_TO_MOVE = {0: "l", 1: "r", 2: "u", 3: "d"}
ACTION_NAMES = {0: "left", 1: "right", 2: "up", 3: "down"}


@dataclass(frozen=True)
class PlannedAction:
    action: int
    move: str
    q_values: tuple[float, ...]
    legal_actions: tuple[int, ...]


def _spawn_on_grid(
    grid: np.ndarray,
    *,
    rng: random.Random,
    spawn_exponents: tuple[int, int] = (1, 2),
    spawn_probs: tuple[float, float] = (0.9, 0.1),
) -> np.ndarray:
    open_positions = list(zip(*np.where(grid == 0)))
    if not open_positions:
        return grid
    row_idx, col_idx = rng.choice(open_positions)
    exponent = rng.choices(spawn_exponents, weights=spawn_probs, k=1)[0]
    next_grid = grid.copy()
    next_grid[row_idx, col_idx] = int(exponent)
    return next_grid


def choose_myopic_greedy(game: GameLogic, *, rng: random.Random) -> PlannedAction:
    """Pick the legal move with the largest immediate merge score gain."""
    q_values = np.full(4, -1.0e9, dtype=np.float32)
    legal_actions: list[int] = []

    for action, move in ACTION_TO_MOVE.items():
        _after, score_gain, moved = game.preview_move(move)
        if not moved:
            continue
        legal_actions.append(action)
        q_values[action] = float(score_gain)

    if not legal_actions:
        raise RuntimeError("No legal actions available")

    best_value = max(float(q_values[action]) for action in legal_actions)
    best_actions = [
        action
        for action in legal_actions
        if np.isclose(float(q_values[action]), best_value)
    ]
    chosen = int(rng.choice(best_actions))
    return PlannedAction(
        action=chosen,
        move=ACTION_NAMES[chosen],
        q_values=tuple(float(value) for value in q_values),
        legal_actions=tuple(int(action) for action in legal_actions),
    )


def _rollout_sequence_expected_value(
    game: GameLogic,
    start_grid: np.ndarray,
    sequence: tuple[int, ...],
    *,
    scenarios: int,
    rng: random.Random,
) -> float:
    """Monte Carlo EV of cumulative merge score over a fixed action sequence."""
    totals: list[float] = []
    for _ in range(scenarios):
        grid = start_grid.copy()
        total = 0.0
        feasible = True
        for action in sequence:
            move = ACTION_TO_MOVE[int(action)]
            next_grid, score_gain, moved = game.preview_move_on_grid(grid, move)
            if not moved:
                feasible = False
                break
            total += float(score_gain)
            grid = _spawn_on_grid(next_grid, rng=rng)
        totals.append(total if feasible else -1.0e6)
    return float(np.mean(totals))


def choose_n_step_mc(
    game: GameLogic,
    *,
    stages: int = 3,
    scenarios: int = 20,
    rng: random.Random,
) -> PlannedAction:
    """Choose action by maximizing MC-approximated N-step expected merge score."""
    if stages <= 0:
        raise ValueError("stages must be positive")
    if scenarios <= 0:
        raise ValueError("scenarios must be positive")

    start_grid = game.get_board()
    legal_moves = game.available_moves()
    legal_actions = tuple(
        sorted(action for action, move in ACTION_TO_MOVE.items() if move in legal_moves)
    )
    if not legal_actions:
        raise RuntimeError("No legal actions available")

    q_values = np.full(4, -1.0e9, dtype=np.float32)
    for action in legal_actions:
        remaining = tuple(range(4))
        sequences = itertools.product(remaining, repeat=stages - 1)
        best_for_action = -1.0e9
        for suffix in sequences:
            seq = (int(action),) + tuple(int(item) for item in suffix)
            value = _rollout_sequence_expected_value(
                game, start_grid, seq, scenarios=scenarios, rng=rng
            )
            if value > best_for_action:
                best_for_action = value
        q_values[int(action)] = float(best_for_action)

    best_value = max(float(q_values[action]) for action in legal_actions)
    best_actions = [
        action
        for action in legal_actions
        if np.isclose(float(q_values[action]), best_value)
    ]
    chosen = int(rng.choice(best_actions))
    return PlannedAction(
        action=chosen,
        move=ACTION_NAMES[chosen],
        q_values=tuple(float(value) for value in q_values),
        legal_actions=legal_actions,
    )


class MyopicGreedyRunner:
    def __init__(self, *, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.game = GameLogic()
        self.move_count = 0

    def reset(self) -> dict[str, object]:
        self.game = GameLogic()
        self.move_count = 0
        return self.payload(event="state", model_action=None)

    def step(self) -> dict[str, object]:
        if self.game.done or not self.game.available_moves():
            return self.payload(event="game_over", model_action=None)
        model_action = choose_myopic_greedy(self.game, rng=self.rng)
        self.game.make_move(ACTION_TO_MOVE[model_action.action])
        self.move_count += 1
        return self.payload(
            event="game_over" if self.game.done else "agent_move",
            model_action=model_action,
        )

    def payload(
        self, *, event: str, model_action: PlannedAction | None
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": event,
            "tiles": [int(tile) for tile in self.game.get_board().flatten()],
            "score": self.game.get_score(),
            "move_count": self.move_count,
            "done": self.game.done,
            "max_tile": self.game.max_square(),
            "checkpoint": None,
            "model_type": "greedy_myopic",
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


class NStepMCRunner:
    def __init__(self, *, stages: int = 3, scenarios: int = 20, seed: int = 7) -> None:
        self.stages = int(stages)
        self.scenarios = int(scenarios)
        self.rng = random.Random(seed)
        self.game = GameLogic()
        self.move_count = 0

    def reset(self) -> dict[str, object]:
        self.game = GameLogic()
        self.move_count = 0
        return self.payload(event="state", model_action=None)

    def step(self) -> dict[str, object]:
        if self.game.done or not self.game.available_moves():
            return self.payload(event="game_over", model_action=None)
        model_action = choose_n_step_mc(
            self.game, stages=self.stages, scenarios=self.scenarios, rng=self.rng
        )
        self.game.make_move(ACTION_TO_MOVE[model_action.action])
        self.move_count += 1
        return self.payload(
            event="game_over" if self.game.done else "agent_move",
            model_action=model_action,
        )

    def payload(
        self, *, event: str, model_action: PlannedAction | None
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": event,
            "tiles": [int(tile) for tile in self.game.get_board().flatten()],
            "score": self.game.get_score(),
            "move_count": self.move_count,
            "done": self.game.done,
            "max_tile": self.game.max_square(),
            "checkpoint": None,
            "model_type": f"mc_{self.stages}step",
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

