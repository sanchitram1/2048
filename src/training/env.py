"""RL environment wrapper around the pure 2048 game engine."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from game2048.game import GameLogic


@dataclass(frozen=True)
class RewardConfig:
    """Weights used to shape reward from game transitions."""

    max_tile_bonus: float = 2.0
    score_gain_scale: float = 0.01
    empty_tile_bonus: float = 0.05
    invalid_move_penalty: float = -0.2
    game_over_penalty: float = -1.0


class Game2048Env:
    """Training-friendly wrapper for GameLogic.

    This adapter owns RL concerns (action mapping, reward shaping, and step API),
    while the game engine remains a pure rules implementation.
    """

    ACTION_TO_MOVE = {0: "l", 1: "r", 2: "u", 3: "d"}
    MOVE_TO_ACTION = {move: action for action, move in ACTION_TO_MOVE.items()}

    def __init__(self, reward_config: RewardConfig | None = None) -> None:
        self.game = GameLogic()
        self.reward_config = reward_config or RewardConfig()

    @staticmethod
    def action_space_n() -> int:
        return 4

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    def reset(self) -> Tuple[np.ndarray, Dict[str, object]]:
        board, score, done = self.game.reset()
        return board, self._build_info(
            score=score, done=done, moved=False, score_gain=0
        )

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, object]]:
        if action not in self.ACTION_TO_MOVE:
            raise ValueError(f"Invalid action {action}; expected one of 0, 1, 2, 3")

        old_board = self.game.get_board()
        old_max = int(np.max(old_board))
        old_empty = int(np.count_nonzero(old_board == 0))

        moved, score_gain = self.game.make_move(self.ACTION_TO_MOVE[action])
        new_board = self.game.get_board()
        new_score = self.game.get_score()
        done = self.game.done

        reward = self._compute_reward(
            moved=moved,
            old_max=old_max,
            new_max=int(np.max(new_board)),
            old_empty=old_empty,
            new_empty=int(np.count_nonzero(new_board == 0)),
            score_gain=score_gain,
            done=done,
        )
        info = self._build_info(
            score=new_score, done=done, moved=moved, score_gain=score_gain
        )
        return new_board, reward, done, False, info

    def legal_actions(self) -> list[int]:
        return [self.MOVE_TO_ACTION[move] for move in self.game.available_moves()]

    def sample_legal_action(self) -> int:
        legal = self.legal_actions()
        if not legal:
            raise RuntimeError("Cannot sample action: no legal actions available")
        return int(random.choice(legal))

    def _compute_reward(
        self,
        *,
        moved: bool,
        old_max: int,
        new_max: int,
        old_empty: int,
        new_empty: int,
        score_gain: int,
        done: bool,
    ) -> float:
        cfg = self.reward_config
        if not moved:
            return cfg.invalid_move_penalty

        reward = 0.0
        if new_max > old_max:
            reward += cfg.max_tile_bonus * float(new_max - old_max)
        reward += cfg.score_gain_scale * float(score_gain)
        reward += cfg.empty_tile_bonus * float(new_empty - old_empty)
        if done:
            reward += cfg.game_over_penalty
        return reward

    def _build_info(
        self, *, score: int, done: bool, moved: bool, score_gain: int
    ) -> Dict[str, object]:
        return {
            "score": int(score),
            "done": bool(done),
            "moved": bool(moved),
            "score_gain": int(score_gain),
            "max_tile": int(self.game.max_square()),
            "board_values": self.game.get_board_values(),
        }
