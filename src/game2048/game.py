"""Game logic goes here"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import numpy as np


class GameLogic:
    """Core 2048 rules with log2 board encoding.

    Board encoding:
    - 0 means empty
    - 1 means tile value 2
    - 2 means tile value 4
    """

    def __init__(
        self,
        grid_size: int = 4,
        spawn_choices: Sequence[int] = (1, 2),
        spawn_probs: Sequence[float] = (0.9, 0.1),
    ) -> None:
        if len(spawn_choices) != len(spawn_probs):
            raise ValueError("spawn_choices and spawn_probs must have equal length")
        if not np.isclose(sum(spawn_probs), 1.0):
            raise ValueError("spawn_probs must sum to 1.0")

        self.grid_size = grid_size
        self.spawn_choices = list(spawn_choices)
        self.spawn_probs = list(spawn_probs)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int16)
        self.score = 0
        self.done = False
        self.new_number(k=2)

    def __str__(self) -> str:
        return str(self.get_board_values())

    def choose_number(self) -> int:
        return int(random.choices(self.spawn_choices, self.spawn_probs)[0])

    def get_score(self) -> int:
        return int(self.score)

    def get_board(self) -> np.ndarray:
        """Return a copy of the log2-encoded board."""
        return self.grid.copy()

    def get_board_values(self) -> np.ndarray:
        """Return decoded tile values for display/debugging."""
        values = np.zeros_like(self.grid, dtype=np.int64)
        non_zero_mask = self.grid > 0
        values[non_zero_mask] = 2 ** self.grid[non_zero_mask]
        return values

    def max_square(self) -> int:
        if self.grid.size == 0:
            return 0
        highest = int(np.max(self.grid))
        return 0 if highest == 0 else int(2**highest)

    def open_positions(self) -> List[Tuple[int, int]]:
        return list(zip(*np.where(self.grid == 0)))

    def _row_move_left(self, row: np.ndarray) -> Tuple[np.ndarray, int]:
        tiles = [int(tile) for tile in row if tile != 0]
        merged_tiles: List[int] = []
        score_gain = 0
        idx = 0
        while idx < len(tiles):
            current = tiles[idx]
            if idx + 1 < len(tiles) and tiles[idx + 1] == current:
                merged_exp = current + 1
                merged_tiles.append(merged_exp)
                score_gain += 2**merged_exp
                idx += 2
            else:
                merged_tiles.append(current)
                idx += 1
        merged_tiles.extend([0] * (self.grid_size - len(merged_tiles)))
        return np.array(merged_tiles, dtype=np.int16), score_gain

    def _apply_move_to_grid(
        self, grid: np.ndarray, move: str
    ) -> Tuple[np.ndarray, int]:
        if move not in {"l", "r", "u", "d"}:
            raise ValueError(f"Invalid move '{move}'. Expected one of: l, r, u, d.")

        working_grid = grid.copy()
        score_gain = 0

        if move in {"u", "d"}:
            working_grid = working_grid.T
        if move in {"r", "d"}:
            working_grid = np.fliplr(working_grid)

        result = np.zeros_like(working_grid)
        for row_idx in range(self.grid_size):
            moved_row, row_gain = self._row_move_left(working_grid[row_idx])
            result[row_idx] = moved_row
            score_gain += row_gain

        if move in {"r", "d"}:
            result = np.fliplr(result)
        if move in {"u", "d"}:
            result = result.T

        return result, score_gain

    def has_valid_moves(self) -> bool:
        return len(self.available_moves()) > 0

    def available_moves(self) -> List[str]:
        legal_moves: List[str] = []
        for move in ("l", "r", "u", "d"):
            next_grid, _ = self._apply_move_to_grid(self.grid, move)
            if not np.array_equal(next_grid, self.grid):
                legal_moves.append(move)
        return legal_moves

    def new_number(self, k: int = 1) -> None:
        open_positions = self.open_positions()
        if not open_positions:
            return
        selected_positions = random.sample(open_positions, min(k, len(open_positions)))
        for row_idx, col_idx in selected_positions:
            self.grid[row_idx, col_idx] = self.choose_number()

    def reset(self) -> Tuple[np.ndarray, int, bool]:
        self.grid.fill(0)
        self.score = 0
        self.done = False
        self.new_number(k=2)
        return self.generate_observation()

    def generate_observation(self) -> Tuple[np.ndarray, int, bool]:
        return self.get_board(), self.get_score(), self.done

    def make_move(self, move: str) -> Tuple[bool, int]:
        """Apply one move and spawn if movement occurred.

        Returns:
            tuple[bool, int]: (board_changed, score_gain)
        """
        if self.done:
            return False, 0

        next_grid, score_gain = self._apply_move_to_grid(self.grid, move)
        board_changed = not np.array_equal(next_grid, self.grid)
        if board_changed:
            self.grid = next_grid
            self.score += score_gain
            self.new_number(k=1)

        self.done = not self.has_valid_moves()
        return board_changed, score_gain