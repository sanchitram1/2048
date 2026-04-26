"""Structured log lines for human vs agent play."""

from __future__ import annotations

from typing import Literal

LoggerName = Literal["HUMAN", "AGENT"]

_MOVE_WORD: dict[str, str] = {
    "l": "left",
    "r": "right",
    "u": "up",
    "d": "down",
}


class GameLogger:
    """Formats console lines like ``[HUMAN] Pressed up, new random number 2 generated at 0``."""

    def __init__(self, name: LoggerName) -> None:
        self._name = name

    def line_for_move(self, move: str, spawn_value: int, grid_index: int) -> str:
        if move not in _MOVE_WORD:
            raise ValueError(f"Invalid move {move!r}; expected one of l, r, u, d.")
        direction = _MOVE_WORD[move]
        return (
            f"[{self._name}] Pressed {direction}, new random number {spawn_value} "
            f"generated at {grid_index}"
        )
