"""Human / agent log line formatting."""

from __future__ import annotations

from game2048.game_logger import GameLogger


def test_human_log_line_matches_contract() -> None:
    line = GameLogger("HUMAN").line_for_move("u", 2, 0)
    assert line == "[HUMAN] Pressed up, new random number 2 generated at 0"


def test_agent_uses_same_shape() -> None:
    line = GameLogger("AGENT").line_for_move("d", 4, 15)
    assert line == "[AGENT] Pressed down, new random number 4 generated at 15"
