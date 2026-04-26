from __future__ import annotations

from textwrap import dedent

from game2048.ui.models import (
    AppView,
    BoardFrame,
    BoardView,
    InferenceCard,
    TerminalView,
)


def _frame(
    rows: tuple[tuple[int, int, int, int], ...],
    *,
    score: int,
    move_count: int,
    caption: str,
) -> BoardFrame:
    tiles = tuple(tile for row in rows for tile in row)
    return BoardFrame(
        tiles=tiles,
        score=score,
        move_count=move_count,
        caption=caption,
    )


def build_mock_view() -> AppView:
    empty_row = (0, 0, 0, 0)
    human_frames = (
        _frame(
            (empty_row, empty_row, empty_row, empty_row),
            score=0,
            move_count=0,
            caption="",
        ),
    )

    agent_frames = (
        _frame(
            ((0, 1, 0, 2), (0, 0, 3, 0), (1, 0, 0, 2), (0, 1, 0, 0)),
            score=256,
            move_count=11,
            caption="Agent board is reserved for streamed decisions over WebSocket.",
        ),
        _frame(
            ((0, 0, 1, 2), (0, 0, 3, 1), (0, 1, 0, 2), (0, 0, 0, 1)),
            score=292,
            move_count=12,
            caption="Policy scores, chosen action, and value estimate can all render beside this board.",
        ),
        _frame(
            ((0, 0, 0, 3), (0, 0, 1, 2), (0, 0, 3, 0), (0, 1, 0, 1)),
            score=356,
            move_count=13,
            caption="When inference is real, this side should advance one move after the human turn commits.",
        ),
    )

    terminal_payload = dedent(
        """
        {
          "event": "agent_move",
          "step": 18204,
          "move": "left",
          "policy": {"left": 0.54, "up": 0.21, "right": 0.15, "down": 0.10},
          "value": 0.73,
          "score": 4096
        }
        """
    ).strip()

    return AppView(
        title="2048 Human vs Agent",
        subtitle="",
        roadmap=(),
        boards=(
            BoardView(
                board_id="human-board",
                eyebrow="",
                title="Human",
                subtitle="",
                accent="#cf6b2d",
                status="Click the board to focus, then use arrow keys",
                interactive=True,
                frames=human_frames,
                chips=(),
                component_notes=(),
            ),
            BoardView(
                board_id="agent-board",
                eyebrow="",
                title="Agent",
                subtitle="",
                accent="#1d8f88",
                status="Waiting for agent stream",
                interactive=False,
                frames=agent_frames,
                chips=(),
                component_notes=(),
            ),
        ),
        terminal=TerminalView(
            log_lines=(
                "[system] app shell booted",
                "[human] live game runs over WebSocket /ws/human",
                "[agent] /ws/agent is reserved for the future inference stream",
                "[render] agent board preview is static until inference is wired",
            ),
            inference_cards=(
                InferenceCard(label="Transport", value="WebSocket /ws/agent"),
                InferenceCard(
                    label="Cadence", value="1 agent move per committed human turn"
                ),
                InferenceCard(
                    label="Payload",
                    value="chosen move + policy scores + value estimate",
                ),
                InferenceCard(
                    label="Target",
                    value="agent board, inference tab, and shared log stream",
                ),
            ),
            inference_payload=terminal_payload,
        ),
    )
