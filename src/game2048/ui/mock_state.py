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
    human_frames = (
        _frame(
            ((1, 0, 2, 0), (0, 3, 0, 1), (0, 0, 2, 0), (1, 0, 0, 0)),
            score=128,
            move_count=12,
            caption="Arrow keys are wired to a fake preview loop for now.",
        ),
        _frame(
            ((1, 2, 0, 0), (3, 1, 0, 0), (2, 0, 0, 0), (1, 0, 1, 0)),
            score=132,
            move_count=13,
            caption="This is where local move reduction and tile spawn feedback will show up.",
        ),
        _frame(
            ((0, 0, 1, 2), (0, 3, 1, 0), (0, 0, 0, 2), (0, 0, 1, 1)),
            score=164,
            move_count=14,
            caption="Human input should eventually drive the real game engine instead of this carousel.",
        ),
        _frame(
            ((0, 0, 0, 3), (0, 0, 3, 1), (0, 0, 0, 2), (0, 0, 0, 2)),
            score=196,
            move_count=15,
            caption="The score, move count, and board diff are the key things to preserve in the final loop.",
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
        subtitle=(
            "A mocked front-end shell for comparing manual play, agent inference, "
            "and runtime logs on one screen."
        ),
        roadmap=(
            "Human lane: arrow-key input and local board rendering.",
            "Agent lane: WebSocket-fed moves and policy overlays.",
            "Console lane: logs, inference traces, and score deltas.",
        ),
        boards=(
            BoardView(
                board_id="human-board",
                eyebrow="Milestone 01",
                title="Human Board",
                subtitle="This side is for manual play. Arrow keys advance the mocked preview.",
                accent="#cf6b2d",
                status="Arrow keys ready",
                interactive=True,
                frames=human_frames,
                chips=("keyboard listener", "move reducer", "board renderer"),
                component_notes=(
                    "Capture arrow keys and map them to move intents.",
                    "Translate the new board state into tile components and HUD stats.",
                    "Append each move to the shared terminal so the comparison lane stays synchronized.",
                ),
            ),
            BoardView(
                board_id="agent-board",
                eyebrow="Milestone 02",
                title="RL Agent Board",
                subtitle="This side stays mocked until the inference loop is wired in over WebSocket.",
                accent="#1d8f88",
                status="Waiting for agent stream",
                interactive=False,
                frames=agent_frames,
                chips=("websocket client", "policy trace", "decision renderer"),
                component_notes=(
                    "Receive model events over a persistent WebSocket instead of polling.",
                    "Render the chosen move, confidence scores, and value estimate alongside the board.",
                    "Keep agent turns visually paired with the matching human move for side-by-side comparison.",
                ),
            ),
        ),
        terminal=TerminalView(
            log_lines=(
                "[system] app shell booted",
                "[human] arrow keys are captured on the page",
                "[agent] /ws/agent is reserved for the future inference stream",
                "[render] board updates are mocked with prebuilt preview frames",
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
