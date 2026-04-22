from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoardFrame:
    tiles: tuple[int, ...]
    score: int
    move_count: int
    caption: str

    @property
    def max_tile(self) -> int:
        highest = max(self.tiles, default=0)
        return 0 if highest == 0 else 2**highest


@dataclass(frozen=True)
class BoardView:
    board_id: str
    eyebrow: str
    title: str
    subtitle: str
    accent: str
    status: str
    interactive: bool
    frames: tuple[BoardFrame, ...]
    chips: tuple[str, ...]
    component_notes: tuple[str, ...]

    @property
    def initial_frame(self) -> BoardFrame:
        return self.frames[0]


@dataclass(frozen=True)
class InferenceCard:
    label: str
    value: str


@dataclass(frozen=True)
class TerminalView:
    log_lines: tuple[str, ...]
    inference_cards: tuple[InferenceCard, ...]
    inference_payload: str


@dataclass(frozen=True)
class AppView:
    title: str
    subtitle: str
    roadmap: tuple[str, ...]
    boards: tuple[BoardView, ...]
    terminal: TerminalView
