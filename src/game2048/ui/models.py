from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoardFrame:
    tiles: tuple[int, ...] = field(default_factory=tuple)
    score: int = 0
    move_count: int = 0
    caption: str = ""

    @property
    def max_tile(self) -> int:
        highest = max(self.tiles, default=0)
        return 0 if highest == 0 else 2**highest


@dataclass(frozen=True)
class BoardView:
    board_id: str = ""
    eyebrow: str = ""
    title: str = ""
    subtitle: str = ""
    accent: str = ""
    status: str = ""
    interactive: bool = False
    frames: tuple[BoardFrame, ...] = field(default_factory=tuple)
    chips: tuple[str, ...] = field(default_factory=tuple)
    component_notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def initial_frame(self) -> BoardFrame:
        if not self.frames:
            return BoardFrame()
        return self.frames[0]


@dataclass(frozen=True)
class InferenceCard:
    label: str = ""
    value: str = ""


@dataclass(frozen=True)
class TerminalView:
    log_lines: tuple[str, ...] = field(default_factory=tuple)
    inference_cards: tuple[InferenceCard, ...] = field(default_factory=tuple)
    inference_payload: str = ""


@dataclass(frozen=True)
class AppView:
    title: str = ""
    subtitle: str = ""
    roadmap: tuple[str, ...] = field(default_factory=tuple)
    boards: tuple[BoardView, ...] = field(default_factory=tuple)
    terminal: TerminalView = field(default_factory=TerminalView)
