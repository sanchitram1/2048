from __future__ import annotations

from html import escape

from game2048.ui.models import BoardView
from game2048.ui.tile import render_tile


def _render_controls(board: BoardView) -> str:
    if board.interactive:
        return """<div class="lane-tools" aria-label="keyboard controls preview">
  <span class="keycap">&uarr;</span>
  <span class="keycap">&larr;</span>
  <span class="keycap">&darr;</span>
  <span class="keycap">&rarr;</span>
  <span class="lane-tools__hint">Arrow keys drive the mocked human lane.</span>
</div>"""

    return """<div class="lane-tools lane-tools--agent" aria-label="agent stream preview">
  <span class="keycap keycap--agent">WS</span>
  <span class="keycap keycap--agent">P</span>
  <span class="keycap keycap--agent">V</span>
  <span class="lane-tools__hint">Stream: move, policy distribution, value estimate.</span>
</div>"""


def render_board(board: BoardView) -> str:
    frame = board.initial_frame
    tiles_html = "".join(
        render_tile(exponent, index) for index, exponent in enumerate(frame.tiles)
    )
    chips_html = "".join(
        f'<li class="chip">{escape(chip)}</li>' for chip in board.chips
    )
    notes_html = "".join(
        f'<li class="board-notes__item">{escape(note)}</li>'
        for note in board.component_notes
    )
    board_classes = (
        "board-card board-card--interactive" if board.interactive else "board-card"
    )

    return f"""<section class="{board_classes}" data-board-id="{escape(board.board_id)}" style="--board-accent: {escape(board.accent)};">
  <header class="board-card__header">
    <div>
      <p class="eyebrow">{escape(board.eyebrow)}</p>
      <h2>{escape(board.title)}</h2>
      <p class="board-card__subtitle">{escape(board.subtitle)}</p>
    </div>
    <div class="board-status">
      <span class="board-status__label">Status</span>
      <strong class="board-status__value" data-board-status>{escape(board.status)}</strong>
    </div>
  </header>
  <ul class="chip-row">{chips_html}</ul>
  {_render_controls(board)}
  <section class="board-grid-shell">
    <div class="board-hud">
      <div class="stat-card" data-stat="score">
        <span class="stat-card__label">Score</span>
        <strong class="stat-card__value" data-stat-value="score">{frame.score}</strong>
      </div>
      <div class="stat-card" data-stat="best">
        <span class="stat-card__label">Max Tile</span>
        <strong class="stat-card__value" data-stat-value="best">{frame.max_tile}</strong>
      </div>
      <div class="stat-card" data-stat="moves">
        <span class="stat-card__label">Moves</span>
        <strong class="stat-card__value" data-stat-value="moves">{frame.move_count}</strong>
      </div>
    </div>
    <div class="board-grid" role="img" aria-label="{escape(board.title)} grid preview">
      {tiles_html}
    </div>
  </section>
  <footer class="board-footer">
    <p class="board-caption" data-board-caption>{escape(frame.caption)}</p>
    <ul class="board-notes">{notes_html}</ul>
  </footer>
</section>"""
