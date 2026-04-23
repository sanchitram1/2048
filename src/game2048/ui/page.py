from __future__ import annotations

import json
from html import escape

from game2048.ui.board import render_board
from game2048.ui.models import AppView
from game2048.ui.scripts import render_scripts
from game2048.ui.styles import render_styles
from game2048.ui.terminal import render_terminal


def _serialize_boards(view: AppView) -> dict[str, object]:
    return {
        "boards": {
            board.board_id: {
                "frames": [
                    {
                        "tiles": list(frame.tiles),
                        "score": frame.score,
                        "maxTile": frame.max_tile,
                        "moveCount": frame.move_count,
                        "caption": frame.caption,
                    }
                    for frame in board.frames
                ]
            }
            for board in view.boards
        }
    }


def render_page(view: AppView) -> str:
    boards_html = "".join(render_board(board) for board in view.boards)
    ui_state = json.dumps(_serialize_boards(view))

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escape(view.title)}</title>
    <style>{render_styles()}</style>
  </head>
  <body>
    <main class="app-shell">
      <section class="hero">
        <article class="hero__panel">
          <h1>{escape(view.title)}</h1>
          <p class="hero__lede">{escape(view.subtitle)}</p>
        </article>
      </section>
      <section class="board-layout">{boards_html}</section>
      {render_terminal(view.terminal)}
    </main>
    <script id="mock-ui-state" type="application/json">{ui_state}</script>
    <script>{render_scripts()}</script>
  </body>
</html>"""
