from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from game2048.game import GameLogic
from game2048.game_logger import GameLogger
from game2048.ui.mock_state import build_mock_view
from game2048.ui.page import render_page

app = FastAPI(title="2048 Human vs Agent")


def _human_board_tiles(game: GameLogic) -> list[int]:
    grid = game.get_board()
    size = game.grid_size
    return [int(grid[row, col]) for row in range(size) for col in range(size)]


def _max_tile_from_tile_exponents(tiles: list[int]) -> int:
    highest = max(tiles, default=0)
    return 0 if highest == 0 else 2**highest


def _human_ws_payload(
    game: GameLogic,
    move_count: int,
    event: str,
    log_line: str | None,
) -> dict[str, object]:
    tiles = _human_board_tiles(game)
    return {
        "event": event,
        "tiles": tiles,
        "score": game.get_score(),
        "move_count": move_count,
        "done": game.done,
        "max_tile": _max_tile_from_tile_exponents(tiles),
        "log_line": log_line,
    }


@app.get("/", response_class=HTMLResponse)
async def read_index() -> HTMLResponse:
    return HTMLResponse(render_page(build_mock_view()))


@app.websocket("/ws/agent")
async def agent_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json(
        {
            "event": "connected",
            "message": "Mock agent stream is reserved here until the inference loop is wired in.",
        }
    )

    try:
        while True:
            await websocket.receive_text()
            await websocket.send_json(
                {
                    "event": "noop",
                    "message": "This WebSocket is only a placeholder in the current milestone.",
                }
            )
    except WebSocketDisconnect:
        return


@app.websocket("/ws/human")
async def human_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    game = GameLogic()
    logger = GameLogger("HUMAN")
    move_count = 0
    await websocket.send_json(_human_ws_payload(game, move_count, "state", None))

    try:
        while True:
            raw = await websocket.receive_json()
            move = raw.get("move")
            if move not in ("l", "r", "u", "d"):
                await websocket.send_json(
                    {
                        "event": "error",
                        "message": "Expected JSON object with key 'move' in l, r, u, d.",
                    }
                )
                continue

            changed, _gain, spawn_flat, spawn_val = game.make_move(move)
            log_line: str | None = None
            if changed:
                move_count += 1
                if spawn_flat is not None and spawn_val is not None:
                    log_line = logger.line_for_move(move, spawn_val, spawn_flat)

            await websocket.send_json(
                _human_ws_payload(game, move_count, "move_result", log_line)
            )
    except WebSocketDisconnect:
        return


def main() -> None:
    uvicorn.run("game2048.app:app", host="127.0.0.1", port=8000)
