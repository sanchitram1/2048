from __future__ import annotations

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from game2048.ui.mock_state import build_mock_view
from game2048.ui.page import render_page

app = FastAPI(title="2048 Human vs Agent")


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


def main() -> None:
    uvicorn.run("game2048.app:app", host="127.0.0.1", port=8000)
