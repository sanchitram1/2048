from __future__ import annotations

import secrets
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from game2048.game import GameLogic
from game2048.game_logger import GameLogger
from game2048.match import MatchSession, ModelMissingError
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


def _normalize_move_token(move: object) -> str | None:
    """Normalize UI move tokens to canonical game tokens."""
    if not isinstance(move, str):
        return None
    token = move.strip()
    if not token:
        return None
    normalized = token.lower()
    move_map = {
        "l": "l",
        "left": "l",
        "arrowleft": "l",
        "r": "r",
        "right": "r",
        "arrowright": "r",
        "u": "u",
        "up": "u",
        "arrowup": "u",
        "d": "d",
        "down": "d",
        "arrowdown": "d",
    }
    return move_map.get(normalized)


@app.get("/", response_class=HTMLResponse)
async def read_index() -> HTMLResponse:
    return HTMLResponse(render_page(build_mock_view()))


@app.websocket("/ws/agent")
async def agent_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    agent_type = (websocket.query_params.get("agent") or "dqn").lower()
    missing_checkpoint_message = "no checkpoint found in models..."

    if agent_type in {"greedy", "myopic"}:
        from training.planning import MyopicGreedyRunner

        runner = MyopicGreedyRunner()
    elif agent_type in {"mc", "mcts", "nstep"}:
        from training.planning import NStepMCRunner

        runner = NStepMCRunner()
    elif agent_type == "dqn":
        from training.inference import GreedyAgentRunner, find_latest_checkpoint

        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            await websocket.send_json(
                {"event": "model_missing", "message": missing_checkpoint_message}
            )
            await websocket.close()
            return
        runner = GreedyAgentRunner(checkpoint_path=checkpoint_path)
    elif agent_type == "td":
        from training.td_ntuple import TDNTupleAgentRunner, find_latest_td_checkpoint

        td_checkpoint_path = find_latest_td_checkpoint()
        if td_checkpoint_path is None:
            await websocket.send_json(
                {"event": "model_missing", "message": missing_checkpoint_message}
            )
            await websocket.close()
            return
        runner = TDNTupleAgentRunner(checkpoint_path=td_checkpoint_path)
    else:
        await websocket.send_json(
            {
                "event": "error",
                "message": f"Unknown agent type: {agent_type}",
            }
        )
        await websocket.close()
        return
    await websocket.send_json(runner.reset())

    try:
        while True:
            raw = await websocket.receive_json()
            command = raw.get("command", "step")
            if command == "reset":
                await websocket.send_json(runner.reset())
                continue
            if command != "step":
                await websocket.send_json(
                    {
                        "event": "error",
                        "message": "Expected command 'step' or 'reset'.",
                    }
                )
                continue

            await websocket.send_json(runner.step())
    except WebSocketDisconnect:
        return


@app.websocket("/ws/match")
async def match_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    missing_checkpoint_message = "no checkpoint found in models..."
    agent_type = (websocket.query_params.get("agent") or "dqn").lower()
    seed_raw = websocket.query_params.get("seed")
    seed = int(seed_raw) if seed_raw is not None else secrets.randbelow(2**31)

    try:
        session = MatchSession.start(agent_type=agent_type, seed=seed)
    except ModelMissingError:
        await websocket.send_json(
            {"event": "model_missing", "message": missing_checkpoint_message}
        )
        await websocket.close()
        return
    except ValueError as exc:
        await websocket.send_json({"event": "error", "message": str(exc)})
        await websocket.close()
        return

    await websocket.send_json(session.snapshot("match_state"))

    try:
        while True:
            raw = await websocket.receive_json()
            command = str(raw.get("command", "human_move")).lower()
            if command == "reset":
                new_agent = raw.get("agent", agent_type)
                if isinstance(new_agent, str):
                    new_agent = new_agent.lower()
                seed_in = raw.get("seed")
                new_seed = (
                    int(seed_in) if seed_in is not None else secrets.randbelow(2**31)
                )
                try:
                    session = MatchSession.start(agent_type=new_agent, seed=new_seed)
                except ModelMissingError:
                    await websocket.send_json(
                        {
                            "event": "model_missing",
                            "message": missing_checkpoint_message,
                        }
                    )
                    continue
                agent_type = new_agent
                await websocket.send_json(session.snapshot("match_state"))
                continue
            if command == "human_move":
                move = _normalize_move_token(raw.get("move"))
                if move is None:
                    await websocket.send_json(
                        {
                            "event": "error",
                            "message": (
                                "Expected move in l/r/u/d (or left/right/up/down,"
                                " ArrowLeft/ArrowRight/ArrowUp/ArrowDown)."
                            ),
                        }
                    )
                    continue
                await websocket.send_json(session.human_move(move))
                continue
            if command == "step_agent":
                await websocket.send_json(session.step_agent())
                continue
            await websocket.send_json(
                {
                    "event": "error",
                    "message": "Expected command reset, human_move, or step_agent.",
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
            move = _normalize_move_token(raw.get("move"))
            if move is None:
                await websocket.send_json(
                    {
                        "event": "error",
                        "message": (
                            "Expected JSON key 'move' in l/r/u/d (or left/right/up/down,"
                            " ArrowLeft/ArrowRight/ArrowUp/ArrowDown)."
                        ),
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
