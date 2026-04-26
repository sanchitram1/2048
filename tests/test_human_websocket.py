"""Human play WebSocket contract."""

from __future__ import annotations

import random

from fastapi.testclient import TestClient

from game2048.app import app


def test_human_ws_initial_state_and_move_with_log() -> None:
    random.seed(0)
    client = TestClient(app)
    with client.websocket_connect("/ws/human") as ws:
        first = ws.receive_json()
        assert first["event"] == "state"
        assert first["log_line"] is None
        assert len(first["tiles"]) == 16
        assert first["move_count"] == 0

        ws.send_json({"move": "l"})
        second = ws.receive_json()
        assert second["event"] == "move_result"
        assert second["move_count"] == 1
        assert second["log_line"] == (
            "[HUMAN] Pressed left, new random number 2 generated at 8"
        )


def test_human_ws_no_op_has_no_log_line_and_same_move_count() -> None:
    random.seed(13)
    client = TestClient(app)
    with client.websocket_connect("/ws/human") as ws:
        ws.receive_json()
        ws.send_json({"move": "l"})
        reply = ws.receive_json()
        assert reply["event"] == "move_result"
        assert reply["move_count"] == 0
        assert reply["log_line"] is None


def test_human_ws_rejects_bad_move_key() -> None:
    random.seed(0)
    client = TestClient(app)
    with client.websocket_connect("/ws/human") as ws:
        ws.receive_json()
        ws.send_json({"move": "x"})
        err = ws.receive_json()
        assert err["event"] == "error"
