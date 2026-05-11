from __future__ import annotations

from fastapi.testclient import TestClient

import game2048.app as app_module
import training.inference as inference_module


def test_match_ws_initial_boards_match_for_greedy() -> None:
    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/match?agent=greedy&seed=999") as ws:
        msg = ws.receive_json()

    assert msg["event"] == "match_state"
    assert msg["human"]["tiles"] == msg["agent"]["tiles"]
    assert msg["human"]["score"] == msg["agent"]["score"]
    assert msg["match"]["seed"] == 999


def test_match_step_agent_before_human_done_errors() -> None:
    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/match?agent=greedy&seed=3") as ws:
        ws.receive_json()
        ws.send_json({"command": "step_agent"})
        out = ws.receive_json()

    assert out["event"] == "error"
    assert "still playing" in (out.get("message") or "").lower()


def test_match_human_move_returns_turn_payload() -> None:
    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/match?agent=greedy&seed=3") as ws:
        ws.receive_json()
        ws.send_json({"command": "human_move", "move": "l"})
        out = ws.receive_json()

    assert out["event"] in ("turn_result", "match_complete")
    assert "human" in out and "agent" in out and "match" in out


def test_match_human_move_accepts_arrow_key_style_input() -> None:
    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/match?agent=greedy&seed=3") as ws:
        ws.receive_json()
        ws.send_json({"command": "human_move", "move": "ArrowUp"})
        out = ws.receive_json()

    assert out["event"] in ("turn_result", "match_complete")
    assert out["last_human_moved"] is True


def test_match_human_move_accepts_word_style_input() -> None:
    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/match?agent=greedy&seed=3") as ws:
        ws.receive_json()
        ws.send_json({"command": "human_move", "move": "up"})
        out = ws.receive_json()

    assert out["event"] in ("turn_result", "match_complete")
    assert out["last_human_moved"] is True


def test_match_dqn_missing_checkpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        inference_module, "find_latest_checkpoint", lambda *_a, **_k: None
    )

    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/match?agent=dqn") as ws:
        msg = ws.receive_json()

    assert msg["event"] == "model_missing"
