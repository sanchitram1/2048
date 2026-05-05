from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import game2048.app as app_module
import training.inference as inference_module
import training.td_ntuple as td_module


class FakeRunner:
    def __init__(self, *, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.move_count = 0

    def reset(self) -> dict[str, object]:
        self.move_count = 0
        return {
            "event": "state",
            "tiles": [0] * 16,
            "score": 0,
            "move_count": 0,
            "done": False,
            "max_tile": 0,
            "checkpoint": str(self.checkpoint_path),
        }

    def step(self) -> dict[str, object]:
        self.move_count += 1
        return {
            "event": "agent_move",
            "tiles": [0, 1, *([0] * 14)],
            "score": 4,
            "move_count": self.move_count,
            "done": False,
            "max_tile": 2,
            "checkpoint": str(self.checkpoint_path),
            "move": "left",
            "action": 0,
            "q_values": [1.0, 0.0, -1.0, -2.0],
            "legal_actions": [0, 2],
        }


def test_agent_ws_reports_missing_model(monkeypatch) -> None:
    monkeypatch.setattr(td_module, "find_latest_td_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(inference_module, "find_latest_checkpoint", lambda: None)

    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/agent") as ws:
        message = ws.receive_json()

    assert message["event"] == "model_missing"


def test_agent_ws_streams_model_steps(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint_10.pt"
    monkeypatch.setattr(td_module, "find_latest_td_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(inference_module, "find_latest_checkpoint", lambda: checkpoint_path)
    monkeypatch.setattr(inference_module, "GreedyAgentRunner", FakeRunner)

    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/agent") as ws:
        first = ws.receive_json()
        assert first["event"] == "state"

        ws.send_json({"command": "step"})
        second = ws.receive_json()

    assert second["event"] == "agent_move"
    assert second["move"] == "left"
    assert second["q_values"] == [1.0, 0.0, -1.0, -2.0]


def test_agent_ws_can_select_myopic_greedy() -> None:
    client = TestClient(app_module.app)
    with client.websocket_connect("/ws/agent?agent=greedy") as ws:
        first = ws.receive_json()

    assert first["event"] == "state"
    assert first["model_type"] == "greedy_myopic"
