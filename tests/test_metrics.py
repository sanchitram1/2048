"""Unit tests for the canonical compare metrics module."""

from __future__ import annotations

from evaluation.metrics import compute_summary


def _board(
    *,
    board_id: str,
    moves_from_episode_end: int,
    planner_action: int,
    model_actions: list[int],
    q_per_model: list[list[float]] | None = None,
) -> dict:
    """Build a synthetic dict-shaped CompareOutputBoard."""
    if q_per_model is None:
        q_per_model = [[0.0, 0.0, 0.0, 0.0] for _ in model_actions]

    move = {0: "l", 1: "r", 2: "u", 3: "d"}
    return {
        "board_id": board_id,
        "source": {"moves_from_episode_end": moves_from_episode_end},
        "board": [[0] * 4] * 4,
        "legal_actions": [0, 1, 2, 3],
        "planner": {
            "selected_action": planner_action,
            "selected_move": move[planner_action],
            "q_values": [0.0] * 4,
        },
        "models": [
            {
                "model_id": f"model_{i}",
                "q_values": q_per_model[i],
                "selected_action": a,
                "selected_move": move[a],
                "action_margin": 1.0,
                "bellman": {
                    "gamma": 0.99,
                    "per_action": [
                        {
                            "immediate_reward": 0.0,
                            "next_value": 0.0,
                            "discounted_next_value": 0.0,
                            "td_target": 0.0,
                            "current_q": 0.0,
                            "td_delta": 0.0,
                        }
                        for _ in range(4)
                    ],
                },
            }
            for i, a in enumerate(model_actions)
        ],
    }


def test_compute_summary_schema() -> None:
    boards = [
        _board(
            board_id="b0",
            moves_from_episode_end=0,
            planner_action=1,
            model_actions=[1, 2],
        ),
        _board(
            board_id="b1",
            moves_from_episode_end=12,
            planner_action=2,
            model_actions=[2, 3],
        ),
    ]
    summary = compute_summary(boards)

    assert summary["boards_count"] == 2
    expected_keys = {
        "run_id",
        "boards_count",
        "planner_alignment",
        "pairwise_alignment",
        "disagreement_by_action",
        "action_margin",
        "bellman_metrics",
        "disagreement_by_tail_state",
        "diagnostics",
    }
    assert expected_keys.issubset(summary.keys())

    assert summary["planner_alignment"]["model_0"]["agree_count"] == 2
    # model_1 picks the wrong action on both boards (2, 3 vs planner 1, 2).
    assert summary["planner_alignment"]["model_1"]["agree_count"] == 0

    pairs = summary["pairwise_alignment"]
    assert "model_0__model_1" in pairs
    assert pairs["model_0__model_1"]["disagree_count"] == 2

    tail = summary["disagreement_by_tail_state"]
    assert "model_1__planner" in tail
    assert tail["model_1__planner"]["total"] == 2
    # Of the two disagreements, only the one at moves_from_episode_end == 0
    # falls inside the last_10 window; the other is at 12.
    assert tail["model_1__planner"]["last_10"] == 1


def test_bellman_uses_selected_action() -> None:
    """Alignment must come from selected_action, not raw argmax(q_values).

    The planner's raw argmax is action 0 (illegal-but-highest); the model's
    raw argmax is action 3. If we incorrectly used raw argmax the boards
    would disagree. With the canonical masked path both have
    selected_action == 1 and they agree.
    """
    boards = [
        _board(
            board_id="b0",
            moves_from_episode_end=0,
            planner_action=1,
            model_actions=[1],
            q_per_model=[[0.0, 5.0, 1.0, 9.0]],
        ),
    ]
    boards[0]["planner"]["q_values"] = [10.0, 5.0, 0.0, 0.0]

    summary = compute_summary(boards)
    assert summary["planner_alignment"]["model_0"]["agree_count"] == 1
    assert summary["planner_alignment"]["model_0"]["disagree_count"] == 0
    assert summary["planner_alignment"]["model_0"]["agree_rate"] == 1.0


def test_compute_summary_empty_returns_minimal_dict() -> None:
    summary = compute_summary([])
    assert summary == {"boards_count": 0}
