"""Canonical comparison metrics for diagnostics.

This module owns the **single source of truth** for alignment, action
agreement, and Bellman aggregation between models and the MCTS planner.

Invariants
----------
* Alignment is computed from the masked ``selected_action`` field on each
  model and planner output. We never call ``np.argmax`` on raw
  ``q_values`` for alignment, because the raw maximum may correspond to
  an illegal action.
* These functions accept either ``CompareOutputBoard`` instances (live
  in-memory, see ``game2048.compare``) or plain ``dict`` rows loaded from
  ``boards.jsonl`` files. Both shapes are handled uniformly.
"""

from __future__ import annotations

import math
import statistics
import uuid
from typing import Any


def _board_models(board: Any) -> list[dict[str, Any]]:
    if hasattr(board, "models"):
        return list(board.models)
    return list(board.get("models", []))


def _board_planner(board: Any) -> dict[str, Any] | None:
    if hasattr(board, "planner"):
        return board.planner
    return board.get("planner")


def _board_source(board: Any) -> dict[str, Any]:
    if hasattr(board, "source"):
        return board.source or {}
    return board.get("source", {}) or {}


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    xs = sorted(values)

    def q(p: float) -> float:
        return float(xs[round((len(xs) - 1) * p)])

    return {
        "n": len(xs),
        "mean": float(statistics.fmean(xs)),
        "p50": q(0.50),
        "p90": q(0.90),
        "p95": q(0.95),
        "p99": q(0.99),
        "min": float(xs[0]),
        "max": float(xs[-1]),
    }


def _softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [v / temperature for v in values]
    max_scaled = max(scaled)
    exps = [math.exp(v - max_scaled) for v in scaled]
    denom = sum(exps)
    return [v / denom for v in exps]


def _entropy(probs: list[float]) -> float:
    return float(-sum(p * math.log(p) for p in probs if p > 0))


def compute_summary(boards: list[Any]) -> dict[str, Any]:
    """Compute agreement, alignment, and Bellman summary statistics.

    Canonical implementation. Replaces the old ``_compute_summary`` in
    ``game2048.compare`` and the alignment counting in ``run_bellman``.

    Always uses masked ``selected_action`` for alignment decisions.
    """
    if not boards:
        return {"boards_count": 0}

    planner_alignment: dict[str, Any] = {}
    pairwise_alignment: dict[str, Any] = {}
    disagreement_by_action: dict[str, dict[str, int]] = {}
    action_margin_stats: dict[str, dict[str, float]] = {}
    disagreement_by_tail_state: dict[str, dict[str, Any]] = {}
    bellman_metrics: dict[str, dict[str, float]] = {}
    q_scale_diagnostics: dict[str, dict[str, Any]] = {}

    first_models = _board_models(boards[0])
    model_ids = [m["model_id"] for m in first_models]

    model_vs_planner_disagreements: dict[str, list[int]] = {}
    pairwise_disagreements: dict[str, list[int]] = {}

    for model_id in model_ids:
        agree_count = 0
        disagree_count = 0
        disagreement_pairs: dict[str, int] = {}
        margin_agree: list[float] = []
        margin_disagree: list[float] = []
        model_vs_planner_disagreements[model_id] = []
        td_delta_aligned: list[float] = []
        td_delta_disagreed: list[float] = []
        immediate_reward_shares: list[float] = []
        discounted_next_value_shares: list[float] = []
        teacher_gaps: list[float] = []
        student_gaps: list[float] = []
        teacher_ranges: list[float] = []
        student_ranges: list[float] = []
        teacher_student_gap_ratios: list[float] = []
        teacher_softmax_temperatures = (0.5, 1.0, 2.0, 5.0, 10.0)
        teacher_softmax_entropy: dict[float, list[float]] = {
            t: [] for t in teacher_softmax_temperatures
        }
        teacher_softmax_max_prob: dict[float, list[float]] = {
            t: [] for t in teacher_softmax_temperatures
        }

        for board in boards:
            planner = _board_planner(board)
            if planner is None:
                continue
            planner_action = planner["selected_action"]
            planner_move = planner["selected_move"]

            model_output = next(
                (m for m in _board_models(board) if m["model_id"] == model_id),
                None,
            )
            if not model_output:
                continue

            model_action = model_output["selected_action"]
            model_move = model_output["selected_move"]
            margin = float(model_output.get("action_margin", 0.0))
            legal_actions = list(getattr(board, "legal_actions", []))
            if not legal_actions and isinstance(board, dict):
                legal_actions = list(board.get("legal_actions", []))
            planner_q = list(planner.get("q_values", []))
            student_q = list(model_output.get("q_values", []))
            q_pairs: list[tuple[float, float]] = []
            for action in legal_actions:
                if action >= len(planner_q) or action >= len(student_q):
                    continue
                tq = float(planner_q[action])
                sq = float(student_q[action])
                if math.isfinite(tq) and math.isfinite(sq) and tq > -1.0e8:
                    q_pairs.append((tq, sq))
            if len(q_pairs) >= 2:
                teacher_vals = [pair[0] for pair in q_pairs]
                student_vals = [pair[1] for pair in q_pairs]
                teacher_sorted = sorted(teacher_vals, reverse=True)
                student_sorted = sorted(student_vals, reverse=True)
                teacher_gap = float(teacher_sorted[0] - teacher_sorted[1])
                student_gap = float(student_sorted[0] - student_sorted[1])
                teacher_gaps.append(teacher_gap)
                student_gaps.append(student_gap)
                teacher_ranges.append(float(max(teacher_vals) - min(teacher_vals)))
                student_ranges.append(float(max(student_vals) - min(student_vals)))
                if abs(student_gap) > 1.0e-9:
                    teacher_student_gap_ratios.append(
                        float(teacher_gap / student_gap)
                    )
                for temperature in teacher_softmax_temperatures:
                    probs = _softmax(teacher_vals, temperature)
                    teacher_softmax_entropy[temperature].append(_entropy(probs))
                    teacher_softmax_max_prob[temperature].append(float(max(probs)))

            if model_action == planner_action:
                agree_count += 1
                margin_agree.append(margin)
            else:
                disagree_count += 1
                margin_disagree.append(margin)
                pair_key = f"planner_{planner_move}__model_{model_move}"
                disagreement_pairs[pair_key] = disagreement_pairs.get(pair_key, 0) + 1
                model_vs_planner_disagreements[model_id].append(
                    int(_board_source(board).get("moves_from_episode_end", 0))
                )

            bellman_data = model_output.get("bellman", {}) or {}
            per_action = bellman_data.get("per_action", []) or []
            if 0 <= model_action < len(per_action):
                action_bellman = per_action[model_action]
                if isinstance(action_bellman, dict):
                    td_delta = action_bellman.get("td_delta")
                    td_target = action_bellman.get("td_target")
                    immediate_reward = action_bellman.get("immediate_reward")
                    discounted_next_value = action_bellman.get("discounted_next_value")

                    if td_delta is not None:
                        if model_action == planner_action:
                            td_delta_aligned.append(float(td_delta))
                        else:
                            td_delta_disagreed.append(float(td_delta))

                    if td_target is not None and td_target != 0:
                        if immediate_reward is not None:
                            immediate_reward_shares.append(
                                float(immediate_reward) / float(td_target)
                            )
                        if discounted_next_value is not None:
                            discounted_next_value_shares.append(
                                float(discounted_next_value) / float(td_target)
                            )

        total = agree_count + disagree_count
        planner_alignment[model_id] = {
            "agree_count": agree_count,
            "disagree_count": disagree_count,
            "agree_rate": float(agree_count) / total if total > 0 else 0.0,
        }
        disagreement_by_action[model_id] = disagreement_pairs
        action_margin_stats[model_id] = {
            "mean_when_agrees_with_planner": (
                sum(margin_agree) / len(margin_agree) if margin_agree else 0.0
            ),
            "mean_when_disagrees_with_planner": (
                sum(margin_disagree) / len(margin_disagree) if margin_disagree else 0.0
            ),
        }
        bellman_metrics[model_id] = {
            "mean_td_delta_aligned": (
                sum(td_delta_aligned) / len(td_delta_aligned)
                if td_delta_aligned
                else 0.0
            ),
            "mean_td_delta_disagreed": (
                sum(td_delta_disagreed) / len(td_delta_disagreed)
                if td_delta_disagreed
                else 0.0
            ),
            "mean_immediate_reward_share": (
                sum(immediate_reward_shares) / len(immediate_reward_shares)
                if immediate_reward_shares
                else 0.0
            ),
            "mean_discounted_next_value_share": (
                sum(discounted_next_value_shares) / len(discounted_next_value_shares)
                if discounted_next_value_shares
                else 0.0
            ),
        }
        q_scale_diagnostics[model_id] = {
            "teacher_gap": _summary_stats(teacher_gaps),
            "student_gap": _summary_stats(student_gaps),
            "teacher_range": _summary_stats(teacher_ranges),
            "student_range": _summary_stats(student_ranges),
            "teacher_student_gap_ratio": _summary_stats(
                teacher_student_gap_ratios
            ),
            "teacher_softmax": {
                str(temperature): {
                    "entropy": _summary_stats(
                        teacher_softmax_entropy[temperature]
                    ),
                    "max_prob": _summary_stats(
                        teacher_softmax_max_prob[temperature]
                    ),
                    "max_prob_gt_0_90": sum(
                        1 for p in teacher_softmax_max_prob[temperature] if p > 0.90
                    ),
                    "max_prob_gt_0_99": sum(
                        1 for p in teacher_softmax_max_prob[temperature] if p > 0.99
                    ),
                }
                for temperature in teacher_softmax_temperatures
            },
        }

    for i, id_a in enumerate(model_ids):
        for id_b in model_ids[i + 1 :]:
            pair_key = f"{id_a}__{id_b}"
            agree_count = 0
            disagree_count = 0
            pairwise_disagreements[pair_key] = []

            for board in boards:
                models = _board_models(board)
                out_a = next((m for m in models if m["model_id"] == id_a), None)
                out_b = next((m for m in models if m["model_id"] == id_b), None)

                if not out_a or not out_b:
                    continue

                if out_a["selected_action"] == out_b["selected_action"]:
                    agree_count += 1
                else:
                    disagree_count += 1
                    pairwise_disagreements[pair_key].append(
                        int(_board_source(board).get("moves_from_episode_end", 0))
                    )

            total = agree_count + disagree_count
            pairwise_alignment[pair_key] = {
                "agree_count": agree_count,
                "disagree_count": disagree_count,
                "agree_rate": float(agree_count) / total if total > 0 else 0.0,
            }

    def compute_tail_stats(disagreement_list: list[int]) -> dict[str, int]:
        stats = {"total": len(disagreement_list)}
        for window_size in (30, 20, 10):
            window_key = f"last_{window_size}"
            stats[window_key] = sum(1 for d in disagreement_list if d <= window_size)
        return stats

    for model_id in model_ids:
        disagreement_by_tail_state[f"{model_id}__planner"] = compute_tail_stats(
            model_vs_planner_disagreements[model_id]
        )
    for pair_key, disagreements in pairwise_disagreements.items():
        disagreement_by_tail_state[pair_key] = compute_tail_stats(disagreements)

    return {
        "run_id": str(uuid.uuid4()),
        "boards_count": len(boards),
        "planner_alignment": planner_alignment,
        "pairwise_alignment": pairwise_alignment,
        "disagreement_by_action": disagreement_by_action,
        "action_margin": action_margin_stats,
        "bellman_metrics": bellman_metrics,
        "q_scale_diagnostics": q_scale_diagnostics,
        "disagreement_by_tail_state": disagreement_by_tail_state,
        "diagnostics": {"easy_flags": []},
    }
