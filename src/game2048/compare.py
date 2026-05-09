"""Diagnostic comparison of model outputs on the same board set."""

from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.dqn import legal_actions_to_mask, mask_illegal_actions
from training.env import Game2048Env
from training.inference import load_q_network
from training.planning import choose_n_step_mc

ACTION_TO_MOVE = {0: "l", 1: "r", 2: "u", 3: "d"}


@dataclass(frozen=True)
class BoardTransition:
    """A single transition from rollout collection."""

    game_id: int
    move_index: int
    score: float
    max_tile: int
    observed_action: int
    observed_move: str
    observed_reward: float
    done: bool
    board: list[list[int]]  # 4x4 log2 exponents
    moves_from_episode_end: int = 0  # How many moves from end of episode (0=last move)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutConfig:
    """Metadata for a rollout board collection."""

    source_checkpoint: str
    episodes: int
    tail_moves: int
    seed: int
    created_at: str
    total_transitions: int


def collect_rollout_boards(
    *,
    checkpoint_path: Path,
    episodes: int,
    tail_moves: int,
    seed: int,
    device_name: str = "cpu",
) -> tuple[list[BoardTransition], RolloutConfig]:
    """
    Collect the last tail_moves transitions from each episode of rollout.
    Returns board transitions and metadata for saving.
    """
    from datetime import datetime, timezone

    q_network, config, device = load_q_network(checkpoint_path, device_name=device_name)

    transitions: list[BoardTransition] = []
    env = Game2048Env()

    for game_id in range(episodes):
        env.seed(seed + game_id)
        state_board, info = env.reset()
        move_index = 0
        tail_window: list[BoardTransition] = []

        while True:
            legal_actions = env.legal_actions()
            action_mask = torch.as_tensor(
                legal_actions_to_mask(4, legal_actions),
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
            state_tensor = torch.as_tensor(
                state_board, dtype=torch.long, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = q_network(state_tensor)
                masked_q_values = mask_illegal_actions(q_values, action_mask)
                action = int(masked_q_values.argmax(dim=1).item())

            next_board, reward, done, truncated, next_info = env.step(action)

            transition = BoardTransition(
                game_id=game_id,
                move_index=move_index,
                score=float(info["score"]),
                max_tile=int(info["max_tile"]),
                observed_action=action,
                observed_move=ACTION_TO_MOVE[action],
                observed_reward=float(reward),
                done=done or truncated,
                board=state_board.tolist(),
                moves_from_episode_end=0,  # Will be updated below
            )

            # Maintain sliding window
            tail_window.append(transition)
            if len(tail_window) > tail_moves:
                tail_window.pop(0)

            state_board = next_board
            info = next_info
            move_index += 1

            if done or truncated:
                # Update moves_from_episode_end for each transition (reverse chronological)
                for i, trans in enumerate(reversed(tail_window)):
                    # Replace transition with one that has correct moves_from_episode_end
                    fixed_trans = BoardTransition(
                        game_id=trans.game_id,
                        move_index=trans.move_index,
                        score=trans.score,
                        max_tile=trans.max_tile,
                        observed_action=trans.observed_action,
                        observed_move=trans.observed_move,
                        observed_reward=trans.observed_reward,
                        done=trans.done,
                        board=trans.board,
                        moves_from_episode_end=i,
                    )
                    tail_window[len(tail_window) - 1 - i] = fixed_trans

                transitions.extend(tail_window)
                break

    cfg = RolloutConfig(
        source_checkpoint=str(checkpoint_path),
        episodes=episodes,
        tail_moves=tail_moves,
        seed=seed,
        created_at=datetime.now(timezone.utc).isoformat(),
        total_transitions=len(transitions),
    )
    return transitions, cfg


def load_rollout_npz(path: Path) -> tuple[list[BoardTransition], RolloutConfig]:
    """Load a rollout board set and config from .npz."""
    data = np.load(path, allow_pickle=True)

    # Extract config from array element
    cfg_array = data["config"]
    cfg_dict = cfg_array[0] if len(cfg_array) > 0 else {}
    cfg = RolloutConfig(
        source_checkpoint=str(cfg_dict.get("source_checkpoint", "")),
        episodes=int(cfg_dict.get("episodes", 0)),
        tail_moves=int(cfg_dict.get("tail_moves", 0)),
        seed=int(cfg_dict.get("seed", 0)),
        created_at=str(cfg_dict.get("created_at", "")),
        total_transitions=int(cfg_dict.get("total_transitions", 0)),
    )

    boards = data["boards"]
    transitions = []
    for board in boards:
        board_data = board["board"]
        if isinstance(board_data, np.ndarray):
            board_data = board_data.tolist()
        transitions.append(
            BoardTransition(
                game_id=int(board["game_id"]),
                move_index=int(board["move_index"]),
                score=float(board["score"]),
                max_tile=int(board["max_tile"]),
                observed_action=int(board["observed_action"]),
                observed_move=str(board["observed_move"]),
                observed_reward=float(board["observed_reward"]),
                done=bool(board["done"]),
                board=board_data,
                moves_from_episode_end=int(board.get("moves_from_episode_end", 0)),
            )
        )

    return transitions, cfg


def save_rollout_npz(
    path: Path,
    transitions: list[BoardTransition],
    cfg: RolloutConfig,
) -> None:
    """Save rollout board set and config to .npz."""
    path.parent.mkdir(parents=True, exist_ok=True)

    boards_data = [t.to_dict() for t in transitions]
    cfg_data = asdict(cfg)

    np.savez(
        path,
        boards=np.array(boards_data, dtype=object),
        config=np.array([cfg_data], dtype=object),
    )


@dataclass(frozen=True)
class BellmanTarget:
    """Bellman TD target for one action."""

    immediate_reward: float
    next_value: float
    discounted_next_value: float
    td_target: float
    current_q: float
    td_delta: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelOutput:
    """Per-model inference output on one board."""

    model_id: str
    q_values: list[float]
    masked_q_values: list[float | None]
    selected_action: int
    selected_move: str
    action_margin: float
    bellman: dict[str, Any]  # {gamma, per_action: [BellmanTarget, ...]}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlannerOutput:
    """Planner (MCTS) output on one board."""

    selected_action: int
    selected_move: str
    q_values: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_bellman_targets(
    *,
    board: np.ndarray,
    q_values_raw: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    gamma: float = 0.99,
) -> dict[str, Any]:
    """
    Compute Bellman TD targets for each action by simulating one step ahead.
    Returns dict with gamma and per_action list of BellmanTarget dicts.
    """
    bellman_targets = []

    # For each action, simulate the step and compute the TD target
    for action in range(4):
        # Create fresh env with current board state
        env = Game2048Env()
        env.game.grid = board.copy()

        # Execute the action to get reward and next state
        next_board, reward, done, truncated, info = env.step(action)

        # Get next value: max Q over legal actions in next state
        if done or truncated:
            next_value = 0.0
        else:
            # Set env to next state and get legal actions
            env.game.grid = next_board.copy()
            next_legal = env.legal_actions()

            state_tensor = torch.as_tensor(
                next_board, dtype=torch.long, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                next_q = model(state_tensor).squeeze(0).cpu().numpy()

            # Max over legal actions
            legal_q = [next_q[a] for a in next_legal]
            next_value = float(max(legal_q)) if legal_q else 0.0

        discounted_next_value = gamma * next_value
        td_target = float(reward) + discounted_next_value
        current_q = float(q_values_raw[action])
        td_delta = td_target - current_q

        target = BellmanTarget(
            immediate_reward=float(reward),
            next_value=next_value,
            discounted_next_value=discounted_next_value,
            td_target=td_target,
            current_q=current_q,
            td_delta=td_delta,
        )
        bellman_targets.append(target.to_dict())

    return {
        "gamma": gamma,
        "per_action": bellman_targets,
    }


def evaluate_board_with_model(
    *,
    board: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    model_id: str,
) -> ModelOutput:
    """Evaluate a single board with a model, return action and Q-values."""
    env = Game2048Env()
    game = env.game
    game.grid = board.copy()
    legal_actions = env.legal_actions()

    action_mask = torch.as_tensor(
        legal_actions_to_mask(4, legal_actions),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    state_tensor = torch.as_tensor(board, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values_raw = model(state_tensor).squeeze(0).cpu().numpy()
        q_tensor = torch.tensor(q_values_raw, device=device).unsqueeze(0)
        masked_q = mask_illegal_actions(q_tensor, action_mask)
        action = int(masked_q.argmax(dim=1).item())

    q_list = q_values_raw.tolist()

    # Serialize masked q-values: legal moves get actual values, illegal moves are null
    # (mask_illegal_actions fills illegal indices with finfo.min sentinel)
    masked_q_numpy = masked_q.squeeze(0).cpu().numpy()
    masked_q_list = []
    for a in range(4):
        # Check if this action was masked (filled with sentinel value)
        if a in legal_actions:
            masked_q_list.append(float(masked_q_numpy[a]))
        else:
            masked_q_list.append(None)

    # Action margin: difference between best and second-best legal action
    legal_q = [q_values_raw[a] for a in legal_actions]
    legal_q_sorted = sorted(legal_q, reverse=True)
    margin = (
        float(legal_q_sorted[0] - legal_q_sorted[1]) if len(legal_q_sorted) > 1 else 0.0
    )

    # Compute Bellman TD targets for analysis
    bellman = compute_bellman_targets(
        board=board,
        q_values_raw=q_values_raw,
        model=model,
        device=device,
    )

    return ModelOutput(
        model_id=model_id,
        q_values=q_list,
        masked_q_values=masked_q_list,
        selected_action=action,
        selected_move=ACTION_TO_MOVE[action],
        action_margin=margin,
        bellman=bellman,
    )


def evaluate_board_with_planner(
    *,
    board: np.ndarray,
    stages: int,
    scenarios: int,
    seed: int,
) -> PlannerOutput:
    """Evaluate a single board with MCTS planner."""
    game = Game2048Env().game
    game.grid = board.copy()

    planned = choose_n_step_mc(
        game=game,
        stages=stages,
        scenarios=scenarios,
        rng=random.Random(seed),
    )
    q_list = list(planned.q_values)

    return PlannerOutput(
        selected_action=planned.action,
        selected_move=planned.move,
        q_values=q_list,
    )


@dataclass(frozen=True)
class CompareOutputBoard:
    """One board's comparison output."""

    board_id: str
    source: dict[str, Any]
    board: list[list[int]]
    legal_actions: list[int]
    planner: dict[str, Any] | None
    models: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_compare(
    *,
    transitions: list[BoardTransition],
    model_paths: list[Path],
    stages: int,
    scenarios: int,
    seed: int,
    device_name: str = "cpu",
) -> tuple[list[CompareOutputBoard], dict[str, Any]]:
    """
    Compare N models on all transitions. Return board-level and summary outputs.
    """
    from training.train import resolve_device

    device = resolve_device(device_name)

    # Load all models
    models: list[tuple[str, torch.nn.Module]] = []
    for i, path in enumerate(model_paths):
        model, _config, _ = load_q_network(path, device_name=device_name)
        models.append((f"model_{i}", model))

    board_outputs: list[CompareOutputBoard] = []

    for i, transition in enumerate(transitions):
        board = np.array(transition.board, dtype=np.int16)
        game = Game2048Env().game
        game.grid = board.copy()
        available_moves = game.available_moves()
        move_to_action = {v: k for k, v in ACTION_TO_MOVE.items()}
        legal_actions = sorted([move_to_action[m] for m in available_moves])

        board_id = f"rollout-{transition.game_id:04d}-move-{transition.move_index:04d}"
        source_dict = {
            "kind": "rollout",
            "collector_model_id": "rollout_source",
            "game_id": transition.game_id,
            "move_index": transition.move_index,
            "score": transition.score,
            "max_tile": transition.max_tile,
            "observed_action": transition.observed_action,
            "observed_move": transition.observed_move,
            "observed_reward": transition.observed_reward,
            "done": transition.done,
            "observed_next_board": None,  # Not saved in this version
            "moves_from_episode_end": transition.moves_from_episode_end,
        }

        # Evaluate planner
        planner_out = evaluate_board_with_planner(
            board=board,
            stages=stages,
            scenarios=scenarios,
            seed=seed + i,
        )

        # Evaluate all models
        model_outs = []
        for model_id, model in models:
            out = evaluate_board_with_model(
                board=board,
                model=model,
                device=device,
                model_id=model_id,
            )
            model_outs.append(out.to_dict())

        cmp_board = CompareOutputBoard(
            board_id=board_id,
            source=source_dict,
            board=board.tolist(),
            legal_actions=legal_actions,
            planner=planner_out.to_dict(),
            models=model_outs,
        )
        board_outputs.append(cmp_board)

    # Compute summary metrics
    summary = _compute_summary(board_outputs)
    return board_outputs, summary


def _compute_summary(boards: list[CompareOutputBoard]) -> dict[str, Any]:
    """Compute agreement and disagreement statistics."""
    if not boards:
        return {"boards_count": 0}

    planner_alignment: dict[str, Any] = {}
    pairwise_alignment: dict[str, Any] = {}
    disagreement_by_action: dict[str, dict[str, int]] = {}
    action_margin_stats: dict[str, dict[str, float]] = {}
    disagreement_by_tail_state: dict[str, dict[str, Any]] = {}
    bellman_metrics: dict[str, dict[str, float]] = {}

    # Extract model IDs from first board
    model_ids = [m["model_id"] for m in boards[0].models] if boards else []

    # Collect moves_from_episode_end for tail state analysis
    model_vs_planner_disagreements: dict[
        str, list[int]
    ] = {}  # model_id -> list of moves_from_episode_end
    pairwise_disagreements: dict[
        str, list[int]
    ] = {}  # pair_key -> list of moves_from_episode_end

    for model_id in model_ids:
        agree_count = 0
        disagree_count = 0
        disagreement_pairs: dict[str, int] = {}
        margin_agree = []
        margin_disagree = []
        model_vs_planner_disagreements[model_id] = []
        # Bellman metrics tracking
        td_delta_aligned = []
        td_delta_disagreed = []
        immediate_reward_shares = []
        discounted_next_value_shares = []

        for board in boards:
            planner_action = board.planner["selected_action"]
            planner_move = board.planner["selected_move"]

            model_output = next(
                (m for m in board.models if m["model_id"] == model_id), None
            )
            if not model_output:
                continue

            model_action = model_output["selected_action"]
            model_move = model_output["selected_move"]
            margin = model_output["action_margin"]

            if model_action == planner_action:
                agree_count += 1
                margin_agree.append(margin)
            else:
                disagree_count += 1
                margin_disagree.append(margin)
                pair_key = f"planner_{planner_move}__model_{model_move}"
                disagreement_pairs[pair_key] = disagreement_pairs.get(pair_key, 0) + 1
                # Track when disagreement happened (by distance from episode end)
                model_vs_planner_disagreements[model_id].append(
                    board.source.get("moves_from_episode_end", 0)
                )

            # Extract bellman metrics from selected action
            bellman_data = model_output.get("bellman", {})
            per_action = bellman_data.get("per_action", [])
            if model_action < len(per_action):
                action_bellman = per_action[model_action]
                if isinstance(action_bellman, dict):
                    td_delta = action_bellman.get("td_delta")
                    td_target = action_bellman.get("td_target")
                    immediate_reward = action_bellman.get("immediate_reward")
                    discounted_next_value = action_bellman.get("discounted_next_value")

                    if td_delta is not None:
                        if model_action == planner_action:
                            td_delta_aligned.append(td_delta)
                        else:
                            td_delta_disagreed.append(td_delta)

                    if td_target is not None and td_target != 0:
                        if immediate_reward is not None:
                            immediate_reward_shares.append(immediate_reward / td_target)
                        if discounted_next_value is not None:
                            discounted_next_value_shares.append(
                                discounted_next_value / td_target
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

    # Pairwise alignment between models
    for i, id_a in enumerate(model_ids):
        for id_b in model_ids[i + 1 :]:
            pair_key = f"{id_a}__{id_b}"
            agree_count = 0
            disagree_count = 0
            pairwise_disagreements[pair_key] = []

            for board in boards:
                out_a = next((m for m in board.models if m["model_id"] == id_a), None)
                out_b = next((m for m in board.models if m["model_id"] == id_b), None)

                if not out_a or not out_b:
                    continue

                if out_a["selected_action"] == out_b["selected_action"]:
                    agree_count += 1
                else:
                    disagree_count += 1
                    pairwise_disagreements[pair_key].append(
                        board.source.get("moves_from_episode_end", 0)
                    )

            total = agree_count + disagree_count
            pairwise_alignment[pair_key] = {
                "agree_count": agree_count,
                "disagree_count": disagree_count,
                "agree_rate": float(agree_count) / total if total > 0 else 0.0,
            }

    # Compute disagreement_by_tail_state with windows: total, last_30, last_20, last_10
    # Windows are based on moves_from_episode_end (distance from episode end)
    def compute_tail_stats(disagreement_list: list[int]) -> dict[str, int]:
        """Compute disagreement counts in tail state windows (last 10, 20, 30 moves)."""
        stats = {"total": len(disagreement_list)}
        for window_size in [30, 20, 10]:
            window_key = f"last_{window_size}"
            # Count disagreements within last window_size moves (moves_from_episode_end <= window_size)
            window_count = sum(1 for d in disagreement_list if d <= window_size)
            stats[window_key] = window_count
        return stats

    # Model vs planner disagreements
    for model_id in model_ids:
        disagreement_by_tail_state[f"{model_id}__planner"] = compute_tail_stats(
            model_vs_planner_disagreements[model_id]
        )

    # Pairwise model disagreements
    for pair_key in pairwise_disagreements:
        disagreement_by_tail_state[pair_key] = compute_tail_stats(
            pairwise_disagreements[pair_key]
        )

    return {
        "run_id": str(uuid.uuid4()),
        "boards_count": len(boards),
        "planner_alignment": planner_alignment,
        "pairwise_alignment": pairwise_alignment,
        "disagreement_by_action": disagreement_by_action,
        "action_margin": action_margin_stats,
        "bellman_metrics": bellman_metrics,
        "disagreement_by_tail_state": disagreement_by_tail_state,
        "diagnostics": {"easy_flags": []},
    }


def run_bellman(compare_paths: list[Path]):
    def _load_boards(path: Path) -> list[dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return []

        try:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError:
            # Fallback: parse concatenated JSON objects with arbitrary whitespace.
            decoder = json.JSONDecoder()
            idx = 0
            length = len(text)
            parsed: list[dict[str, Any]] = []
            while idx < length:
                while idx < length and text[idx].isspace():
                    idx += 1
                if idx >= length:
                    break
                obj, next_idx = decoder.raw_decode(text, idx)
                if isinstance(obj, dict):
                    parsed.append(obj)
                idx = next_idx
            return parsed

    rows: list[dict[str, Any]] = []

    for compare_path in compare_paths:
        boards_path = compare_path / "boards.jsonl"
        boards = _load_boards(boards_path)

        metrics_by_model: dict[str, dict[str, Any]] = {}

        for board in boards:
            planner_q_values = board.get("planner", {}).get("q_values", [])
            if not planner_q_values:
                continue
            planner_best_action = int(np.argmax(np.asarray(planner_q_values)))

            for model in board.get("models", []):
                model_id = str(model["model_id"])
                model_key = f"{compare_path.name}/{model_id}"
                bucket = metrics_by_model.setdefault(
                    model_key,
                    {
                        "current_q": [],
                        "td_target": [],
                        "td_delta": [],
                        "aligned_count": 0,
                    },
                )

                q_values = model.get("q_values", [])
                if not q_values:
                    continue

                model_best_action = int(np.argmax(np.asarray(q_values)))
                selected_q = float(q_values[model_best_action])
                bucket["current_q"].append(selected_q)

                per_action = model.get("bellman", {}).get("per_action", [])
                if model_best_action < len(per_action):
                    selected_bellman = per_action[model_best_action]
                    if isinstance(selected_bellman, dict):
                        td_target = selected_bellman.get("td_target")
                        td_delta = selected_bellman.get("td_delta")
                        if td_target is not None:
                            bucket["td_target"].append(float(td_target))
                        if td_delta is not None:
                            bucket["td_delta"].append(float(td_delta))

                if model_best_action == planner_best_action:
                    bucket["aligned_count"] += 1

        for model_name, values in metrics_by_model.items():
            current_q = values["current_q"]
            td_target = values["td_target"]
            td_delta = values["td_delta"]
            boards_count = len(current_q)
            rows.append(
                {
                    "model": model_name,
                    "boards_count": boards_count,
                    "current_q_mean": float(np.mean(current_q)) if current_q else 0.0,
                    "current_q_p50": float(np.median(current_q)) if current_q else 0.0,
                    "td_target_mean": float(np.mean(td_target)) if td_target else 0.0,
                    "td_target_p50": float(np.median(td_target)) if td_target else 0.0,
                    "td_delta_mean": float(np.mean(td_delta)) if td_delta else 0.0,
                    "td_delta_p50": float(np.median(td_delta)) if td_delta else 0.0,
                    "times_aligned_with_planner": int(values["aligned_count"]),
                }
            )

    rows.sort(key=lambda r: r["model"])

    print(f"[bellman] Summary: {len(rows)} model metrics rows")
    for row in rows:
        boards_count = int(row["boards_count"])
        aligned = int(row["times_aligned_with_planner"])
        align_rate = (aligned / boards_count) if boards_count > 0 else 0.0
        print(
            f"  {row['model']}: "
            f"current_q mean/p50 {row['current_q_mean']:.2f}/{row['current_q_p50']:.2f}, "
            f"td_target mean/p50 {row['td_target_mean']:.2f}/{row['td_target_p50']:.2f}, "
            f"td_delta mean/p50 {row['td_delta_mean']:.2f}/{row['td_delta_p50']:.2f}, "
            f"planner alignment {align_rate:.1%} ({aligned}/{boards_count})"
        )

    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bellman_parser = subparsers.add_parser(
        "bellman",
        help="Aggregate Bellman metrics from compare directories.",
    )
    bellman_parser.add_argument(
        "compare_paths",
        nargs="+",
        type=Path,
        help="Paths to compare run directories containing boards.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "bellman":
        run_bellman(args.compare_paths)


if __name__ == "__main__":
    main()
