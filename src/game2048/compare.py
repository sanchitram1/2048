"""Diagnostic comparison of model outputs on the same board set."""

from __future__ import annotations

import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.dqn import (
    legal_actions_to_mask,
    mask_illegal_actions,
)
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
            )

            # Maintain sliding window
            tail_window.append(transition)
            if len(tail_window) > tail_moves:
                tail_window.pop(0)

            state_board = next_board
            info = next_info
            move_index += 1

            if done or truncated:
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
class ModelOutput:
    """Per-model inference output on one board."""

    model_id: str
    q_values: list[float]
    masked_q_values: list[float | None]
    selected_action: int
    selected_move: str
    action_margin: float

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


def evaluate_board_with_model(
    *,
    board: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    model_id: str,
) -> ModelOutput:
    """Evaluate a single board with a model, return action and Q-values."""
    env = Game2048Env()
    env.board = board.copy()
    legal_actions = env.legal_actions()

    action_mask = torch.as_tensor(
        legal_actions_to_mask(4, legal_actions),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    state_tensor = torch.as_tensor(board, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor).squeeze(0).cpu().numpy()
        masked_q = mask_illegal_actions(
            torch.tensor(q_values, device=device).unsqueeze(0), action_mask
        )
        action = int(masked_q.argmax(dim=1).item())

    q_list = q_values.tolist()
    masked_q_list = [None] * 4
    for a in legal_actions:
        masked_q_list[a] = float(q_values[a])

    # Action margin: difference between best and second-best legal action
    legal_q = [q_values[a] for a in legal_actions]
    legal_q_sorted = sorted(legal_q, reverse=True)
    margin = (
        float(legal_q_sorted[0] - legal_q_sorted[1]) if len(legal_q_sorted) > 1 else 0.0
    )

    return ModelOutput(
        model_id=model_id,
        q_values=q_list,
        masked_q_values=masked_q_list,
        selected_action=action,
        selected_move=ACTION_TO_MOVE[action],
        action_margin=margin,
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

    # Extract model IDs from first board
    model_ids = [m["model_id"] for m in boards[0].models] if boards else []

    for model_id in model_ids:
        agree_count = 0
        disagree_count = 0
        disagreement_pairs: dict[str, int] = {}
        margin_agree = []
        margin_disagree = []

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

    # Pairwise alignment between models
    for i, id_a in enumerate(model_ids):
        for id_b in model_ids[i + 1 :]:
            pair_key = f"{id_a}__{id_b}"
            agree_count = 0
            disagree_count = 0

            for board in boards:
                out_a = next((m for m in board.models if m["model_id"] == id_a), None)
                out_b = next((m for m in board.models if m["model_id"] == id_b), None)

                if not out_a or not out_b:
                    continue

                if out_a["selected_action"] == out_b["selected_action"]:
                    agree_count += 1
                else:
                    disagree_count += 1

            total = agree_count + disagree_count
            pairwise_alignment[pair_key] = {
                "agree_count": agree_count,
                "disagree_count": disagree_count,
                "agree_rate": float(agree_count) / total if total > 0 else 0.0,
            }

    return {
        "run_id": str(uuid.uuid4()),
        "boards_count": len(boards),
        "planner_alignment": planner_alignment,
        "pairwise_alignment": pairwise_alignment,
        "disagreement_by_action": disagreement_by_action,
        "action_margin": action_margin_stats,
        "diagnostics": {"easy_flags": []},
    }
