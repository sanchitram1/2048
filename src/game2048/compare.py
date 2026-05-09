"""Diagnostic comparison of model outputs on the same board set.

This module owns **board collection and per-board model/planner
evaluation** for the ``compare`` diagnostic. Metric aggregation lives in
:mod:`evaluation.metrics`; rollout NPZ I/O lives in
:mod:`evaluation.rollout`.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.metrics import compute_summary
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

    # Canonical metric aggregation lives in evaluation.metrics.
    summary = compute_summary(board_outputs)
    return board_outputs, summary
