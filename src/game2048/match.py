"""Synchronized human vs agent match session for /ws/match."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from game2048.game import GameLogic
from game2048.game_logger import GameLogger
from training.env import Game2048Env


def new_game_from_seed(seed: int) -> GameLogic:
    """Create a GameLogic whose initial spawns match Game2048Env(seed).reset()."""
    random.seed(seed)
    np.random.seed(seed)
    env = Game2048Env()
    env.seed(seed)
    env.reset()
    return env.game


def _human_tiles(game: GameLogic) -> list[int]:
    grid = game.get_board()
    size = game.grid_size
    return [int(grid[row, col]) for row in range(size) for col in range(size)]


def _max_tile_from_exponents(tiles: list[int]) -> int:
    highest = max(tiles, default=0)
    return 0 if highest == 0 else 2**highest


class ModelMissingError(Exception):
    """Raised when a checkpoint is required but missing."""


FirstFinished = Literal["human", "agent"] | None


@dataclass
class _GreedyPlan:
    action: int
    move: str
    q_values: tuple[float, ...]
    legal_actions: tuple[int, ...]


class MatchDQNAgent:
    def __init__(self, checkpoint_path: Path) -> None:
        from training.inference import GreedyAgentRunner

        self._runner = GreedyAgentRunner(checkpoint_path=checkpoint_path)

    def reset_episode(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._runner.env = Game2048Env()
        self._runner.env.seed(seed)
        self._runner.move_count = 0
        self._runner.state, self._runner.info = self._runner.env.reset()

    def step(self) -> dict[str, object]:
        return self._runner.step()

    def view(self) -> dict[str, object]:
        max_tile = int(self._runner.info["max_tile"])
        return {
            "tiles": [int(t) for t in self._runner.state.flatten()],
            "score": int(self._runner.info["score"]),
            "move_count": self._runner.move_count,
            "done": bool(self._runner.info["done"]),
            "max_tile": max_tile,
            "model_type": "dqn",
            "checkpoint": str(self._runner.checkpoint_path),
        }


class MatchTDAgent:
    def __init__(self, checkpoint_path: Path) -> None:
        from training.td_ntuple import NTupleValueFunction, choose_td_action

        self._choose_td_action = choose_td_action
        self.value_function, self.config, _episodes = NTupleValueFunction.load(
            checkpoint_path
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.game = GameLogic()
        self.rng = random.Random(0)
        self.move_count = 0

    def reset_episode(self, seed: int) -> None:
        self.game = new_game_from_seed(seed)
        self.rng = random.Random(seed)
        self.move_count = 0

    def step(self) -> dict[str, object]:
        if self.game.done or not self.game.available_moves():
            return self._payload(event="game_over", model_action=None)

        model_action = self._choose_td_action(
            game=self.game,
            value_function=self.value_function,
            epsilon=0.0,
            rng=self.rng,
        )
        self.game.make_move(Game2048Env.ACTION_TO_MOVE[model_action.action])
        self.move_count += 1
        event = "game_over" if self.game.done else "agent_move"
        return self._payload(event=event, model_action=model_action)

    def _payload(
        self,
        *,
        event: str,
        model_action: Any | None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": event,
            "tiles": [int(t) for t in self.game.get_board().flatten()],
            "score": self.game.get_score(),
            "move_count": self.move_count,
            "done": self.game.done,
            "max_tile": self.game.max_square(),
            "checkpoint": str(self.checkpoint_path),
            "model_type": "td_ntuple",
        }
        if model_action is not None:
            payload.update(
                {
                    "move": model_action.move,
                    "action": model_action.action,
                    "q_values": list(model_action.action_values),
                    "legal_actions": list(model_action.legal_actions),
                }
            )
        return payload

    def view(self) -> dict[str, object]:
        return {
            "tiles": _human_tiles(self.game),
            "score": self.game.get_score(),
            "move_count": self.move_count,
            "done": self.game.done,
            "max_tile": self.game.max_square(),
            "model_type": "td_ntuple",
            "checkpoint": str(self.checkpoint_path),
        }


class MatchGreedyAgent:
    def __init__(self) -> None:
        from training.planning import ACTION_TO_MOVE, choose_myopic_greedy

        self._choose = choose_myopic_greedy
        self._ACTION_TO_MOVE = ACTION_TO_MOVE
        self.game = GameLogic()
        self.rng = random.Random(0)
        self.move_count = 0

    def reset_episode(self, seed: int) -> None:
        self.game = new_game_from_seed(seed)
        self.rng = random.Random(seed)
        self.move_count = 0

    def step(self) -> dict[str, object]:
        if self.game.done or not self.game.available_moves():
            return self._payload(event="game_over", model_action=None)

        planned = self._choose(self.game, rng=self.rng)
        self.game.make_move(self._ACTION_TO_MOVE[planned.action])
        self.move_count += 1
        model = _GreedyPlan(
            action=planned.action,
            move=planned.move,
            q_values=planned.q_values,
            legal_actions=planned.legal_actions,
        )
        return self._payload(
            event="game_over" if self.game.done else "agent_move",
            model_action=model,
        )

    def _payload(self, *, event: str, model_action: _GreedyPlan | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": event,
            "tiles": [int(t) for t in self.game.get_board().flatten()],
            "score": self.game.get_score(),
            "move_count": self.move_count,
            "done": self.game.done,
            "max_tile": self.game.max_square(),
            "checkpoint": None,
            "model_type": "greedy_myopic",
        }
        if model_action is not None:
            payload.update(
                {
                    "move": model_action.move,
                    "action": model_action.action,
                    "q_values": list(model_action.q_values),
                    "legal_actions": list(model_action.legal_actions),
                }
            )
        return payload

    def view(self) -> dict[str, object]:
        return {
            "tiles": _human_tiles(self.game),
            "score": self.game.get_score(),
            "move_count": self.move_count,
            "done": self.game.done,
            "max_tile": self.game.max_square(),
            "model_type": "greedy_myopic",
            "checkpoint": None,
        }


class MatchNStepAgent:
    def __init__(self) -> None:
        from training.planning import (
            ACTION_NAMES,
            ACTION_TO_MOVE,
            choose_n_step_mc,
        )

        self._choose = choose_n_step_mc
        self._ACTION_TO_MOVE = ACTION_TO_MOVE
        self._ACTION_NAMES = ACTION_NAMES
        self.game = GameLogic()
        self.rng = random.Random(0)
        self.move_count = 0

    def reset_episode(self, seed: int) -> None:
        self.game = new_game_from_seed(seed)
        self.rng = random.Random(seed)
        self.move_count = 0

    def step(self) -> dict[str, object]:
        if self.game.done or not self.game.available_moves():
            return self._payload(event="game_over", model_action=None)

        planned = self._choose(self.game, rng=self.rng)
        self.game.make_move(self._ACTION_TO_MOVE[planned.action])
        self.move_count += 1
        model = _GreedyPlan(
            action=planned.action,
            move=planned.move,
            q_values=planned.q_values,
            legal_actions=planned.legal_actions,
        )
        return self._payload(
            event="game_over" if self.game.done else "agent_move",
            model_action=model,
        )

    def _payload(self, *, event: str, model_action: _GreedyPlan | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": event,
            "tiles": [int(t) for t in self.game.get_board().flatten()],
            "score": self.game.get_score(),
            "move_count": self.move_count,
            "done": self.game.done,
            "max_tile": self.game.max_square(),
            "checkpoint": None,
            "model_type": "mc_3step",
        }
        if model_action is not None:
            payload.update(
                {
                    "move": model_action.move,
                    "action": model_action.action,
                    "q_values": list(model_action.q_values),
                    "legal_actions": list(model_action.legal_actions),
                }
            )
        return payload

    def view(self) -> dict[str, object]:
        return {
            "tiles": _human_tiles(self.game),
            "score": self.game.get_score(),
            "move_count": self.move_count,
            "done": self.game.done,
            "max_tile": self.game.max_square(),
            "model_type": "mc_3step",
            "checkpoint": None,
        }


MatchAgent = MatchDQNAgent | MatchTDAgent | MatchGreedyAgent | MatchNStepAgent

_MISSING_MSG = "no checkpoint found in models..."


def build_match_agent(agent_type: str) -> MatchAgent:
    at = agent_type.lower()
    if at in {"greedy", "myopic"}:
        return MatchGreedyAgent()
    if at in {"mc", "mcts", "nstep"}:
        return MatchNStepAgent()
    if at == "dqn":
        from training.inference import find_latest_checkpoint

        path = find_latest_checkpoint()
        if path is None:
            raise ModelMissingError(_MISSING_MSG)
        return MatchDQNAgent(path)
    if at == "td":
        from training.td_ntuple import find_latest_td_checkpoint

        path = find_latest_td_checkpoint()
        if path is None:
            raise ModelMissingError(_MISSING_MSG)
        return MatchTDAgent(path)
    raise ValueError(f"Unknown agent type: {agent_type}")


def decide_winner(
    *,
    human_score: int,
    human_max_tile: int,
    human_moves: int,
    agent_score: int,
    agent_max_tile: int,
    agent_moves: int,
) -> tuple[Literal["human", "agent", "tie"], str]:
    if human_score > agent_score:
        return "human", f"You win by score (+{human_score - agent_score})."
    if agent_score > human_score:
        return "agent", f"Agent wins by score (+{agent_score - human_score})."
    if human_max_tile > agent_max_tile:
        return "human", f"You win by max tile ({human_max_tile} vs {agent_max_tile})."
    if agent_max_tile > human_max_tile:
        return "agent", f"Agent wins by max tile ({agent_max_tile} vs {human_max_tile})."
    if human_moves < agent_moves:
        return (
            "human",
            f"You win by fewer moves ({human_moves} vs {agent_moves}, -{agent_moves - human_moves}).",
        )
    if agent_moves < human_moves:
        return (
            "agent",
            f"Agent wins by fewer moves ({agent_moves} vs {human_moves}, -{human_moves - agent_moves}).",
        )
    return "tie", "Tie."


class MatchSession:
    def __init__(
        self,
        *,
        seed: int,
        agent_type: str,
        human_game: GameLogic,
        agent: MatchAgent,
    ) -> None:
        self.seed = seed
        self.agent_type = agent_type
        self.human_game = human_game
        self.human_move_count = 0
        self.agent = agent
        self.logger = GameLogger("HUMAN")
        self.first_finished: FirstFinished = None

    @classmethod
    def start(cls, *, agent_type: str, seed: int) -> MatchSession:
        human_game = new_game_from_seed(seed)
        agent = build_match_agent(agent_type)
        agent.reset_episode(seed)
        return cls(seed=seed, agent_type=agent_type.lower(), human_game=human_game, agent=agent)

    def reset(self, *, seed: int, agent_type: str | None = None) -> None:
        at = (agent_type or self.agent_type).lower()
        self.seed = seed
        self.agent_type = at
        self.human_game = new_game_from_seed(seed)
        self.human_move_count = 0
        self.first_finished = None
        self.agent = build_match_agent(at)
        self.agent.reset_episode(seed)

    @property
    def human_done(self) -> bool:
        return self.human_game.done

    @property
    def agent_done(self) -> bool:
        v = self.agent.view()
        return bool(v["done"])

    @property
    def match_done(self) -> bool:
        return self.human_done and self.agent_done

    def _human_view(self) -> dict[str, object]:
        tiles = _human_tiles(self.human_game)
        return {
            "tiles": tiles,
            "score": self.human_game.get_score(),
            "move_count": self.human_move_count,
            "done": self.human_game.done,
            "max_tile": _max_tile_from_exponents(tiles),
        }

    def _merge_agent_into_view(
        self, agent_view: dict[str, object], step_payload: dict[str, object] | None
    ) -> dict[str, object]:
        out = dict(agent_view)
        if step_payload:
            for key in ("move", "action", "q_values", "legal_actions", "event"):
                if key in step_payload:
                    out[key] = step_payload[key]
        return out

    def snapshot(
        self,
        event: str,
        *,
        last_human_moved: bool = False,
        last_agent_moved: bool = False,
        last_agent_step: dict[str, object] | None = None,
        log_line: str | None = None,
        error: str | None = None,
    ) -> dict[str, object]:
        agent_view = self.agent.view()
        match_payload = self._match_meta(
            last_human_moved=last_human_moved,
            last_agent_moved=last_agent_moved,
            error=error,
        )
        out: dict[str, object] = {
            "event": event,
            "human": self._human_view(),
            "agent": self._merge_agent_into_view(agent_view, last_agent_step),
            "match": match_payload,
            "last_human_moved": last_human_moved,
            "last_agent_moved": last_agent_moved,
            "log_line": log_line,
        }
        if error:
            out["message"] = error
        return out

    def _match_meta(
        self,
        *,
        last_human_moved: bool,
        last_agent_moved: bool,
        error: str | None,
    ) -> dict[str, object]:
        winner: str | None = None
        win_reason: str | None = None
        if self.match_done:
            hv = self._human_view()
            av = self.agent.view()
            w, reason = decide_winner(
                human_score=int(hv["score"]),
                human_max_tile=int(hv["max_tile"]),
                human_moves=int(hv["move_count"]),
                agent_score=int(av["score"]),
                agent_max_tile=int(av["max_tile"]),
                agent_moves=int(av["move_count"]),
            )
            winner = w
            win_reason = reason
        return {
            "match_done": self.match_done,
            "human_done": self.human_done,
            "agent_done": self.agent_done,
            "first_finished": self.first_finished,
            "winner": winner,
            "win_reason": win_reason,
            "seed": self.seed,
            "fairness_note": "Seed-matched start; trajectories diverge after your moves.",
            "last_human_moved": last_human_moved,
            "last_agent_moved": last_agent_moved,
            "error": error,
        }

    def human_move(self, move: str) -> dict[str, object]:
        if self.match_done:
            return self.snapshot(
                "match_complete",
                last_human_moved=False,
                last_agent_moved=False,
            )
        if self.human_game.done:
            return self.snapshot(
                "error",
                error="Human is done — use step_agent to advance the agent.",
            )

        changed, _gain, spawn_flat, spawn_val = self.human_game.make_move(move)
        log_line: str | None = None
        if changed:
            self.human_move_count += 1
            if spawn_flat is not None and spawn_val is not None:
                log_line = self.logger.line_for_move(move, spawn_val, spawn_flat)

        if not changed:
            return self.snapshot(
                "turn_result",
                last_human_moved=False,
                last_agent_moved=False,
                log_line=log_line,
            )

        last_agent_step: dict[str, object] | None = None
        agent_moved = False

        if self.human_game.done:
            if self.first_finished is None:
                self.first_finished = "human"
        elif not self.agent_done:
            last_agent_step = self.agent.step()
            agent_moved = True
            if last_agent_step.get("done") or last_agent_step.get("event") == "game_over":
                if self.first_finished is None:
                    self.first_finished = "agent"

        evt = "match_complete" if self.match_done else "turn_result"
        return self.snapshot(
            evt,
            last_human_moved=True,
            last_agent_moved=agent_moved,
            last_agent_step=last_agent_step,
            log_line=log_line,
        )

    def step_agent(self) -> dict[str, object]:
        if self.match_done:
            return self.snapshot("match_complete")
        if not self.human_game.done:
            return self.snapshot(
                "error",
                error="Human is still playing — make a move first.",
            )
        if self.agent_done:
            return self.snapshot(
                "error",
                error="Agent is already done.",
            )

        last_agent_step = self.agent.step()
        if last_agent_step.get("done") or last_agent_step.get("event") == "game_over":
            if self.first_finished is None:
                self.first_finished = "agent"

        evt = "match_complete" if self.match_done else "turn_result"
        return self.snapshot(
            evt,
            last_human_moved=False,
            last_agent_moved=True,
            last_agent_step=last_agent_step,
        )
