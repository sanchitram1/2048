from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from game2048.game import GameLogic
from training.env import Game2048Env


TD_CHECKPOINT_PATTERN = re.compile(r"td_ntuple_checkpoint_(\d+)\.npz$")
ACTION_NAMES = {0: "left", 1: "right", 2: "up", 3: "down"}


@dataclass(frozen=True)
class TDNTupleConfig:
    """Configuration for a Szubert/Jaskowski-style TD-afterstate baseline."""

    episodes: int = 1_000
    alpha: float = 0.0025
    gamma: float = 1.0
    epsilon_start: float = 0.10
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 750
    max_exponent: int = 15
    seed: int = 7
    checkpoint_interval: int = 100
    log_interval: int = 25
    model_dir: str = "models"
    scale_updates_by_tuples: bool = True


@dataclass(frozen=True)
class TDAction:
    action: int
    move: str
    afterstate: np.ndarray
    score_gain: int
    action_values: tuple[float, ...]
    legal_actions: tuple[int, ...]


def systematic_4_tuples() -> tuple[tuple[int, ...], ...]:
    """Return the 17 four-cell tuples from the CIG 2014 2048 paper."""
    tuples: list[tuple[int, ...]] = []

    for row in range(4):
        tuples.append(tuple(row * 4 + col for col in range(4)))
    for col in range(4):
        tuples.append(tuple(row * 4 + col for row in range(4)))
    for row in range(3):
        for col in range(3):
            tuples.append(
                (
                    row * 4 + col,
                    row * 4 + col + 1,
                    (row + 1) * 4 + col,
                    (row + 1) * 4 + col + 1,
                )
            )

    return tuple(tuples)


class NTupleValueFunction:
    """Tabular n-tuple approximator for 2048 log2-encoded boards."""

    def __init__(
        self,
        *,
        tuples: tuple[tuple[int, ...], ...] | None = None,
        max_exponent: int = 15,
        weights: np.ndarray | None = None,
    ) -> None:
        self.tuples = tuples or systematic_4_tuples()
        if not self.tuples:
            raise ValueError("At least one tuple is required")
        tuple_lengths = {len(tuple_) for tuple_ in self.tuples}
        if len(tuple_lengths) != 1:
            raise ValueError("This baseline expects all tuples to have equal length")

        self.max_exponent = int(max_exponent)
        self.base = self.max_exponent + 1
        self.tuple_length = tuple_lengths.pop()
        self.table_size = self.base**self.tuple_length

        if weights is None:
            self.weights = np.zeros(
                (len(self.tuples), self.table_size),
                dtype=np.float32,
            )
        else:
            expected_shape = (len(self.tuples), self.table_size)
            if weights.shape != expected_shape:
                raise ValueError(
                    f"Expected weights shape {expected_shape}, got {weights.shape}"
                )
            self.weights = weights.astype(np.float32, copy=True)

    def _tuple_indices(self, board: np.ndarray) -> np.ndarray:
        flat = np.asarray(board, dtype=np.int16).reshape(-1)
        clipped = np.clip(flat, 0, self.max_exponent)
        indices = np.zeros(len(self.tuples), dtype=np.int64)
        for tuple_idx, locations in enumerate(self.tuples):
            code = 0
            for location in locations:
                code = code * self.base + int(clipped[location])
            indices[tuple_idx] = code
        return indices

    def value(self, board: np.ndarray) -> float:
        indices = self._tuple_indices(board)
        return float(self.weights[np.arange(len(self.tuples)), indices].sum())

    def update(self, board: np.ndarray, td_error: float, alpha: float) -> None:
        indices = self._tuple_indices(board)
        step_size = alpha / len(self.tuples)
        self.weights[np.arange(len(self.tuples)), indices] += step_size * td_error

    def save(self, path: str | Path, config: TDNTupleConfig, episodes: int) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "config": asdict(config),
            "episodes": int(episodes),
            "tuples": [list(tuple_) for tuple_ in self.tuples],
        }
        np.savez_compressed(
            checkpoint_path,
            weights=self.weights,
            metadata=json.dumps(metadata),
        )
        return checkpoint_path

    @classmethod
    def load(
        cls, checkpoint_path: str | Path
    ) -> tuple[NTupleValueFunction, TDNTupleConfig, int]:
        data = np.load(checkpoint_path, allow_pickle=False)
        metadata = json.loads(str(data["metadata"]))
        config = TDNTupleConfig(**metadata["config"])
        tuples = tuple(tuple(int(item) for item in tuple_) for tuple_ in metadata["tuples"])
        value_function = cls(
            tuples=tuples,
            max_exponent=config.max_exponent,
            weights=data["weights"],
        )
        return value_function, config, int(metadata["episodes"])


def linear_epsilon_episode(episode: int, config: TDNTupleConfig) -> float:
    if config.epsilon_decay_episodes <= 0:
        return config.epsilon_end
    progress = min(episode / config.epsilon_decay_episodes, 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def choose_td_action(
    *,
    game: GameLogic,
    value_function: NTupleValueFunction,
    epsilon: float,
    rng: random.Random,
) -> TDAction:
    legal_actions: list[int] = []
    action_values = np.full(Game2048Env.action_space_n(), -1.0e9, dtype=np.float32)
    afterstates: dict[int, tuple[np.ndarray, int]] = {}

    for action, move in Game2048Env.ACTION_TO_MOVE.items():
        afterstate, score_gain, moved = game.preview_move(move)
        if not moved:
            continue
        legal_actions.append(action)
        afterstates[action] = (afterstate, score_gain)
        action_values[action] = float(score_gain) + value_function.value(afterstate)

    if not legal_actions:
        raise RuntimeError("Cannot choose TD action: no legal actions available")

    if rng.random() < epsilon:
        action = int(rng.choice(legal_actions))
    else:
        best_value = max(float(action_values[action]) for action in legal_actions)
        best_actions = [
            action
            for action in legal_actions
            if np.isclose(float(action_values[action]), best_value)
        ]
        action = int(rng.choice(best_actions))

    afterstate, score_gain = afterstates[action]
    return TDAction(
        action=action,
        move=ACTION_NAMES[action],
        afterstate=afterstate,
        score_gain=score_gain,
        action_values=tuple(float(value) for value in action_values),
        legal_actions=tuple(legal_actions),
    )


def find_latest_td_checkpoint(model_dir: str | Path = "models") -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in Path(model_dir).glob("td_ntuple_checkpoint_*.npz"):
        match = TD_CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def save_td_checkpoint(
    *,
    value_function: NTupleValueFunction,
    config: TDNTupleConfig,
    episodes_completed: int,
) -> Path:
    model_path = Path(config.model_dir)
    return value_function.save(
        model_path / f"td_ntuple_checkpoint_{episodes_completed}.npz",
        config,
        episodes_completed,
    )


def train_td_ntuple(config: TDNTupleConfig) -> None:
    rng = random.Random(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    value_function = NTupleValueFunction(max_exponent=config.max_exponent)
    scores: list[float] = []
    max_tiles: list[float] = []
    lengths: list[float] = []

    print(
        "starting td-ntuple training",
        f"episodes={config.episodes}",
        f"alpha={config.alpha}",
    )

    for episode in range(1, config.episodes + 1):
        game = GameLogic()
        epsilon = linear_epsilon_episode(episode, config)
        move_count = 0

        while not game.done:
            current = choose_td_action(
                game=game,
                value_function=value_function,
                epsilon=epsilon,
                rng=rng,
            )
            changed, score_gain, _spawn_flat, _spawn_value = game.make_move(
                Game2048Env.ACTION_TO_MOVE[current.action]
            )
            if not changed:
                raise RuntimeError("TD policy selected an illegal move")

            move_count += 1
            current_value = value_function.value(current.afterstate)
            if game.done:
                target = 0.0
            else:
                next_action = choose_td_action(
                    game=game,
                    value_function=value_function,
                    epsilon=epsilon,
                    rng=rng,
                )
                target = float(next_action.score_gain) + (
                    config.gamma * value_function.value(next_action.afterstate)
                )

            alpha = config.alpha
            if not config.scale_updates_by_tuples:
                alpha *= len(value_function.tuples)
            value_function.update(current.afterstate, target - current_value, alpha)

            if score_gain != current.score_gain:
                raise RuntimeError("Previewed score gain did not match executed move")

        scores.append(float(game.get_score()))
        max_tiles.append(float(game.max_square()))
        lengths.append(float(move_count))

        if config.log_interval > 0 and episode % config.log_interval == 0:
            print(
                f"episode={episode} "
                f"epsilon={epsilon:.3f} "
                f"mean_score={np.mean(scores[-config.log_interval:]):.1f} "
                f"mean_max_tile={np.mean(max_tiles[-config.log_interval:]):.1f} "
                f"mean_length={np.mean(lengths[-config.log_interval:]):.1f}"
            )

        if (
            config.checkpoint_interval > 0
            and episode % config.checkpoint_interval == 0
        ):
            checkpoint_path = save_td_checkpoint(
                value_function=value_function,
                config=config,
                episodes_completed=episode,
            )
            print(f"saved td checkpoint to {checkpoint_path}")

    final_checkpoint_path = save_td_checkpoint(
        value_function=value_function,
        config=config,
        episodes_completed=config.episodes,
    )
    print(f"td training complete saved checkpoint to {final_checkpoint_path}")


class TDNTupleAgentRunner:
    def __init__(self, *, checkpoint_path: str | Path) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.value_function, self.config, self.episodes = NTupleValueFunction.load(
            self.checkpoint_path
        )
        self.rng = random.Random(self.config.seed)
        self.game = GameLogic()
        self.move_count = 0

    def reset(self) -> dict[str, object]:
        self.game = GameLogic()
        self.move_count = 0
        return self.payload(event="state", model_action=None)

    def step(self) -> dict[str, object]:
        if self.game.done or not self.game.available_moves():
            return self.payload(event="game_over", model_action=None)

        model_action = choose_td_action(
            game=self.game,
            value_function=self.value_function,
            epsilon=0.0,
            rng=self.rng,
        )
        self.game.make_move(Game2048Env.ACTION_TO_MOVE[model_action.action])
        self.move_count += 1
        event = "game_over" if self.game.done else "agent_move"
        return self.payload(event=event, model_action=model_action)

    def payload(
        self,
        *,
        event: str,
        model_action: TDAction | None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": event,
            "tiles": [int(tile) for tile in self.game.get_board().flatten()],
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


def parse_args() -> TDNTupleConfig:
    parser = argparse.ArgumentParser(
        description="Train a TD(0) afterstate n-tuple baseline for 2048."
    )
    parser.add_argument("--episodes", type=int, default=TDNTupleConfig.episodes)
    parser.add_argument("--alpha", type=float, default=TDNTupleConfig.alpha)
    parser.add_argument("--gamma", type=float, default=TDNTupleConfig.gamma)
    parser.add_argument(
        "--epsilon-start", type=float, default=TDNTupleConfig.epsilon_start
    )
    parser.add_argument("--epsilon-end", type=float, default=TDNTupleConfig.epsilon_end)
    parser.add_argument(
        "--epsilon-decay-episodes",
        type=int,
        default=TDNTupleConfig.epsilon_decay_episodes,
    )
    parser.add_argument("--max-exponent", type=int, default=TDNTupleConfig.max_exponent)
    parser.add_argument("--seed", type=int, default=TDNTupleConfig.seed)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=TDNTupleConfig.checkpoint_interval,
    )
    parser.add_argument("--log-interval", type=int, default=TDNTupleConfig.log_interval)
    parser.add_argument("--model-dir", default=TDNTupleConfig.model_dir)
    parser.add_argument(
        "--unscaled-updates",
        action="store_true",
        help="Apply alpha to every tuple weight directly, matching the simplest TD rule.",
    )
    args = parser.parse_args()
    values = vars(args)
    values["scale_updates_by_tuples"] = not values.pop("unscaled_updates")
    return TDNTupleConfig(**values)


def main() -> None:
    train_td_ntuple(parse_args())


if __name__ == "__main__":
    main()
