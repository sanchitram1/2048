from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import random

import numpy as np

from training.env import Game2048Env
from training.inference import (
    choose_greedy_action,
    find_latest_checkpoint,
    load_q_network,
)
from training.td_ntuple import (
    NTupleValueFunction,
    choose_td_action,
    find_latest_td_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained 2048 model checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint. If omitted, latest checkpoint in models/ is used.",
    )
    parser.add_argument(
        "--model-type",
        choices=("auto", "dqn", "td"),
        default="auto",
        help="Model family for evaluation. 'auto' infers from checkpoint filename.",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--model-dir", type=str, default="models")
    return parser.parse_args()


def _resolve_checkpoint(args: argparse.Namespace) -> tuple[str, Path]:
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model_type = args.model_type
        if model_type == "auto":
            model_type = "td" if path.suffix == ".npz" else "dqn"
        return model_type, path

    if args.model_type == "dqn":
        dqn_path = find_latest_checkpoint(args.model_dir)
        if dqn_path is None:
            raise FileNotFoundError(
                f"No DQN checkpoint found in {args.model_dir}. Run `uv run train` first."
            )
        return "dqn", dqn_path

    if args.model_type == "td":
        td_path = find_latest_td_checkpoint(args.model_dir)
        if td_path is None:
            raise FileNotFoundError(
                f"No TD checkpoint found in {args.model_dir}. Run `uv run train-td` first."
            )
        return "td", td_path

    dqn_path = find_latest_checkpoint(args.model_dir)
    td_path = find_latest_td_checkpoint(args.model_dir)
    if dqn_path is None and td_path is None:
        raise FileNotFoundError(
            f"No checkpoints found in {args.model_dir}. Run `uv run train` or `uv run train-td` first."
        )
    if dqn_path is None:
        assert td_path is not None
        return "td", td_path
    if td_path is None:
        assert dqn_path is not None
        return "dqn", dqn_path
    # Prefer the DQN checkpoint when both exist to match the current app default.
    return "dqn", dqn_path


def _print_summary(*, episodes: int, scores: list[float], max_tiles: list[int]) -> None:
    tile_counts = Counter(max_tiles)

    print("\nTrue 2048 performance")
    print(f"Episodes: {episodes}")

    print("\nTile distribution:")
    for tile in sorted(tile_counts):
        count = tile_counts[tile]
        pct = 100 * count / episodes
        print(f"{tile}: {count}/{episodes} games ({pct:.1f}%)")

    print("\nReach rates:")
    for threshold in [64, 128, 256, 512, 1024]:
        reached = sum(tile >= threshold for tile in max_tiles)
        pct = 100 * reached / episodes
        print(f"Reached {threshold}: {reached}/{episodes} games ({pct:.1f}%)")

    print("\nScore summary:")
    print(f"Mean score: {sum(scores) / len(scores):.2f}")
    print(f"Max score: {max(scores):.2f}")
    print(f"Min score: {min(scores):.2f}")


def _evaluate_dqn(*, checkpoint_path: Path, episodes: int, device_name: str) -> None:
    q_network, config, device = load_q_network(checkpoint_path, device_name=device_name)
    scores: list[float] = []
    max_tiles: list[int] = []

    for i in range(episodes):
        env = Game2048Env()
        env.seed(config.seed + i)
        state, info = env.reset()

        while True:
            legal_actions = env.legal_actions()
            model_action = choose_greedy_action(
                q_network=q_network,
                state=state,
                legal_actions=legal_actions,
                device=device,
            )
            state, _reward, done, truncated, info = env.step(model_action.action)
            if done or truncated:
                scores.append(float(info["score"]))
                max_tiles.append(int(info["max_tile"]))
                break

    print(
        f"Model type: dqn value_network={config.value_network} ({checkpoint_path})"
    )
    _print_summary(episodes=episodes, scores=scores, max_tiles=max_tiles)


def _evaluate_td(*, checkpoint_path: Path, episodes: int) -> None:
    value_function, config, _trained_episodes = NTupleValueFunction.load(checkpoint_path)
    rng_seed = config.seed
    scores: list[float] = []
    max_tiles: list[int] = []

    for i in range(episodes):
        game = Game2048Env().game
        # Follow training/inference behavior: deterministic seed stream per episode.
        random.seed(rng_seed + i)
        np.random.seed(rng_seed + i)
        rng = random.Random(rng_seed + i)
        game.reset()

        while not game.done:
            action = choose_td_action(
                game=game,
                value_function=value_function,
                epsilon=0.0,
                rng=rng,
            )
            game.make_move(Game2048Env.ACTION_TO_MOVE[action.action])

        scores.append(float(game.get_score()))
        max_tiles.append(int(game.max_square()))

    print(f"Model type: td ({checkpoint_path})")
    _print_summary(episodes=episodes, scores=scores, max_tiles=max_tiles)


def main() -> None:
    args = _parse_args()
    try:
        model_type, checkpoint_path = _resolve_checkpoint(args)
        if args.episodes <= 0:
            raise ValueError("--episodes must be > 0")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Diagnostics error: {exc}")
        raise SystemExit(1) from exc

    if model_type == "dqn":
        _evaluate_dqn(
            checkpoint_path=checkpoint_path,
            episodes=args.episodes,
            device_name=args.device,
        )
        return

    _evaluate_td(checkpoint_path=checkpoint_path, episodes=args.episodes)
