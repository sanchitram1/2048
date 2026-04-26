from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from training.config import TrainConfig
from training.dqn import QNetwork
from training.env import Game2048Env
from training.train import resolve_device, select_action


def evaluate(model_path: str, episodes: int = 50) -> None:
    device = resolve_device("cpu")
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )
    if "config" in checkpoint:
        config = TrainConfig(**checkpoint["config"])
    else:
        config = TrainConfig()

    env = Game2048Env()
    action_dim = env.action_space_n()

    q_network = QNetwork(
        action_dim,
        max_exponent=config.max_exponent,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    q_network.load_state_dict(checkpoint["q_network_state_dict"])
    q_network.eval()

    scores = []
    max_tiles = []

    for i in range(episodes):
        env = Game2048Env()
        env.seed(config.seed + i)

        state, _info = env.reset()

        while True:
            legal_actions = env.legal_actions()

            action = select_action(
                q_network=q_network,
                state=state,
                legal_actions=legal_actions,
                epsilon=0.0,
                device=device,
                action_dim=action_dim,
            )

            state, _reward, done, truncated, info = env.step(action)

            if done or truncated:
                scores.append(float(info["score"]))
                max_tiles.append(int(info["max_tile"]))
                break

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a training checkpoint on the 2048 environment.",
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to a checkpoint .pt file produced by training.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of complete games to run.",
    )
    args = parser.parse_args()
    path = args.checkpoint
    if not path.is_file():
        parser.error(f"checkpoint is not a file: {path}")
    evaluate(str(path.resolve()), episodes=args.episodes)


if __name__ == "__main__":
    main()
