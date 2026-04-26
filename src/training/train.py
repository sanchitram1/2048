from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.config import TrainConfig
from training.dqn import (
    QNetwork,
    ReplayBatch,
    ReplayBuffer,
    legal_actions_to_mask,
    linear_epsilon,
    mask_illegal_actions,
)
from training.env import Game2048Env


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a masked Double DQN agent for 2048.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--steps", type=int, default=TrainConfig.steps, help="Total env steps to run."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TrainConfig.batch_size,
        help="Replay batch size for each gradient step.",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=TrainConfig.replay_capacity,
        help="Max transitions in replay buffer.",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=TrainConfig.learning_starts,
        help="Step index before any gradient updates begin.",
    )
    parser.add_argument(
        "--train-frequency",
        type=int,
        default=TrainConfig.train_frequency,
        help="Run a gradient update every this many steps.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=TrainConfig.target_update_interval,
        help="Copy online weights to target every this many steps.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=TrainConfig.checkpoint_interval,
        help="Save a checkpoint every this many steps; 0 to disable.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=TrainConfig.eval_interval,
        help="Run eval every this many steps; 0 to disable.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=TrainConfig.eval_episodes,
        help="Episodes per eval run.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=TrainConfig.log_interval,
        help="Log training stats every this many steps; 0 to disable.",
    )
    parser.add_argument(
        "--gamma", type=float, default=TrainConfig.gamma, help="TD discount."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TrainConfig.learning_rate,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=TrainConfig.epsilon_start,
        help="Initial epsilon for epsilon-greedy.",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=TrainConfig.epsilon_end,
        help="Final epsilon after decay.",
    )
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=TrainConfig.epsilon_decay_steps,
        help="Linear epsilon decay length in steps.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=TrainConfig.grad_clip,
        help="Max L2 norm for gradient clipping.",
    )
    parser.add_argument(
        "--seed", type=int, default=TrainConfig.seed, help="Random seed."
    )
    parser.add_argument(
        "--max-exponent",
        type=int,
        default=TrainConfig.max_exponent,
        help="Max tile value as 2**max_exponent (board encoding).",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=TrainConfig.embedding_dim,
        help="Per-tile embedding size.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=TrainConfig.hidden_dim,
        help="MLP hidden width.",
    )
    parser.add_argument(
        "--model-dir", default=TrainConfig.model_dir, help="Directory for checkpoints."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=TrainConfig.device,
        help="Compute device (auto picks cuda, then mps, else cpu).",
    )
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_action(
    *,
    q_network: QNetwork,
    state: np.ndarray,
    legal_actions: list[int],
    epsilon: float,
    device: torch.device,
    action_dim: int,
) -> int:
    if not legal_actions:
        raise RuntimeError("Cannot select an action when no legal actions exist")

    if random.random() < epsilon:
        return int(random.choice(legal_actions))

    action_mask = torch.as_tensor(
        legal_actions_to_mask(action_dim, legal_actions),
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)
    state_tensor = torch.as_tensor(state, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = q_network(state_tensor)
        masked_q_values = mask_illegal_actions(q_values, action_mask)
        return int(masked_q_values.argmax(dim=1).item())


def compute_td_loss(
    *,
    batch: ReplayBatch,
    online_network: QNetwork,
    target_network: QNetwork,
    gamma: float,
) -> torch.Tensor:
    current_q_values = online_network(batch.states)
    chosen_q_values = current_q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        online_next_q_values = online_network(batch.next_states)
        masked_online_next_q_values = mask_illegal_actions(
            online_next_q_values,
            batch.next_action_masks,
        )
        greedy_next_actions = masked_online_next_q_values.argmax(dim=1)

        target_next_q_values = target_network(batch.next_states)
        greedy_target_q_values = target_next_q_values.gather(
            1,
            greedy_next_actions.unsqueeze(1),
        ).squeeze(1)

        has_next_action = batch.next_action_masks.any(dim=1)
        greedy_target_q_values = greedy_target_q_values.masked_fill(
            ~has_next_action, 0.0
        )
        td_target = (
            batch.rewards + gamma * (~batch.dones).float() * greedy_target_q_values
        )

    return F.smooth_l1_loss(chosen_q_values, td_target)


def evaluate_policy(
    *,
    q_network: QNetwork,
    action_dim: int,
    device: torch.device,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    if episodes <= 0:
        return {}

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()
    else:
        cuda_states = None

    try:
        seed_everything(seed)
        env = Game2048Env()
        scores: list[float] = []
        max_tiles: list[float] = []
        for _ in range(episodes):
            state, _ = env.reset()
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
                state, _, done, truncated, info = env.step(action)
                if done or truncated:
                    scores.append(float(info["score"]))
                    max_tiles.append(float(info["max_tile"]))
                    break
        return {
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "mean_max_tile": float(np.mean(max_tiles)),
        }
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(
    *,
    model_path: Path,
    step: int,
    episodes_completed: int,
    q_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> Path:
    model_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_path / f"checkpoint_{step}.pt"
    torch.save(
        {
            "step": step,
            "episodes_completed": episodes_completed,
            "config": asdict(config),
            "q_network_state_dict": q_network.state_dict(),
            "target_network_state_dict": target_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def format_metrics(items: Iterable[tuple[str, float]]) -> str:
    return " ".join(f"{key}={value:.3f}" for key, value in items)


def train(config: TrainConfig) -> None:
    seed_everything(config.seed)
    device = resolve_device(config.device)

    env = Game2048Env()
    env.seed(config.seed)
    action_dim = env.action_space_n()

    q_network = QNetwork(
        action_dim,
        max_exponent=config.max_exponent,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    target_network = QNetwork(
        action_dim,
        max_exponent=config.max_exponent,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_capacity)

    state, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episodes_completed = 0
    losses: list[float] = []
    scores: list[float] = []

    print(
        "starting training",
        format_metrics(
            (
                ("steps", float(config.steps)),
                ("batch_size", float(config.batch_size)),
            )
        ),
        f"device={device.type}",
    )

    for step in range(1, config.steps + 1):
        legal_actions = env.legal_actions()
        epsilon = linear_epsilon(
            step,
            start=config.epsilon_start,
            end=config.epsilon_end,
            decay_steps=config.epsilon_decay_steps,
        )
        action = select_action(
            q_network=q_network,
            state=state,
            legal_actions=legal_actions,
            epsilon=epsilon,
            device=device,
            action_dim=action_dim,
        )

        next_state, reward, done, truncated, info = env.step(action)
        transition_done = done or truncated
        if transition_done:
            next_action_mask = np.zeros(action_dim, dtype=np.bool_)
        else:
            next_action_mask = legal_actions_to_mask(action_dim, env.legal_actions())

        replay_buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=transition_done,
            next_action_mask=next_action_mask,
        )

        state = next_state
        episode_reward += reward
        episode_length += 1

        if (
            step >= config.learning_starts
            and len(replay_buffer) >= config.batch_size
            and step % config.train_frequency == 0
        ):
            batch = replay_buffer.sample(config.batch_size, device)
            loss = compute_td_loss(
                batch=batch,
                online_network=q_network,
                target_network=target_network,
                gamma=config.gamma,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), config.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        if step % config.target_update_interval == 0:
            target_network.load_state_dict(q_network.state_dict())

        if transition_done:
            episodes_completed += 1
            scores.append(float(info["score"]))
            print(
                f"episode={episodes_completed} step={step} "
                f"score={info['score']} max_tile={info['max_tile']} "
                f"episode_reward={episode_reward:.3f} episode_length={episode_length}"
            )
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

        if config.log_interval > 0 and step % config.log_interval == 0:
            log_items: list[tuple[str, float]] = [
                ("step", float(step)),
                ("epsilon", epsilon),
                ("buffer", float(len(replay_buffer))),
            ]
            if losses:
                log_items.append(("loss", float(np.mean(losses[-100:]))))
            if scores:
                log_items.append(("mean_score", float(np.mean(scores[-20:]))))
            print("train", format_metrics(log_items))

        if config.eval_interval > 0 and step % config.eval_interval == 0:
            metrics = evaluate_policy(
                q_network=q_network,
                action_dim=action_dim,
                device=device,
                episodes=config.eval_episodes,
                seed=config.seed + step,
            )
            if metrics:
                print("eval", format_metrics(metrics.items()))

        if config.checkpoint_interval > 0 and step % config.checkpoint_interval == 0:
            checkpoint_path = save_checkpoint(
                model_path=Path(config.model_dir),
                step=step,
                episodes_completed=episodes_completed,
                q_network=q_network,
                target_network=target_network,
                optimizer=optimizer,
                config=config,
            )
            print(f"saved checkpoint to {checkpoint_path}")

    final_checkpoint_path = save_checkpoint(
        model_path=Path(config.model_dir),
        step=config.steps,
        episodes_completed=episodes_completed,
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        config=config,
    )
    print(f"training complete saved checkpoint to {final_checkpoint_path}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
