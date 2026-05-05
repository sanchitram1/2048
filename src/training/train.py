from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.config import TrainConfig, train_config_from_dict
from training.dqn import (
    ReplayBatch,
    ReplayBuffer,
    build_value_network,
    legal_actions_to_mask,
    linear_epsilon,
    mask_illegal_actions,
)
from training.env import Game2048Env
from training.eval_report import summarize_rollouts

_LOG = logging.getLogger("game2048.train")


def get_train_log(*, verbose: bool) -> logging.Logger:
    """Message-only log line to stderr; episode lines use DEBUG (enabled when verbose)."""
    if not _LOG.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        _LOG.addHandler(h)
        _LOG.propagate = False
    _LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    return _LOG


def parse_args() -> tuple[TrainConfig, bool, Path | None]:
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
        "--value-network",
        choices=("qnetwork", "qcnn"),
        default=TrainConfig.value_network,
        dest="value_network",
        help="Q-function architecture: MLP on tile embeddings, or conv + MLP head.",
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
    parser.add_argument(
        "--init-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to imitation or DQN .pt checkpoint to load weights and "
            "(when possible) TrainConfig overrides before RL."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log each finished episode (score, max tile, length, reward).",
    )
    args = parser.parse_args()
    raw = vars(args)
    verbose = bool(raw.pop("verbose"))
    init_ckpt_raw = raw.pop("init_from_checkpoint")
    init_checkpoint = (
        Path(init_ckpt_raw) if init_ckpt_raw is not None else None
    )
    cfg = TrainConfig(**raw)
    return cfg, verbose, init_checkpoint


def merge_config_from_init_checkpoint(config: TrainConfig, path: Path) -> TrainConfig:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    merged_dict = asdict(train_config_from_dict(asdict(config)))
    raw_ck = ck.get("config")
    if isinstance(raw_ck, dict):
        merged_dict.update(asdict(train_config_from_dict(raw_ck)))
    replacements: dict[str, object] = {
        "seed": config.seed,
        "steps": config.steps,
        "model_dir": config.model_dir,
        "device": config.device,
        "replay_capacity": config.replay_capacity,
        "checkpoint_interval": config.checkpoint_interval,
        "eval_interval": config.eval_interval,
        "learning_starts": config.learning_starts,
        "train_frequency": config.train_frequency,
    }
    merged_dict.update(replacements)
    field_names = tuple(TrainConfig.__dataclass_fields__.keys())
    return TrainConfig(**{name: merged_dict[name] for name in field_names})


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
    q_network: nn.Module,
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
    online_network: nn.Module,
    target_network: nn.Module,
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
    q_network: nn.Module,
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
        env = Game2048Env()
        scores: list[float] = []
        max_tiles: list[int] = []
        for ep in range(episodes):
            env.seed(seed + ep)
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
                    max_tiles.append(int(info["max_tile"]))
                    break
        summ = summarize_rollouts(scores, max_tiles)
        return {
            "mean_score": float(summ["mean_score"]),
            "median_score": float(summ["median_score"]),
            "score_variance": float(summ["score_variance"]),
            "mean_max_tile": float(np.mean(max_tiles)),
            "max_score": float(np.max(scores)) if scores else 0.0,
            "reach_256": float(summ["times_reached_256"]),
            "reach_512": float(summ["times_reached_512"]),
            "reach_1024": float(summ["times_reached_1024"]),
            "reach_2048": float(summ["times_reached_2048"]),
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
    q_network: nn.Module,
    target_network: nn.Module,
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


def train(
    config: TrainConfig,
    *,
    log: logging.Logger | None = None,
    init_checkpoint: Path | None = None,
) -> None:
    log = log or get_train_log(verbose=False)
    seed_everything(config.seed)
    device = resolve_device(config.device)

    checkpoint_for_weights: Path | None = None
    if init_checkpoint is not None:
        ck_path = Path(init_checkpoint).expanduser().resolve()
        if not ck_path.is_file():
            raise FileNotFoundError(f"--init-from-checkpoint not found: {ck_path}")
        config = merge_config_from_init_checkpoint(config, ck_path)
        checkpoint_for_weights = ck_path

    env = Game2048Env()
    env.seed(config.seed)
    action_dim = env.action_space_n()

    q_network = build_value_network(
        config.value_network,
        action_dim,
        max_exponent=config.max_exponent,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    target_network = build_value_network(
        config.value_network,
        action_dim,
        max_exponent=config.max_exponent,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    payload: dict[str, object] | None = None
    if checkpoint_for_weights is not None:
        payload = torch.load(
            checkpoint_for_weights, map_location=device, weights_only=False
        )
        q_sd = payload.get("q_network_state_dict")
        if isinstance(q_sd, dict):
            q_network.load_state_dict(q_sd)
            target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
    if checkpoint_for_weights is not None and payload is not None:
        opt_sd = payload.get("optimizer_state_dict")
        if isinstance(opt_sd, dict):
            try:
                optimizer.load_state_dict(opt_sd)
            except (RuntimeError, ValueError):
                pass

    replay_buffer = ReplayBuffer(config.replay_capacity)

    state, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episodes_completed = 0
    losses: list[float] = []
    scores: list[float] = []

    log.info(
        "starting training "
        + format_metrics(
            (
                ("steps", float(config.steps)),
                ("batch_size", float(config.batch_size)),
            )
        )
        + f" device={device.type} value_network={config.value_network}"
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
            log.debug(
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
            log.info(f"[train] {format_metrics(log_items)}")

        if config.eval_interval > 0 and step % config.eval_interval == 0:
            metrics = evaluate_policy(
                q_network=q_network,
                action_dim=action_dim,
                device=device,
                episodes=config.eval_episodes,
                seed=config.seed + step,
            )
            if metrics:
                log.info(f"[eval] {format_metrics(metrics.items())}")

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
            log.info(f"saved checkpoint to {checkpoint_path}")

    final_checkpoint_path = save_checkpoint(
        model_path=Path(config.model_dir),
        step=config.steps,
        episodes_completed=episodes_completed,
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        config=config,
    )
    log.info(f"training complete saved checkpoint to {final_checkpoint_path}")


def main() -> None:
    config, verbose, init_ckpt = parse_args()
    train(
        config,
        log=get_train_log(verbose=verbose),
        init_checkpoint=init_ckpt,
    )


if __name__ == "__main__":
    main()
