from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from game2048.game import GameLogic
from training.config import TrainConfig, train_config_from_dict
from training.dqn import (
    ReplayBatch,
    ReplayBuffer,
    Transition,
    build_value_network,
    legal_actions_to_mask,
    linear_epsilon,
    mask_illegal_actions,
)
from training.env import Game2048Env
from training.eval_report import summarize_rollouts
from training.planning import choose_n_step_mc

_LOG = logging.getLogger("game2048.train")

_METRICS_SCHEMA_VERSION = 1
_MANIFEST_SCHEMA_VERSION = 1


class ModelDirNotEmptyError(ValueError):
    """``model_dir`` already contains files; refuse to overwrite manifest or metrics."""

    pass


class UCB:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)  # n_i: number of pulls per arm
        self.values = np.zeros(n_arms)  # μ̂_i: estimated mean reward
        self.total_counts = 0  # t: total pulls so far

    def select_arm(self, legal_actions):
        # Pull each arm once at the beginning
        for arm in legal_actions:
            if self.counts[arm] == 0:
                return arm

        # Compute UCB values
        ucb_values = np.zeros(self.n_arms)
        for arm in legal_actions:
            bonus = np.sqrt((2 * np.log(self.total_counts)) / self.counts[arm])
            ucb_values[arm] = self.values[arm] + bonus

        # Select the arm with the highest UCB value among legal actions
        legal_ucb = {arm: ucb_values[arm] for arm in legal_actions}
        return max(legal_ucb, key=legal_ucb.get)

    def update(self, chosen_arm, reward):
        self.total_counts += 1
        self.counts[chosen_arm] += 1

        # Incremental mean update
        n = self.counts[chosen_arm]
        old_value = self.values[chosen_arm]
        new_value = old_value + (reward - old_value) / n
        self.values[chosen_arm] = new_value


def get_train_log(*, verbose: bool) -> logging.Logger:
    """Message-only log line to stderr; episode lines use DEBUG (enabled when verbose)."""
    if not _LOG.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        _LOG.addHandler(h)
        _LOG.propagate = False
    _LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    return _LOG


def parse_args() -> tuple[TrainConfig, bool, Path | None, set[str]]:
    """Parse CLI args and return config, verbose flag, checkpoint path, and explicitly supplied args."""
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
        "--exploration",
        choices=("epsilon", "ucb"),
        default=TrainConfig.exploration,
        help="Exploration strategy: epsilon-greedy or UCB.",
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
            "(when possible) TrainConfig overrides before RL. Optimizer state "
            "is not loaded; init starts a fresh optimizer."
        ),
    )
    parser.add_argument(
        "--planner-samples-per-update",
        type=int,
        default=TrainConfig.planner_samples_per_update,
        help="Replay rows per gradient step that run live MC distillation (0 disables).",
    )
    parser.add_argument(
        "--planner-stages",
        type=int,
        default=TrainConfig.planner_stages,
        help="MC depth budget for online planner teacher.",
    )
    parser.add_argument(
        "--planner-scenarios",
        type=int,
        default=TrainConfig.planner_scenarios,
        help="MC rollouts per planner call.",
    )
    parser.add_argument(
        "--planner-temperature",
        type=float,
        default=TrainConfig.planner_temperature,
        help="Temperature for soft-Q distillation softmaxes.",
    )
    parser.add_argument(
        "--planner-loss-weight",
        type=float,
        default=TrainConfig.planner_loss_weight,
        help="Scale for planner auxiliary loss (0 disables MC entirely).",
    )
    parser.add_argument(
        "--planner-q-sentinel-cutoff",
        type=float,
        default=TrainConfig.planner_q_sentinel_cutoff,
        help="Planner Q values at or below this are treated as unusable sentinels.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log each finished episode (score, max tile, length, reward).",
    )

    # Track which arguments were explicitly provided (not defaults)
    explicitly_provided = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            key = arg.split("=")[0].lstrip("--").replace("-", "_")
            explicitly_provided.add(key)

    args = parser.parse_args()
    raw = vars(args)
    verbose = bool(raw.pop("verbose"))
    init_ckpt_raw = raw.pop("init_from_checkpoint")
    init_checkpoint = Path(init_ckpt_raw) if init_ckpt_raw is not None else None
    cfg = TrainConfig(**raw)
    return cfg, verbose, init_checkpoint, explicitly_provided


def merge_config_from_init_checkpoint(
    config: TrainConfig,
    path: Path,
    explicitly_provided: set[str] | None = None,
) -> TrainConfig:
    """Merge checkpoint config with CLI config.

    Checkpoint-owned fields (architecture): value_network, max_exponent, embedding_dim, hidden_dim
    always come from checkpoint for weight compatibility.

    CLI-overridable fields (experiment + run-control): learning_rate, gamma, batch_size,
    target_update_interval, epsilon_start, epsilon_end, epsilon_decay_steps, exploration,
    grad_clip, seed, steps, model_dir, device, replay_capacity, checkpoint_interval,
    eval_interval, learning_starts, train_frequency override checkpoint only when explicitly
    supplied on the command line.

    Unspecified fields stay checkpoint-derived.
    """
    if explicitly_provided is None:
        explicitly_provided = set()

    # Checkpoint-owned (architecture): must preserve for weight compatibility
    checkpoint_owned_fields = {
        "value_network",
        "max_exponent",
        "embedding_dim",
        "hidden_dim",
    }

    # CLI-overridable fields (experiment + run-control)
    cli_overridable_fields = {
        "learning_rate",
        "gamma",
        "batch_size",
        "target_update_interval",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_steps",
        "exploration",
        "grad_clip",
        "seed",
        "steps",
        "model_dir",
        "device",
        "replay_capacity",
        "checkpoint_interval",
        "eval_interval",
        "eval_episodes",
        "learning_starts",
        "train_frequency",
        "planner_samples_per_update",
        "planner_stages",
        "planner_scenarios",
        "planner_temperature",
        "planner_loss_weight",
        "planner_q_sentinel_cutoff",
    }

    # Load checkpoint config
    ck = torch.load(path, map_location="cpu", weights_only=False)
    raw_ck = ck.get("config", {})
    if not isinstance(raw_ck, dict):
        raw_ck = {}

    # Start with checkpoint config normalized through train_config_from_dict
    ck_config = train_config_from_dict(raw_ck)
    merged_dict = asdict(ck_config)

    # Apply checkpoint-owned fields (always from checkpoint)
    for field in checkpoint_owned_fields:
        if field in asdict(ck_config):
            merged_dict[field] = asdict(ck_config)[field]

    # Apply CLI-overridable fields only if explicitly provided
    for field in cli_overridable_fields:
        if field in explicitly_provided:
            merged_dict[field] = getattr(config, field)

    # All other fields stay checkpoint-derived (via merged_dict initialization above)
    return train_config_from_dict(merged_dict)


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
    exploration: str,
    ucb: UCB | None,
    device: torch.device,
    action_dim: int,
) -> int:
    if not legal_actions:
        raise RuntimeError("Cannot select an action when no legal actions exist")

    if exploration == "epsilon":
        if random.random() < epsilon:
            return int(random.choice(legal_actions))
    elif exploration == "ucb":
        return ucb.select_arm(legal_actions)

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


def _td_loss_from_current_q(
    *,
    batch: ReplayBatch,
    current_q_values: torch.Tensor,
    online_network: nn.Module,
    target_network: nn.Module,
    gamma: float,
) -> torch.Tensor:
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


def compute_td_loss(
    *,
    batch: ReplayBatch,
    online_network: nn.Module,
    target_network: nn.Module,
    gamma: float,
) -> torch.Tensor:
    current_q_values = online_network(batch.states)
    return _td_loss_from_current_q(
        batch=batch,
        current_q_values=current_q_values,
        online_network=online_network,
        target_network=target_network,
        gamma=gamma,
    )


def compute_td_loss_and_current_q(
    *,
    batch: ReplayBatch,
    online_network: nn.Module,
    target_network: nn.Module,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TD loss plus online Q(s,·) for the batch (single forward on ``batch.states``)."""
    current_q_values = online_network(batch.states)
    loss = _td_loss_from_current_q(
        batch=batch,
        current_q_values=current_q_values,
        online_network=online_network,
        target_network=target_network,
        gamma=gamma,
    )
    return loss, current_q_values


def planner_distillation_loss(
    *,
    transitions: list[Transition],
    student_q_all: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    train_update_index: int,
    stats: dict[str, float],
) -> torch.Tensor:
    """Mean masked KL over the first ``K`` replay rows (same minibatch as TD).

    Index policy: the first ``min(K, batch_size)`` transitions in the sampled list
    (same order as ``ReplayBatch`` rows).
    """
    k = min(config.planner_samples_per_update, len(transitions))
    row_losses: list[torch.Tensor] = []
    t0 = time.perf_counter()
    for j in range(k):
        tr = transitions[j]
        game = GameLogic(skip_initial_spawn=True)
        game.grid = np.asarray(tr.state, dtype=np.int16).copy()
        stats["planner_rows_attempted"] += 1.0
        if not game.available_moves():
            continue
        mc_rng = random.Random(
            (config.seed + train_update_index * 1_000_003 + j) % (2**31 - 1)
        )
        planned = choose_n_step_mc(
            game,
            stages=config.planner_stages,
            scenarios=config.planner_scenarios,
            rng=mc_rng,
        )
        stats["planner_calls"] += 1.0
        teacher_q = np.array(planned.q_values, dtype=np.float64)
        valid_actions: list[int] = []
        for a in planned.legal_actions:
            tq = float(teacher_q[a])
            if math.isfinite(tq) and tq > config.planner_q_sentinel_cutoff:
                valid_actions.append(int(a))
        if len(valid_actions) < 2:
            continue
        stats["planner_rows_used"] += 1.0
        row_q = student_q_all[j]
        idx_t = torch.tensor(valid_actions, device=device, dtype=torch.long)
        z_s = row_q.index_select(0, idx_t) / float(config.planner_temperature)
        z_t = torch.tensor(
            [teacher_q[a] for a in valid_actions],
            device=device,
            dtype=z_s.dtype,
        ) / float(config.planner_temperature)
        p_t = F.softmax(z_t, dim=0).detach()
        log_p_s = F.log_softmax(z_s, dim=0)
        kl = (p_t * (p_t.clamp(min=1e-12).log() - log_p_s)).sum()
        row_losses.append(kl)
    stats["planner_seconds"] += time.perf_counter() - t0
    if not row_losses:
        return student_q_all.sum() * 0.0
    return torch.stack(row_losses).mean()


def evaluate_policy(
    *,
    q_network: nn.Module,
    action_dim: int,
    device: torch.device,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    """Greedy (argmax-Q) rollouts — independent of training exploration (epsilon vs UCB)."""
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
                    exploration="epsilon",
                    ucb=None,
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


def write_train_manifest(
    model_dir: Path,
    *,
    argv: list[str],
    train_config: TrainConfig,
    device: torch.device,
    explicitly_provided: set[str] | None,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "argv": argv,
        "train_config": asdict(train_config),
        "device": device.type,
        "explicitly_provided_cli_keys": sorted(explicitly_provided)
        if explicitly_provided
        else [],
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def append_train_metrics_jsonl(model_dir: Path, record: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record)
    with open(model_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def _validate_planner_train_config(config: TrainConfig) -> None:
    """Reject invalid planner knobs before training allocates RNG or writes artifacts."""
    if config.planner_samples_per_update < 0:
        msg = (
            "planner_samples_per_update must be >= 0 "
            f"(got {config.planner_samples_per_update})"
        )
        raise ValueError(msg)
    if not math.isfinite(config.planner_loss_weight) or config.planner_loss_weight < 0:
        msg = (
            "planner_loss_weight must be finite and >= 0 "
            f"(got {config.planner_loss_weight})"
        )
        raise ValueError(msg)
    planner_on = (
        config.planner_loss_weight != 0.0 and config.planner_samples_per_update > 0
    )
    if not planner_on:
        return
    if not math.isfinite(config.planner_temperature) or config.planner_temperature <= 0:
        msg = (
            "planner_temperature must be finite and > 0 when planner auxiliary loss is enabled "
            f"(got {config.planner_temperature})"
        )
        raise ValueError(msg)
    if config.planner_stages <= 0:
        msg = (
            "planner_stages must be positive when planner auxiliary loss is enabled "
            f"(got {config.planner_stages})"
        )
        raise ValueError(msg)
    if config.planner_scenarios <= 0:
        msg = (
            "planner_scenarios must be positive when planner auxiliary loss is enabled "
            f"(got {config.planner_scenarios})"
        )
        raise ValueError(msg)


def train(
    config: TrainConfig,
    *,
    log: logging.Logger | None = None,
    init_checkpoint: Path | None = None,
    explicitly_provided: set[str] | None = None,
    argv: list[str] | None = None,
) -> None:
    log = log or get_train_log(verbose=False)
    argv = sys.argv if argv is None else argv

    checkpoint_for_weights: Path | None = None
    if init_checkpoint is not None:
        ck_path = Path(init_checkpoint).expanduser().resolve()
        if not ck_path.is_file():
            raise FileNotFoundError(f"--init-from-checkpoint not found: {ck_path}")
        config = merge_config_from_init_checkpoint(
            config, ck_path, explicitly_provided=explicitly_provided
        )
        checkpoint_for_weights = ck_path

    _validate_planner_train_config(config)

    seed_everything(config.seed)
    device = resolve_device(config.device)

    env = Game2048Env()
    env.seed(config.seed)
    action_dim = env.action_space_n()

    if config.exploration == "ucb":
        ucb = UCB(action_dim)
    else:
        ucb = None

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

    replay_buffer = ReplayBuffer(config.replay_capacity)

    model_path = Path(config.model_dir)
    if model_path.is_dir() and any(model_path.iterdir()):
        raise ModelDirNotEmptyError(
            f"model_dir {model_path} is not empty; refusing to overwrite "
            "manifest.json, metrics.jsonl, or checkpoints. "
            "Use an empty directory or remove existing files."
        )
    write_train_manifest(
        model_path,
        argv=list(argv),
        train_config=config,
        device=device,
        explicitly_provided=explicitly_provided,
    )
    if config.log_interval > 0:
        metrics_path = model_path / "metrics.jsonl"
        metrics_path.unlink(missing_ok=True)

    state, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    episodes_completed = 0
    losses: list[float] = []
    dqn_losses: list[float] = []
    planner_losses: list[float] = []
    scores: list[float] = []
    train_update_idx = 0
    planner_rows_attempted_log = 0.0
    planner_rows_used_log = 0.0
    planner_calls_log = 0.0
    planner_seconds_log = 0.0
    planner_enabled = (
        config.planner_loss_weight != 0.0 and config.planner_samples_per_update > 0
    )

    log.info(
        "starting training "
        + format_metrics(
            (
                ("steps", float(config.steps)),
                ("batch_size", float(config.batch_size)),
            )
        )
        + f" device={device.type} value_network={config.value_network} exploration={config.exploration}"
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
            exploration=config.exploration,
            ucb=ucb,
            device=device,
            action_dim=action_dim,
        )

        next_state, reward, done, truncated, info = env.step(action)
        transition_done = done or truncated
        if transition_done:
            next_action_mask = np.zeros(action_dim, dtype=bool)
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

        if ucb is not None:
            ucb.update(action, reward)

        state = next_state
        episode_reward += reward
        episode_length += 1

        if (
            step >= config.learning_starts
            and len(replay_buffer) >= config.batch_size
            and step % config.train_frequency == 0
        ):
            transitions, batch = replay_buffer.sample_transitions_and_batch(
                config.batch_size, device
            )
            if planner_enabled:
                dqn_loss, cur_q = compute_td_loss_and_current_q(
                    batch=batch,
                    online_network=q_network,
                    target_network=target_network,
                    gamma=config.gamma,
                )
                mc_stats = {
                    "planner_rows_attempted": 0.0,
                    "planner_rows_used": 0.0,
                    "planner_calls": 0.0,
                    "planner_seconds": 0.0,
                }
                planner_loss = planner_distillation_loss(
                    transitions=transitions,
                    student_q_all=cur_q,
                    config=config,
                    device=device,
                    train_update_index=train_update_idx,
                    stats=mc_stats,
                )
                total_loss = dqn_loss + config.planner_loss_weight * planner_loss
                planner_rows_attempted_log += mc_stats["planner_rows_attempted"]
                planner_rows_used_log += mc_stats["planner_rows_used"]
                planner_calls_log += mc_stats["planner_calls"]
                planner_seconds_log += mc_stats["planner_seconds"]
                dqn_f = float(dqn_loss.item())
                plan_f = float(planner_loss.item())
            else:
                total_loss = compute_td_loss(
                    batch=batch,
                    online_network=q_network,
                    target_network=target_network,
                    gamma=config.gamma,
                )
                dqn_f = float(total_loss.item())
                plan_f = 0.0

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), config.grad_clip)
            optimizer.step()
            train_update_idx += 1
            losses.append(float(total_loss.item()))
            dqn_losses.append(dqn_f)
            planner_losses.append(plan_f)

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

        eval_metrics_this_step: dict[str, float] | None = None
        if config.eval_interval > 0 and step % config.eval_interval == 0:
            eval_metrics_this_step = evaluate_policy(
                q_network=q_network,
                action_dim=action_dim,
                device=device,
                episodes=config.eval_episodes,
                seed=config.seed + step,
            )
            if eval_metrics_this_step:
                log.info(f"[eval] {format_metrics(eval_metrics_this_step.items())}")

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
            metrics_record: dict[str, Any] = {
                "schema_version": _METRICS_SCHEMA_VERSION,
                "step": step,
                "epsilon": epsilon,
                "replay_buffer_size": len(replay_buffer),
                "dqn_loss_mean_last_100": float(np.mean(dqn_losses[-100:]))
                if dqn_losses
                else None,
                "planner_loss_mean_last_100": float(np.mean(planner_losses[-100:]))
                if planner_losses
                else None,
                "total_loss_mean_last_100": float(np.mean(losses[-100:]))
                if losses
                else None,
                "train_loss_mean_last_100": float(np.mean(losses[-100:]))
                if losses
                else None,
                "mean_score_last_20_episodes": float(np.mean(scores[-20:]))
                if scores
                else None,
                "episodes_completed": episodes_completed,
                "eval": eval_metrics_this_step if eval_metrics_this_step else None,
                "planner_rows_attempted_since_last_log": planner_rows_attempted_log,
                "planner_rows_used_since_last_log": planner_rows_used_log,
                "planner_calls_since_last_log": planner_calls_log,
                "planner_seconds_since_last_log": planner_seconds_log,
            }
            append_train_metrics_jsonl(model_path, metrics_record)
            planner_rows_attempted_log = 0.0
            planner_rows_used_log = 0.0
            planner_calls_log = 0.0
            planner_seconds_log = 0.0

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
            log.info(f"[checkpoint] saved checkpoint to {checkpoint_path}")

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
    config, verbose, init_ckpt, explicitly_provided = parse_args()
    try:
        train(
            config,
            log=get_train_log(verbose=verbose),
            init_checkpoint=init_ckpt,
            explicitly_provided=explicitly_provided,
            argv=sys.argv,
        )
    except ModelDirNotEmptyError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
