from __future__ import annotations

import argparse
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from training.env import Game2048Env
from training.config import train_config_from_dict
from training.dqn import (
    build_value_network,
    legal_actions_to_mask,
    mask_illegal_actions,
)
from training.eval_report import print_rollout_eval_summary
from training.eval_report import summarize_rollouts
from training.inference import (
    ACTION_NAMES,
    ModelAction,
    choose_greedy_action,
    find_latest_checkpoint,
    load_q_network,
)
from training.planning import NStepMCRunner
from training.train import resolve_device
from training.td_ntuple import (
    NTupleValueFunction,
    choose_td_action,
    find_latest_td_checkpoint,
)


@dataclass(frozen=True)
class RolloutEvaluation:
    checkpoint_path: Path
    model_type: str
    episodes: int
    eval_base_seed: int
    scores: tuple[float, ...]
    max_tiles: tuple[int, ...]
    metrics: dict[str, float | int]
    value_network: str | None = None
    head: str | None = None


@dataclass(frozen=True)
class CheckpointInspection:
    model_type: str
    preferred_head: str | None = None
    available_heads: tuple[str, ...] = ()


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
        choices=("auto", "dqn", "td", "mc", "multihead", "greedy"),
        default="auto",
        help=(
            "Model family for evaluation. 'auto' infers from checkpoint filename. "
            "'mc' is N-step Monte Carlo (no checkpoint). "
            "'greedy' is random action selection (no checkpoint)."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=250,
        help="Number of greedy rollout episodes.",
    )
    parser.add_argument(
        "--eval-base-seed",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "Episode i uses RNG seed (eval_base_seed + i) via env.seed. "
            "Default: checkpoint training seed for DQN/TD, or 7 for MC."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="With --model-type mc, print a per-episode table as each game finishes.",
    )
    parser.add_argument(
        "--early-exit",
        action="store_true",
        help="With --model-type mc, stop when max tile >= 2048 (--stop-at-max-tile overrides the value).",
    )
    parser.add_argument(
        "--stop-at-max-tile",
        type=int,
        default=None,
        metavar="N",
        help=(
            "With --model-type mc, end the game when the largest tile reaches at least N "
            "(e.g. 1024, 2048). No effect on dqn/td."
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu"
    )
    parser.add_argument(
        "--head",
        choices=("policy", "q", "both", "auto"),
        default="auto",
        help=(
            "For multihead checkpoints, evaluate policy, q, or both heads. "
            "Precedence is CLI --head > checkpoint preference > both fallback."
        ),
    )
    parser.add_argument("--model-dir", type=str, default="models")
    return parser.parse_args()


def inspect_checkpoint_type(path: Path) -> CheckpointInspection:
    if path.suffix == ".npz":
        return CheckpointInspection(model_type="td")
    if path.suffix != ".pt":
        raise ValueError(f"Unsupported checkpoint format: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint payload type for {path}")

    if "multihead_state_dict" in payload:
        state = payload.get("multihead_state_dict")
        preferred_head = payload.get("preferred_head")
        if not isinstance(preferred_head, str):
            preferred_head = None
        available_heads: list[str] = []
        if isinstance(state, dict):
            has_q = isinstance(state.get("q_network_state_dict"), dict) or isinstance(
                state.get("q"), dict
            )
            has_policy = isinstance(
                state.get("policy_network_state_dict"), dict
            ) or isinstance(state.get("policy"), dict)
            if has_policy:
                available_heads.append("policy")
            if has_q:
                available_heads.append("q")
        if isinstance(state, dict) and preferred_head is None:
            if has_q and has_policy:
                preferred_head = "both"
            elif has_q:
                preferred_head = "q"
            elif has_policy:
                preferred_head = "policy"
        return CheckpointInspection(
            model_type="multihead",
            preferred_head=preferred_head,
            available_heads=tuple(available_heads),
        )
    if "q_network_state_dict" in payload:
        return CheckpointInspection(model_type="dqn")
    raise ValueError(f"Unknown checkpoint payload format: {path}")


def resolve_multihead_head_mode(
    *,
    requested_head: str,
    preferred_head: str | None,
    available_heads: tuple[str, ...] = (),
) -> str:
    if requested_head != "auto":
        return requested_head
    available = set(available_heads)
    if preferred_head == "both" and {"policy", "q"}.issubset(available):
        return "both"
    if preferred_head in {"policy", "q"} and preferred_head in available:
        return preferred_head
    if {"policy", "q"}.issubset(available):
        return "both"
    if "q" in available:
        return "q"
    if "policy" in available:
        return "policy"
    return "both"


class _FlatMultiheadNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(256, 4)
        self.policy_head = nn.Linear(256, 4)

    def _features(self, boards: torch.Tensor) -> torch.Tensor:
        encoded = boards.long().clamp(min=0, max=15)
        if encoded.dim() == 2:
            encoded = encoded.view(-1, 4, 4)
        one_hot = F.one_hot(encoded, num_classes=16).float()
        x = one_hot.permute(0, 3, 1, 2)
        x = self.conv(x)
        return self.trunk(x.flatten(start_dim=1))

    def q_logits(self, boards: torch.Tensor) -> torch.Tensor:
        return self.q_head(self._features(boards))

    def policy_logits(self, boards: torch.Tensor) -> torch.Tensor:
        return self.policy_head(self._features(boards))


def _resolve_checkpoint(args: argparse.Namespace) -> tuple[str, Path, str | None]:
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model_type = args.model_type
        if model_type == "auto":
            inspection = inspect_checkpoint_type(path)
            model_type = inspection.model_type
            head_mode = resolve_multihead_head_mode(
                requested_head=args.head,
                preferred_head=inspection.preferred_head,
                available_heads=inspection.available_heads,
            )
        else:
            if model_type == "multihead":
                inspection = inspect_checkpoint_type(path)
                head_mode = resolve_multihead_head_mode(
                    requested_head=args.head,
                    preferred_head=inspection.preferred_head,
                    available_heads=inspection.available_heads,
                )
            else:
                head_mode = None
        return model_type, path, head_mode

    if args.model_type == "dqn":
        dqn_path = find_latest_checkpoint(args.model_dir)
        if dqn_path is None:
            raise FileNotFoundError(
                f"No DQN checkpoint found in {args.model_dir}. Run `uv run train` first."
            )
        return "dqn", dqn_path, None

    if args.model_type == "td":
        td_path = find_latest_td_checkpoint(args.model_dir)
        if td_path is None:
            raise FileNotFoundError(
                f"No TD checkpoint found in {args.model_dir}. Run `uv run train-td` first."
            )
        return "td", td_path, None

    if args.model_type == "multihead":
        raise ValueError("--model-type multihead requires --checkpoint")

    dqn_path = find_latest_checkpoint(args.model_dir)
    td_path = find_latest_td_checkpoint(args.model_dir)
    if dqn_path is None and td_path is None:
        raise FileNotFoundError(
            f"No checkpoints found in {args.model_dir}. Run `uv run train` or `uv run train-td` first."
        )
    if dqn_path is None:
        assert td_path is not None
        return "td", td_path, None
    if td_path is None:
        assert dqn_path is not None
        return "dqn", dqn_path, None
    # Prefer the DQN checkpoint when both exist to match the current app default.
    return "dqn", dqn_path, None


def _print_mc_verbose_header() -> None:
    print("\n Per-episode results")
    print(f"  {'Scenario':>10}  {'Max Tile':>10}  {'Score':>14}")
    print(f"  {'─' * 10}  {'─' * 10}  {'─' * 14}")


def _print_mc_verbose_row(scenario: int, max_tile: int, score: int) -> None:
    print(
        f"  {scenario:>10}  {max_tile:>10}  {score:>14,}",
        flush=True,
    )


def evaluate_dqn_checkpoint(
    *,
    checkpoint_path: Path,
    episodes: int,
    device_name: str,
    eval_base_seed: int | None,
) -> RolloutEvaluation:
    q_network, config, device = load_q_network(checkpoint_path, device_name=device_name)
    base = int(config.seed) if eval_base_seed is None else int(eval_base_seed)
    scores: list[float] = []
    max_tiles: list[int] = []

    for i in range(episodes):
        env = Game2048Env()
        env.seed(base + i)
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

    metrics = summarize_rollouts(scores, max_tiles)
    if scores:
        metrics.update(
            {
                "min_score": float(min(scores)),
                "max_score": float(max(scores)),
                "episodes": int(episodes),
            }
        )
    return RolloutEvaluation(
        checkpoint_path=checkpoint_path,
        model_type="dqn",
        episodes=episodes,
        eval_base_seed=base,
        scores=tuple(scores),
        max_tiles=tuple(max_tiles),
        metrics=metrics,
        value_network=config.value_network,
    )


def evaluate_multihead_checkpoint(
    *,
    checkpoint_path: Path,
    episodes: int,
    device_name: str,
    eval_base_seed: int | None,
    head: str,
) -> RolloutEvaluation:
    device = resolve_device(device_name)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected checkpoint payload type for {checkpoint_path}")

    config_raw = payload.get("config")
    config = (
        train_config_from_dict(config_raw) if isinstance(config_raw, dict) else None
    )
    if config is None:
        raise ValueError(f"Checkpoint missing config dict: {checkpoint_path}")

    state = payload.get("multihead_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint missing multihead_state_dict: {checkpoint_path}")

    selected_state: dict | None = None
    flat_state = False
    if any(str(key).startswith("conv.") for key in state.keys()):
        flat_state = True
    else:
        if head == "q":
            selected_state = state.get("q_network_state_dict")
            if not isinstance(selected_state, dict):
                selected_state = state.get("q")
            if not isinstance(selected_state, dict):
                raise ValueError(
                    "multihead_state_dict must include q_network_state_dict (or q) for q-head evaluation"
                )
        elif head == "policy":
            selected_state = state.get("policy_network_state_dict")
            if not isinstance(selected_state, dict):
                selected_state = state.get("policy")
            if not isinstance(selected_state, dict):
                raise ValueError(
                    "multihead_state_dict must include policy_network_state_dict (or policy) for policy-head evaluation"
                )
        else:
            raise ValueError(f"Unsupported multihead eval head: {head}")

    model = None
    if flat_state:
        model = _FlatMultiheadNetwork().to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
    else:
        model = build_value_network(
            config.value_network,
            Game2048Env.action_space_n(),
            max_exponent=config.max_exponent,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
        ).to(device)
        model.load_state_dict(selected_state)
        model.eval()

    base = int(config.seed) if eval_base_seed is None else int(eval_base_seed)
    scores: list[float] = []
    max_tiles: list[int] = []
    for i in range(episodes):
        env = Game2048Env()
        env.seed(base + i)
        state_board, info = env.reset()
        while True:
            legal_actions = env.legal_actions()
            action_mask = torch.as_tensor(
                legal_actions_to_mask(Game2048Env.action_space_n(), legal_actions),
                dtype=torch.bool,
                device=device,
            ).unsqueeze(0)
            state_tensor = torch.as_tensor(
                state_board, dtype=torch.long, device=device
            ).unsqueeze(0)
            with torch.no_grad():
                if flat_state:
                    assert isinstance(model, _FlatMultiheadNetwork)
                    logits = (
                        model.q_logits(state_tensor)
                        if head == "q"
                        else model.policy_logits(state_tensor)
                    )
                elif head == "q":
                    model_action = choose_greedy_action(
                        q_network=model,
                        state=state_board,
                        legal_actions=legal_actions,
                        device=device,
                    )
                    state_board, _reward, done, truncated, info = env.step(
                        model_action.action
                    )
                    if done or truncated:
                        scores.append(float(info["score"]))
                        max_tiles.append(int(info["max_tile"]))
                        break
                    continue
                else:
                    logits = model(state_tensor)
                masked = mask_illegal_actions(logits, action_mask)
                action = int(masked.argmax(dim=1).item())
            model_action = ModelAction(
                action=action,
                move=ACTION_NAMES[action],
                q_values=tuple(float(v) for v in logits.squeeze(0).tolist()),
                legal_actions=tuple(int(a) for a in legal_actions),
            )
            state_board, _reward, done, truncated, info = env.step(model_action.action)
            if done or truncated:
                scores.append(float(info["score"]))
                max_tiles.append(int(info["max_tile"]))
                break

    metrics = summarize_rollouts(scores, max_tiles)
    if scores:
        metrics.update(
            {
                "min_score": float(min(scores)),
                "max_score": float(max(scores)),
                "episodes": int(episodes),
            }
        )
    base_result = RolloutEvaluation(
        checkpoint_path=checkpoint_path,
        model_type="dqn",
        episodes=episodes,
        eval_base_seed=base,
        scores=tuple(scores),
        max_tiles=tuple(max_tiles),
        metrics=metrics,
        value_network=config.value_network,
    )
    return replace(base_result, model_type="multihead", head=head)


def _evaluate_dqn(
    *,
    checkpoint_path: Path,
    episodes: int,
    device_name: str,
    eval_base_seed: int | None,
) -> None:
    result = evaluate_dqn_checkpoint(
        checkpoint_path=checkpoint_path,
        episodes=episodes,
        device_name=device_name,
        eval_base_seed=eval_base_seed,
    )
    print(f"Model type: dqn value_network={result.value_network} ({checkpoint_path})")
    print_rollout_eval_summary(
        episodes=result.episodes,
        scores=list(result.scores),
        max_tiles=list(result.max_tiles),
        eval_base_seed=result.eval_base_seed,
    )


def _evaluate_mc(
    *,
    episodes: int,
    verbose: bool,
    stop_at_max_tile: int | None,
    eval_base_seed: int,
) -> None:
    scores: list[float] = []
    max_tiles: list[int] = []
    if verbose:
        _print_mc_verbose_header()
    for i in range(episodes):
        runner = NStepMCRunner(
            seed=eval_base_seed + i, stop_at_max_tile=stop_at_max_tile
        )
        runner.reset()
        while True:
            out = runner.step()
            if out["event"] == "game_over":
                score_f = float(out["score"])
                mt = int(out["max_tile"])
                scores.append(score_f)
                max_tiles.append(mt)
                if verbose:
                    _print_mc_verbose_row(i + 1, mt, int(round(score_f)))
                break

    line = f"\nModel type: mc (stages={runner.stages}, scenarios={runner.scenarios}"
    if stop_at_max_tile is not None:
        line += f", early exit at max tile >= {stop_at_max_tile}"
    print(f"{line})")
    print_rollout_eval_summary(
        episodes=episodes,
        scores=scores,
        max_tiles=max_tiles,
        eval_base_seed=eval_base_seed,
    )


def _evaluate_greedy(
    *,
    episodes: int,
    eval_base_seed: int,
) -> None:
    """Evaluate greedy random action selection baseline."""
    scores: list[float] = []
    max_tiles: list[int] = []

    for i in range(episodes):
        env = Game2048Env()
        env.seed(eval_base_seed + i)
        state_board, info = env.reset()
        while True:
            legal_actions = env.legal_actions()
            action = int(random.choice(legal_actions))
            state_board, _reward, done, truncated, info = env.step(action)
            if done or truncated:
                scores.append(float(info["score"]))
                max_tiles.append(int(info["max_tile"]))
                break

    print("Model type: greedy (random action selection)")
    print_rollout_eval_summary(
        episodes=episodes,
        scores=scores,
        max_tiles=max_tiles,
        eval_base_seed=eval_base_seed,
    )


def _evaluate_td(
    *, checkpoint_path: Path, episodes: int, eval_base_seed: int | None
) -> None:
    value_function, config, _trained_episodes = NTupleValueFunction.load(
        checkpoint_path
    )
    base = int(config.seed) if eval_base_seed is None else int(eval_base_seed)
    scores: list[float] = []
    max_tiles: list[int] = []

    for i in range(episodes):
        game = Game2048Env().game
        # Follow training/inference behavior: deterministic seed stream per episode.
        random.seed(base + i)
        np.random.seed(base + i)
        rng = random.Random(base + i)
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
    print_rollout_eval_summary(
        episodes=episodes,
        scores=scores,
        max_tiles=max_tiles,
        eval_base_seed=base,
    )


def main() -> None:
    args = _parse_args()
    try:
        if args.episodes <= 0:
            raise ValueError("--episodes must be > 0")
        if args.model_type != "mc" and (
            args.early_exit or args.stop_at_max_tile is not None
        ):
            raise ValueError(
                "--early-exit and --stop-at-max-tile require --model-type mc"
            )
        if args.model_type == "greedy":
            if args.checkpoint:
                raise ValueError("--checkpoint cannot be used with --model-type greedy")
            greedy_seed = args.eval_base_seed if args.eval_base_seed is not None else 42
            _evaluate_greedy(
                episodes=args.episodes,
                eval_base_seed=greedy_seed,
            )
            return
        if args.model_type == "mc":
            if args.checkpoint:
                raise ValueError("--checkpoint cannot be used with --model-type mc")
            stop_at = args.stop_at_max_tile
            if args.early_exit and stop_at is None:
                stop_at = 2048
            mc_seed = args.eval_base_seed if args.eval_base_seed is not None else 7
            _evaluate_mc(
                episodes=args.episodes,
                verbose=args.verbose,
                stop_at_max_tile=stop_at,
                eval_base_seed=mc_seed,
            )
            return
        model_type, checkpoint_path, head_mode = _resolve_checkpoint(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Diagnostics error: {exc}")
        raise SystemExit(1) from exc

    if model_type == "dqn":
        _evaluate_dqn(
            checkpoint_path=checkpoint_path,
            episodes=args.episodes,
            device_name=args.device,
            eval_base_seed=args.eval_base_seed,
        )
        return
    if model_type == "multihead":
        modes = ("policy", "q") if head_mode == "both" else (str(head_mode),)
        for mode in modes:
            result = evaluate_multihead_checkpoint(
                checkpoint_path=checkpoint_path,
                episodes=args.episodes,
                device_name=args.device,
                eval_base_seed=args.eval_base_seed,
                head=mode,
            )
            print(f"Model type: multihead head={mode} ({checkpoint_path})")
            print_rollout_eval_summary(
                episodes=result.episodes,
                scores=list(result.scores),
                max_tiles=list(result.max_tiles),
                eval_base_seed=result.eval_base_seed,
            )
        return

    _evaluate_td(
        checkpoint_path=checkpoint_path,
        episodes=args.episodes,
        eval_base_seed=args.eval_base_seed,
    )
