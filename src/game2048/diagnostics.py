from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np

from training.env import Game2048Env
from training.eval_report import print_rollout_eval_summary
from training.inference import (
    choose_greedy_action,
    find_latest_checkpoint,
    load_q_network,
)
from training.planning import NStepMCRunner
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
        choices=("auto", "dqn", "td", "mc"),
        default="auto",
        help=(
            "Model family for evaluation. 'auto' infers from checkpoint filename. "
            "'mc' is N-step Monte Carlo (no checkpoint)."
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


def _print_mc_verbose_header() -> None:
    print("\n Per-episode results")
    print(f"  {'Scenario':>10}  {'Max Tile':>10}  {'Score':>14}")
    print(f"  {'─' * 10}  {'─' * 10}  {'─' * 14}")


def _print_mc_verbose_row(scenario: int, max_tile: int, score: int) -> None:
    print(
        f"  {scenario:>10}  {max_tile:>10}  {score:>14,}",
        flush=True,
    )


def _evaluate_dqn(
    *,
    checkpoint_path: Path,
    episodes: int,
    device_name: str,
    eval_base_seed: int | None,
) -> None:
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

    print(
        f"Model type: dqn value_network={config.value_network} ({checkpoint_path})"
    )
    print_rollout_eval_summary(
        episodes=episodes,
        scores=scores,
        max_tiles=max_tiles,
        eval_base_seed=base,
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

    line = (
        f"\nModel type: mc (stages={runner.stages}, scenarios={runner.scenarios}"
    )
    if stop_at_max_tile is not None:
        line += f", early exit at max tile >= {stop_at_max_tile}"
    print(f"{line})")
    print_rollout_eval_summary(
        episodes=episodes,
        scores=scores,
        max_tiles=max_tiles,
        eval_base_seed=eval_base_seed,
    )


def _evaluate_td(*, checkpoint_path: Path, episodes: int, eval_base_seed: int | None) -> None:
    value_function, config, _trained_episodes = NTupleValueFunction.load(checkpoint_path)
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
        model_type, checkpoint_path = _resolve_checkpoint(args)
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

    _evaluate_td(
        checkpoint_path=checkpoint_path,
        episodes=args.episodes,
        eval_base_seed=args.eval_base_seed,
    )
