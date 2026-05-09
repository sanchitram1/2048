"""CLI for the ``compare`` diagnostic.

Collects (or loads) rollout boards, runs N models + planner on each
board, and writes ``manifest.json`` / ``boards.jsonl`` / ``summary.json``
to the output directory. All metric logic delegates to
:mod:`evaluation.metrics` so alignment is computed identically here and
in any other consumer.
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import numpy as np

from evaluation.rollout import load_rollout_npz, save_rollout_npz
from game2048.compare import collect_rollout_boards, run_compare


def _parse_compare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multiple model outputs on the same board set."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Board source: rollout:<checkpoint.pt> or npz:<path.npz>",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Checkpoint paths to compare",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="For --source rollout:, number of episodes to collect.",
    )
    parser.add_argument(
        "--tail-moves",
        type=int,
        default=10,
        help="Sliding window size: retain last N transitions per episode.",
    )
    parser.add_argument(
        "--stages",
        type=int,
        default=2,
        help="MCTS stages for planner comparison.",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=10,
        help="MCTS scenarios for planner comparison.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="RNG seed for rollout collection and planner.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/compare"),
        help="Output directory for compare artifacts.",
    )
    return parser.parse_args()


def compare_main() -> None:
    """Entry point for the ``compare`` script."""
    args = _parse_compare_args()
    rollout_npz_path: Path | None = None

    try:
        source_spec = args.source
        if ":" not in source_spec:
            raise ValueError(
                f"Invalid source spec: {source_spec}. "
                "Use rollout:<path.pt> or npz:<path.npz>"
            )
        source_kind, source_path_str = source_spec.split(":", 1)
        source_path = Path(source_path_str)

        if source_kind == "rollout":
            if not source_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {source_path}")
            print(
                f"[compare] Collecting rollout from {source_path} "
                f"({args.episodes} episodes, {args.tail_moves} tail moves)..."
            )
            transitions, rollout_cfg = collect_rollout_boards(
                checkpoint_path=source_path,
                episodes=args.episodes,
                tail_moves=args.tail_moves,
                seed=args.seed,
                device_name=args.device,
            )
            rollout_id = str(uuid.uuid4())
            rollout_npz_path = Path("data/rollout") / f"{rollout_id}.npz"
            save_rollout_npz(rollout_npz_path, transitions, rollout_cfg)
            print(
                f"[compare] Saved {len(transitions)} transitions to {rollout_npz_path}"
            )

        elif source_kind == "npz":
            if not source_path.exists():
                raise FileNotFoundError(f"NPZ not found: {source_path}")
            print(f"[compare] Loading boards from {source_path}...")
            transitions, rollout_cfg = load_rollout_npz(source_path)
            print(f"[compare] Loaded {len(transitions)} transitions")

        else:
            raise ValueError(f"Unknown source kind: {source_kind}. Use rollout or npz")

        model_paths = [Path(m) for m in args.models]
        for mp in model_paths:
            if not mp.exists():
                raise FileNotFoundError(f"Model not found: {mp}")

        print(f"[compare] Comparing {len(model_paths)} models...")
        board_outputs, summary = run_compare(
            transitions=transitions,
            model_paths=model_paths,
            stages=args.stages,
            scenarios=args.scenarios,
            seed=args.seed,
            device_name=args.device,
        )

        compare_id = str(uuid.uuid4())
        output_dir = args.output_dir / compare_id
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "run_id": compare_id,
            "created": str(np.datetime64("now")),
            "source": {
                "kind": source_kind,
                "path": str(source_path),
                "rollout_boards_npz": (
                    str(rollout_npz_path) if rollout_npz_path is not None else None
                ),
                "episodes": args.episodes,
                "tail_moves": args.tail_moves,
                "seed": args.seed,
            },
            "planner": {
                "enabled": True,
                "stages": args.stages,
                "scenarios": args.scenarios,
            },
            "models": [
                {"model_id": f"model_{i}", "path": str(mp)}
                for i, mp in enumerate(model_paths)
            ],
            "artifacts": {
                "boards_jsonl": str(output_dir / "boards.jsonl"),
                "summary_json": str(output_dir / "summary.json"),
            },
        }

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        boards_path = output_dir / "boards.jsonl"
        with open(boards_path, "w") as f:
            for board_out in board_outputs:
                f.write(json.dumps(board_out.to_dict()) + "\n")

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[compare] Saved artifacts to {output_dir}")
        print(f"[compare] Boards: {boards_path}")
        print(f"[compare] Summary: {summary_path}")
        print(f"[compare] Manifest: {manifest_path}")

        print(f"\n[compare] Summary: {len(board_outputs)} boards compared")
        for model_id, alignment in summary.get("planner_alignment", {}).items():
            rate = alignment.get("agree_rate", 0.0)
            print(f"  {model_id}: planner agreement {rate:.1%}")

    except (FileNotFoundError, ValueError) as exc:
        print(f"Compare error: {exc}")
        raise SystemExit(1) from exc
