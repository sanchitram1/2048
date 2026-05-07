from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Iterable

from game2048.diagnostics import (
    evaluate_dqn_checkpoint,
    evaluate_multihead_checkpoint,
    inspect_checkpoint_type,
    resolve_multihead_head_mode,
)


STEP_CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.pt$")
DEFAULT_METRIC = "mean_score"
DEFAULT_PATTERNS = ("checkpoint_[0-9]*.pt", "checkpoint_best.pt")
DEFAULT_OUTPUT_DIR = Path("models/best")


@dataclass(frozen=True)
class CheckpointResult:
    path: str
    step: int | None
    episodes: int
    eval_base_seed: int
    metric: str
    metric_value: float
    metrics: dict[str, float | int]
    value_network: str | None
    model_type: str
    head: str | None


def checkpoint_step(path: Path) -> int | None:
    match = STEP_CHECKPOINT_RE.fullmatch(path.name)
    if not match:
        return None
    return int(match.group(1))


def discover_checkpoints(
    paths: Iterable[Path], *, patterns: Iterable[str]
) -> list[Path]:
    seen: set[Path] = set()
    checkpoints: list[Path] = []
    for path in paths:
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(
                candidate
                for pattern in patterns
                for candidate in path.rglob(pattern)
                if candidate.is_file()
            )
        else:
            raise FileNotFoundError(f"checkpoint path does not exist: {path}")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                checkpoints.append(candidate)

    return sorted(
        checkpoints,
        key=lambda p: (
            checkpoint_step(p) is None,
            checkpoint_step(p) if checkpoint_step(p) is not None else 0,
            str(p),
        ),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    relative_target = os.path.relpath(
        target_path.resolve(),
        start=link_path.parent.resolve(),
    )
    link_path.symlink_to(relative_target)


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def select_best(
    results: list[CheckpointResult],
    *,
    metric: str,
    lower_is_better: bool,
) -> CheckpointResult:
    direction = -1.0 if lower_is_better else 1.0

    def key(result: CheckpointResult) -> tuple[float, float, float, int, str]:
        return (
            direction * float(result.metric_value),
            float(result.metrics.get("mean_score", 0.0)),
            float(result.metrics.get("times_reached_512", 0.0)),
            int(result.step if result.step is not None else -1),
            result.path,
        )

    try:
        return max(results, key=key)
    except ValueError as exc:
        raise ValueError("no checkpoint results to select from") from exc


def evaluate_checkpoints(
    checkpoints: list[Path],
    *,
    episodes: int,
    eval_base_seed: int | None,
    device: str,
    metric: str,
    head: str,
) -> list[CheckpointResult]:
    results: list[CheckpointResult] = []
    for idx, checkpoint in enumerate(checkpoints, start=1):
        print(
            f"[find-best] evaluating {idx}/{len(checkpoints)} {checkpoint}", flush=True
        )
        inspection = inspect_checkpoint_type(checkpoint)
        if inspection.model_type == "multihead":
            mode = resolve_multihead_head_mode(
                requested_head=head,
                preferred_head=inspection.preferred_head,
                available_heads=inspection.available_heads,
            )
            heads = ("policy", "q") if mode == "both" else (mode,)
            evals = [
                evaluate_multihead_checkpoint(
                    checkpoint_path=checkpoint,
                    episodes=episodes,
                    device_name=device,
                    eval_base_seed=eval_base_seed,
                    head=mode_name,
                )
                for mode_name in heads
            ]
        else:
            evals = [
                evaluate_dqn_checkpoint(
                    checkpoint_path=checkpoint,
                    episodes=episodes,
                    device_name=device,
                    eval_base_seed=eval_base_seed,
                )
            ]

        for evaluation in evals:
            if metric not in evaluation.metrics:
                available = ", ".join(sorted(evaluation.metrics))
                raise ValueError(
                    f"metric {metric!r} not available; choose one of: {available}"
                )
            metric_value = float(evaluation.metrics[metric])
            results.append(
                CheckpointResult(
                    path=str(checkpoint.resolve()),
                    step=checkpoint_step(checkpoint),
                    episodes=evaluation.episodes,
                    eval_base_seed=evaluation.eval_base_seed,
                    metric=metric,
                    metric_value=metric_value,
                    metrics=evaluation.metrics,
                    value_network=evaluation.value_network,
                    model_type=evaluation.model_type,
                    head=evaluation.head,
                )
            )
    return results


def write_outputs(
    *,
    output_dir: Path,
    results: list[CheckpointResult],
    best: CheckpointResult,
    metric: str,
    lower_is_better: bool,
    symlink_name: str,
    copy_best: bool,
    copy_name: str,
    argv: list[str],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = Path(best.path)
    best_sha256 = sha256_file(best_path)

    results_path = output_dir / "results.jsonl"
    with results_path.open("w") as f:
        for result in results:
            f.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    symlink_path = output_dir / symlink_name
    replace_symlink(symlink_path, best_path)

    copied_path: Path | None = None
    if copy_best:
        copied_path = output_dir / copy_name
        shutil.copy2(best_path, copied_path)

    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(argv),
        "episodes": best.episodes,
        "eval_base_seed": best.eval_base_seed,
        "metric": metric,
        "lower_is_better": lower_is_better,
        "best": asdict(best),
        "best_checkpoint_sha256": best_sha256,
        "best_symlink": str(symlink_path.absolute()),
        "best_symlink_target": str(best_path.resolve()),
        "results_jsonl": str(results_path.resolve()),
        "num_checkpoints": len(results),
    }
    if copied_path is not None:
        manifest["best_copy"] = str(copied_path.resolve())
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep DQN checkpoints with the same greedy rollout evaluator used by "
            "`uv run diagnose`, then write a best-checkpoint manifest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Checkpoint .pt files or directories to scan recursively.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help=(
            "Glob pattern used when scanning directories. Repeat to scan multiple "
            "patterns. Defaults include numbered checkpoints and checkpoint_best.pt."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=250,
        help="Greedy rollout episodes per checkpoint.",
    )
    parser.add_argument(
        "--eval-base-seed",
        type=int,
        default=1000,
        help="Episode i uses RNG seed eval_base_seed + i.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="cpu",
        help="Device used for DQN inference.",
    )
    parser.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="Metric to maximize unless --lower-is-better is set.",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Select the lowest metric value instead of the highest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for manifest.json, results.jsonl, and best pointer.",
    )
    parser.add_argument(
        "--symlink-name",
        default="current_best.pt",
        help="Symlink name to update inside --output-dir.",
    )
    parser.add_argument(
        "--copy-best",
        action="store_true",
        help="Also copy the selected checkpoint into --output-dir.",
    )
    parser.add_argument(
        "--copy-name",
        default="checkpoint_best.pt",
        help="Copied checkpoint name when --copy-best is set.",
    )
    parser.add_argument(
        "--head",
        choices=("policy", "q", "both", "auto"),
        default="auto",
        help=(
            "For multihead checkpoints, select head evaluation mode. "
            "Precedence is CLI --head > checkpoint preference > both fallback."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.episodes <= 0:
            raise ValueError("--episodes must be > 0")
        patterns = args.pattern if args.pattern is not None else list(DEFAULT_PATTERNS)
        checkpoints = discover_checkpoints(args.paths, patterns=patterns)
        if not checkpoints:
            raise FileNotFoundError(
                f"no checkpoints matched patterns {patterns!r} under: "
                + ", ".join(str(path) for path in args.paths)
            )
        results = evaluate_checkpoints(
            checkpoints,
            episodes=args.episodes,
            eval_base_seed=args.eval_base_seed,
            device=args.device,
            metric=args.metric,
            head=args.head,
        )
        best = select_best(
            results,
            metric=args.metric,
            lower_is_better=args.lower_is_better,
        )
        manifest = write_outputs(
            output_dir=args.output_dir,
            results=results,
            best=best,
            metric=args.metric,
            lower_is_better=args.lower_is_better,
            symlink_name=args.symlink_name,
            copy_best=args.copy_best,
            copy_name=args.copy_name,
            argv=sys.argv if argv is None else ["find-best", *argv],
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"find-best error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        f"[find-best] best {args.metric}={best.metric_value:.4f} checkpoint={best.path}"
    )
    print(f"[find-best] manifest={args.output_dir / 'manifest.json'}")
    print(f"[find-best] current_best={manifest['best_symlink']}")


if __name__ == "__main__":
    main()
