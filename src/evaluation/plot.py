from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt


EvalMetric = Literal["mean_score", "times_reached_1024", "times_reached_2048"]
TrainingMetric = Literal["dqn_loss", "planner_loss", "total_loss", "mean_score"]

EVAL_METRICS: tuple[EvalMetric, ...] = (
    "mean_score",
    "times_reached_1024",
    "times_reached_2048",
)
TRAINING_METRICS: tuple[TrainingMetric, ...] = (
    "dqn_loss",
    "planner_loss",
    "total_loss",
    "mean_score",
)

TRAINING_FIELD_BY_METRIC: dict[TrainingMetric, str] = {
    "dqn_loss": "dqn_loss_mean_last_100",
    "planner_loss": "planner_loss_mean_last_100",
    "total_loss": "total_loss_mean_last_100",
    "mean_score": "mean_score_last_20_episodes",
}

TITLE_BY_TRAINING_METRIC: dict[TrainingMetric, str] = {
    "dqn_loss": "DQN loss",
    "planner_loss": "Planner loss",
    "total_loss": "Total loss",
    "mean_score": "Mean score",
}


@dataclass(frozen=True)
class Series:
    label: str
    x: tuple[int, ...]
    y: tuple[float, ...]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected JSON object")
                rows.append(value)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"expected a JSONL text file, but {path} could not be decoded as UTF-8"
        ) from exc
    return rows


def resolve_jsonl_path(path: Path, *, default_name: str) -> Path:
    if path.is_dir():
        path = path / default_name
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    if path.suffix != ".jsonl":
        raise ValueError(
            f"expected a JSONL file, got {path}. "
            "Checkpoint .pt files do not contain a checkpoint sweep; pass a "
            "find-best results.jsonl file instead."
        )
    return path


def _require_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"field {field!r} must be numeric")
    return float(value)


def eval_series(path: Path, *, metric: EvalMetric, label: str | None = None) -> Series:
    points: list[tuple[int, float]] = []
    for row in load_jsonl(path):
        step = row.get("step")
        if step is None:
            continue
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"field 'step' must be an integer in {path}")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"field 'metrics' must be an object in {path}")
        if metric not in metrics:
            available = ", ".join(sorted(str(key) for key in metrics))
            raise ValueError(
                f"metric {metric!r} not available in {path}; choose one of: {available}"
            )
        points.append((step, _require_number(metrics[metric], field=metric)))

    if not points:
        raise ValueError(f"No plottable checkpoint-step rows found in {path}")

    points.sort(key=lambda point: point[0])
    return Series(
        label=label or path.parent.name or path.stem,
        x=tuple(point[0] for point in points),
        y=tuple(point[1] for point in points),
    )


def training_series(
    path: Path, *, metric: TrainingMetric, label: str | None = None
) -> Series:
    field = TRAINING_FIELD_BY_METRIC[metric]
    points: list[tuple[int, float]] = []
    for row in load_jsonl(path):
        step = row.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"field 'step' must be an integer in {path}")
        value = row.get(field)
        if value is None:
            continue
        points.append((step, _require_number(value, field=field)))

    if not points:
        raise ValueError(f"No plottable rows for metric {metric!r} found in {path}")

    points.sort(key=lambda point: point[0])
    return Series(
        label=label or path.parent.name or path.stem,
        x=tuple(point[0] for point in points),
        y=tuple(point[1] for point in points),
    )


def save_filename(*, command: str, metric: str) -> str:
    return f"{command}_{metric}.svg"


def render_series(
    series: list[Series],
    *,
    title: str,
    ylabel: str,
    save_dir: Path | None,
    filename: str,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    try:
        for item in series:
            ax.plot(
                item.x,
                item.y,
                marker="o",
                linewidth=1.6,
                markersize=3,
                label=item.label,
            )
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if len(series) > 1:
            ax.legend()
        fig.tight_layout()

        if save_dir is None:
            plt.show()
            return None

        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / filename
        fig.savefig(output_path)
        return output_path
    finally:
        plt.close(fig)


def plot_model(args: argparse.Namespace) -> Path | None:
    metric = args.metric
    results_jsonl = resolve_jsonl_path(args.results_jsonl, default_name="results.jsonl")
    series = eval_series(results_jsonl, metric=metric, label=args.label)
    return render_series(
        [series],
        title=f"Checkpoint evaluation: {metric}",
        ylabel=metric,
        save_dir=args.save_dir,
        filename=save_filename(command="model", metric=metric),
    )


def plot_comparison(args: argparse.Namespace) -> Path | None:
    metric = args.metric
    left_results_jsonl = resolve_jsonl_path(
        args.left_results_jsonl, default_name="results.jsonl"
    )
    right_results_jsonl = resolve_jsonl_path(
        args.right_results_jsonl, default_name="results.jsonl"
    )
    left = eval_series(left_results_jsonl, metric=metric, label=args.left_label)
    right = eval_series(right_results_jsonl, metric=metric, label=args.right_label)
    return render_series(
        [left, right],
        title=f"Checkpoint comparison: {metric}",
        ylabel=metric,
        save_dir=args.save_dir,
        filename=save_filename(command="comparison", metric=metric),
    )


def plot_training(args: argparse.Namespace) -> Path | None:
    metric = args.metric
    metrics_jsonl = resolve_jsonl_path(args.metrics_jsonl, default_name="metrics.jsonl")
    series = training_series(metrics_jsonl, metric=metric, label=args.label)
    return render_series(
        [series],
        title=f"Training metric: {TITLE_BY_TRAINING_METRIC[metric]}",
        ylabel=TITLE_BY_TRAINING_METRIC[metric],
        save_dir=args.save_dir,
        filename=save_filename(command="training", metric=metric),
    )


def _add_save_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save an SVG. If omitted, display the plot interactively.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 2048 training and checkpoint-evaluation JSONL artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    model = subparsers.add_parser(
        "model", help="Plot one find-best results.jsonl metric over checkpoint steps."
    )
    model.add_argument("results_jsonl", type=Path)
    model.add_argument("--metric", choices=EVAL_METRICS, default="mean_score")
    model.add_argument("--label", default=None)
    _add_save_dir(model)
    model.set_defaults(func=plot_model)

    comparison = subparsers.add_parser(
        "comparison", help="Plot two find-best results.jsonl files on one chart."
    )
    comparison.add_argument("left_results_jsonl", type=Path)
    comparison.add_argument("right_results_jsonl", type=Path)
    comparison.add_argument("--metric", choices=EVAL_METRICS, default="mean_score")
    comparison.add_argument("--left-label", default=None)
    comparison.add_argument("--right-label", default=None)
    _add_save_dir(comparison)
    comparison.set_defaults(func=plot_comparison)

    training = subparsers.add_parser(
        "training", help="Plot one training metrics.jsonl metric over training steps."
    )
    training.add_argument("metrics_jsonl", type=Path)
    training.add_argument("--metric", choices=TRAINING_METRICS, required=True)
    training.add_argument("--label", default=None)
    _add_save_dir(training)
    training.set_defaults(func=plot_training)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path = args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"plot error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if output_path is not None:
        print(f"saved plot to {output_path}")


if __name__ == "__main__":
    main()
