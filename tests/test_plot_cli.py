from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

from evaluation.plot import (  # noqa: E402
    eval_series,
    plot_comparison,
    plot_model,
    plot_training,
    resolve_jsonl_path,
    save_filename,
    training_series,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_eval_series_reads_mean_score_and_reach_metrics(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_jsonl(
        results,
        [
            {
                "step": 1000,
                "metrics": {
                    "mean_score": 10.5,
                    "times_reached_1024": 2,
                    "times_reached_2048": 0,
                },
            },
            {
                "step": 500,
                "metrics": {
                    "mean_score": 7.0,
                    "times_reached_1024": 1,
                    "times_reached_2048": 0,
                },
            },
        ],
    )

    mean = eval_series(results, metric="mean_score")
    reached = eval_series(results, metric="times_reached_1024")

    assert mean.x == (500, 1000)
    assert mean.y == (7.0, 10.5)
    assert reached.y == (1.0, 2.0)


def test_eval_series_rejects_unknown_metric_in_file(tmp_path: Path) -> None:
    results = tmp_path / "results.jsonl"
    _write_jsonl(results, [{"step": 1, "metrics": {"mean_score": 1.0}}])

    with pytest.raises(ValueError, match="metric 'times_reached_2048' not available"):
        eval_series(results, metric="times_reached_2048")


def test_resolve_jsonl_path_accepts_directory_and_rejects_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = run_dir / "results.jsonl"
    results.write_text("")
    checkpoint = tmp_path / "checkpoint_1.pt"
    checkpoint.write_bytes(b"\x80not-json")

    assert resolve_jsonl_path(run_dir, default_name="results.jsonl") == results

    with pytest.raises(ValueError, match="Checkpoint \\.pt files"):
        resolve_jsonl_path(checkpoint, default_name="results.jsonl")


def test_training_series_maps_names_and_skips_null_values(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(
        metrics,
        [
            {
                "step": 1000,
                "dqn_loss_mean_last_100": None,
                "mean_score_last_20_episodes": 10.0,
            },
            {
                "step": 2000,
                "dqn_loss_mean_last_100": 0.4,
                "mean_score_last_20_episodes": 20.0,
            },
        ],
    )

    loss = training_series(metrics, metric="dqn_loss")
    score = training_series(metrics, metric="mean_score")

    assert loss.x == (2000,)
    assert loss.y == (0.4,)
    assert score.x == (1000, 2000)
    assert score.y == (10.0, 20.0)


def test_save_filename_is_deterministic() -> None:
    assert save_filename(command="model", metric="mean_score") == "model_mean_score.svg"
    assert (
        save_filename(command="training", metric="dqn_loss") == "training_dqn_loss.svg"
    )


class _Args:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def test_plot_commands_save_svg_files(tmp_path: Path) -> None:
    results_a = tmp_path / "a.jsonl"
    results_b = tmp_path / "b.jsonl"
    metrics = tmp_path / "metrics.jsonl"
    save_dir = tmp_path / "plots"
    _write_jsonl(
        results_a,
        [{"step": 1, "metrics": {"mean_score": 1.0, "times_reached_1024": 0}}],
    )
    _write_jsonl(
        results_b,
        [{"step": 1, "metrics": {"mean_score": 2.0, "times_reached_1024": 1}}],
    )
    _write_jsonl(metrics, [{"step": 1, "total_loss_mean_last_100": 0.5}])

    model_path = plot_model(
        _Args(
            results_jsonl=results_a,
            metric="mean_score",
            label=None,
            save_dir=save_dir,
        )
    )
    comparison_path = plot_comparison(
        _Args(
            left_results_jsonl=results_a,
            right_results_jsonl=results_b,
            metric="mean_score",
            left_label="a",
            right_label="b",
            save_dir=save_dir,
        )
    )
    training_path = plot_training(
        _Args(
            metrics_jsonl=metrics,
            metric="total_loss",
            label=None,
            save_dir=save_dir,
        )
    )

    assert model_path == save_dir / "model_mean_score.svg"
    assert comparison_path == save_dir / "comparison_mean_score.svg"
    assert training_path == save_dir / "training_total_loss.svg"
    assert model_path.exists()
    assert comparison_path.exists()
    assert training_path.exists()
