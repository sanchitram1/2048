from __future__ import annotations

import json
from pathlib import Path

from game2048.diagnostics import CheckpointInspection, resolve_multihead_head_mode
from evaluation.find_best import (
    CheckpointResult,
    discover_checkpoints,
    evaluate_checkpoints,
    select_best,
    write_outputs,
)


def _result(path: Path, *, step: int, mean_score: float) -> CheckpointResult:
    return CheckpointResult(
        path=str(path),
        step=step,
        episodes=250,
        eval_base_seed=1000,
        metric="mean_score",
        metric_value=mean_score,
        metrics={
            "mean_score": mean_score,
            "median_score": mean_score,
            "times_reached_512": 0,
        },
        value_network="qcnn",
        model_type="dqn",
        head=None,
    )


def test_discover_checkpoints_recurses_and_sorts_by_step(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    nested = model_dir / "nested"
    nested.mkdir(parents=True)
    ckpt_10 = nested / "checkpoint_10.pt"
    ckpt_2 = model_dir / "checkpoint_2.pt"
    best = model_dir / "checkpoint_best.pt"
    ckpt_10.write_bytes(b"10")
    ckpt_2.write_bytes(b"2")
    best.write_bytes(b"best")

    found = discover_checkpoints([model_dir], patterns=["checkpoint_[0-9]*.pt"])

    assert found == [ckpt_2, ckpt_10]


def test_discover_checkpoints_can_include_best_checkpoints(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    numeric = model_dir / "checkpoint_10.pt"
    best = model_dir / "checkpoint_best.pt"
    numeric.write_bytes(b"10")
    best.write_bytes(b"best")

    found = discover_checkpoints(
        [model_dir],
        patterns=["checkpoint_[0-9]*.pt", "checkpoint_best.pt"],
    )

    assert found == [numeric, best]


def test_select_best_uses_metric_then_tie_breakers(tmp_path: Path) -> None:
    low = _result(tmp_path / "checkpoint_100.pt", step=100, mean_score=100.0)
    high = _result(tmp_path / "checkpoint_200.pt", step=200, mean_score=200.0)

    assert select_best([low, high], metric="mean_score", lower_is_better=False) == high
    assert select_best([low, high], metric="mean_score", lower_is_better=True) == low


def test_write_outputs_records_manifest_and_best_pointer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_200.pt"
    checkpoint.write_bytes(b"model")
    output_dir = tmp_path / "best"
    result = _result(checkpoint, step=200, mean_score=200.0)

    manifest = write_outputs(
        output_dir=output_dir,
        results=[result],
        best=result,
        metric="mean_score",
        lower_is_better=False,
        symlink_name="current_best.pt",
        copy_best=True,
        copy_name="checkpoint_best.pt",
        argv=["find-best", str(checkpoint)],
    )

    manifest_json = json.loads((output_dir / "manifest.json").read_text())
    result_rows = (output_dir / "results.jsonl").read_text().splitlines()

    assert len(result_rows) == 1
    assert manifest_json["best"]["path"] == str(checkpoint)
    assert manifest["best_symlink"] == str((output_dir / "current_best.pt").absolute())
    assert manifest["best_symlink_target"] == str(checkpoint.resolve())
    assert manifest["best_copy"] == str((output_dir / "checkpoint_best.pt").resolve())
    assert (output_dir / "checkpoint_best.pt").read_bytes() == b"model"
    assert (output_dir / "current_best.pt").resolve() == checkpoint.resolve()


def test_resolve_multihead_head_mode_precedence() -> None:
    assert resolve_multihead_head_mode(requested_head="q", preferred_head="policy") == "q"
    assert (
        resolve_multihead_head_mode(requested_head="auto", preferred_head="policy")
        == "policy"
    )
    assert (
        resolve_multihead_head_mode(requested_head="auto", preferred_head="unknown")
        == "both"
    )
    assert resolve_multihead_head_mode(requested_head="auto", preferred_head=None) == "both"


def test_evaluate_checkpoints_expands_multihead_both(
    tmp_path: Path, monkeypatch
) -> None:
    ckpt = tmp_path / "checkpoint_1.pt"
    ckpt.write_bytes(b"model")

    monkeypatch.setattr(
        "evaluation.find_best.inspect_checkpoint_type",
        lambda _path: CheckpointInspection(model_type="multihead", preferred_head="both"),
    )

    class _Eval:
        def __init__(self, head: str) -> None:
            self.episodes = 10
            self.eval_base_seed = 123
            self.metrics = {"mean_score": 1.0, "times_reached_512": 0}
            self.value_network = "qcnn"
            self.model_type = "multihead"
            self.head = head

    def _fake_eval(*, checkpoint_path, episodes, device_name, eval_base_seed, head):
        return _Eval(head)

    monkeypatch.setattr(
        "evaluation.find_best.evaluate_multihead_checkpoint",
        _fake_eval,
    )

    results = evaluate_checkpoints(
        [ckpt],
        episodes=10,
        eval_base_seed=123,
        device="cpu",
        metric="mean_score",
        head="auto",
    )

    assert [r.head for r in results] == ["policy", "q"]
    assert all(r.model_type == "multihead" for r in results)
