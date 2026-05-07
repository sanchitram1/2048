from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from training import expert
from training.imitation import load_labels_for_training


def _patch_fast_expert(monkeypatch) -> None:
    monkeypatch.setattr(
        expert,
        "load_q_network",
        lambda checkpoint, device_name: (object(), object(), "cpu"),
    )
    monkeypatch.setattr(
        expert,
        "choose_greedy_action",
        lambda *, q_network, state, legal_actions, device: SimpleNamespace(
            action=int(legal_actions[0])
        ),
    )

    def fake_label_board_states(
        boards,
        *,
        stages,
        scenarios,
        seed,
        max_boards,
        log_every=250,
        usable_rng_offset=0,
        usable_prefiltered=None,
    ):
        usable, src = usable_prefiltered
        n = usable.shape[0]
        masks = np.ones((n, 4), dtype=np.bool_)
        actions = np.zeros(n, dtype=np.int64)
        teacher_q = np.tile(
            np.array([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32),
            (n, 1),
        )
        return usable, masks, actions, teacher_q, src

    monkeypatch.setattr(expert, "label_board_states", fake_label_board_states)


def test_expert_iteration_writes_schema_v2_shards(tmp_path, monkeypatch) -> None:
    _patch_fast_expert(monkeypatch)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake")

    manifest = expert.run_expert_iteration(
        checkpoint=checkpoint,
        output_dir=tmp_path / "expert",
        device="cpu",
        stages=2,
        scenarios=10,
        seed=7,
        epsilon=0.0,
        snapshot_every_rows=2,
        snapshot_every_minutes=10,
        max_episodes=None,
        max_rows=3,
        resume=False,
        force=False,
        dedupe=False,
        log_every=0,
    )

    assert manifest.global_usable_labeled == 3
    assert len(manifest.shard_files) == 2

    payload = load_labels_for_training(tmp_path / "expert")
    assert payload["schema_version"] == 2
    assert payload["boards"].shape == (3, 4, 4)
    assert payload["teacher_policy"].shape == (3, 4)
    np.testing.assert_array_equal(payload["source_indexes"], np.array([0, 1, 2]))
    np.testing.assert_array_equal(payload["episode_id"], np.array([0, 0, 0]))
    assert payload["policy_checkpoint"] == str(checkpoint.resolve())


def test_expert_iteration_resume_appends_shards(tmp_path, monkeypatch) -> None:
    _patch_fast_expert(monkeypatch)
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fake")
    output_dir = tmp_path / "expert"

    expert.run_expert_iteration(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device="cpu",
        stages=2,
        scenarios=10,
        seed=7,
        epsilon=0.0,
        snapshot_every_rows=2,
        snapshot_every_minutes=10,
        max_episodes=None,
        max_rows=2,
        resume=False,
        force=False,
        dedupe=False,
        log_every=0,
    )
    manifest = expert.run_expert_iteration(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device="cpu",
        stages=2,
        scenarios=10,
        seed=7,
        epsilon=0.0,
        snapshot_every_rows=2,
        snapshot_every_minutes=10,
        max_episodes=None,
        max_rows=4,
        resume=True,
        force=False,
        dedupe=False,
        log_every=0,
    )

    payload = load_labels_for_training(output_dir)
    assert manifest.global_usable_labeled == 4
    assert len(manifest.shard_files) == 2
    np.testing.assert_array_equal(payload["source_indexes"], np.array([0, 1, 2, 3]))
