from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from training.config import TrainConfig
from training.imitation import (
    boards_face_values_to_log2,
    filter_usable_boards,
    game_from_board,
    imitation_loss_batch,
    label_board_states,
    load_board_dataset,
    load_labels_npz,
    merge_train_config_with_init,
    save_labels_npz,
    train_imitation,
)
from training.train import merge_config_from_init_checkpoint


def _tiny_board_dataset(tmp_path: Path) -> tuple[Path, np.ndarray]:
    b0 = np.array(
        [
            [3, 0, 4, 0],
            [0, 8, 0, 12],
            [2, 0, 11, 0],
            [0, 13, 0, 14],
        ],
        dtype=np.int64,
    )
    path = tmp_path / "tiny.npy"
    np.save(path, b0[np.newaxis, ...])
    return path, b0[np.newaxis, ...]


def test_load_board_dataset_shape(tmp_path: Path) -> None:
    path, _ = _tiny_board_dataset(Path(tmp_path))
    boards = load_board_dataset(path)
    assert boards.shape == (1, 4, 4)


def test_filter_excludes_deadlock_board(tmp_path: Path) -> None:
    deadlock = np.array(
        [
            [15, 2, 15, 2],
            [2, 15, 2, 15],
            [15, 2, 15, 2],
            [2, 15, 2, 15],
        ],
        dtype=np.int64,
    )
    assert not game_from_board(deadlock).available_moves()
    path, playable_board = _tiny_board_dataset(Path(tmp_path))
    boards = np.concatenate([playable_board, deadlock[np.newaxis, ...]])
    playable, ix = filter_usable_boards(boards)
    assert playable.shape[0] == 1
    assert int(ix[0]) == 0


def test_label_board_states_mask_and_teacher_legal(tmp_path: Path) -> None:
    path, _boards = _tiny_board_dataset(Path(tmp_path))
    boards = load_board_dataset(path)
    playable, masks, tac, _tq, _src = label_board_states(
        boards,
        stages=1,
        scenarios=1,
        seed=123,
        max_boards=None,
    )
    n = playable.shape[0]
    assert n == masks.shape[0]
    assert (masks[np.arange(n), tac]).all()


def test_label_artifact_npz_roundtrip(tmp_path: Path) -> None:
    path, _ = _tiny_board_dataset(Path(tmp_path))
    boards = load_board_dataset(path)
    play, masks, tac, tq, src_ix = label_board_states(
        boards, stages=1, scenarios=1, seed=0, max_boards=None
    )
    out_npz = Path(tmp_path) / "lab.npz"
    save_labels_npz(
        path=out_npz,
        boards=play,
        action_masks=masks,
        teacher_actions=tac,
        teacher_q=tq,
        source_indexes=src_ix,
        stages=1,
        scenarios=1,
        seed=0,
        dataset_path=str(path.resolve()),
    )
    loaded = load_labels_npz(out_npz)
    assert np.array_equal(loaded["boards"], play)
    assert np.array_equal(loaded["action_masks"], masks)
    assert np.array_equal(loaded["teacher_actions"], tac)
    np.testing.assert_allclose(loaded["teacher_q"], tq, rtol=0, atol=1e-5)


def test_imitation_loss_soft_and_hard() -> None:
    logits = torch.zeros(2, 4)
    masks = torch.tensor(
        [[True, False, True, False], [False, True, True, False]],
        dtype=torch.bool,
    )
    ta = torch.tensor([2, 1], dtype=torch.long)
    tq = torch.zeros(2, 4, dtype=torch.float32)

    hard = imitation_loss_batch(
        logits=logits,
        action_masks=masks,
        teacher_actions=ta,
        teacher_q=None,
        soft_target_weight=0.0,
    )
    soft = imitation_loss_batch(
        logits=logits,
        action_masks=masks,
        teacher_actions=ta,
        teacher_q=tq,
        soft_target_weight=1.0,
    )
    assert hard.ndim == soft.ndim == 0


def test_merge_configs_from_ckpt_stub(tmp_path: Path) -> None:
    base = TrainConfig(seed=999, steps=111, model_dir=str(Path(tmp_path) / "m"))
    stub = Path(tmp_path) / "stub.pt"
    torch.save(
        {
            "step": 0,
            "config": {"max_exponent": 12, "value_network": "qcnn"},
            "q_network_state_dict": {},
        },
        stub,
    )
    merged_im = merge_train_config_with_init(
        base,
        init_checkpoint_path=stub,
        learning_rate_override=2e-4,
        value_network_override="qcnn",
    )
    assert merged_im.max_exponent == 12
    assert merged_im.seed == base.seed

    merged_tr = merge_config_from_init_checkpoint(base, stub)
    assert merged_tr.steps == base.steps


def test_smoke_train_imitation_writes_loadable_ckpt(tmp_path: Path) -> None:
    ds = Path(tmp_path) / "play.npy"
    boards = np.array(
        [
            [[2, 0, 3, 0], [3, 0, 11, 0], [14, 0, 13, 0], [13, 0, 13, 0]],
        ],
        dtype=np.int64,
    )
    np.save(ds, boards)
    play, masks, tac, tq, _ = label_board_states(
        boards,
        stages=1,
        scenarios=1,
        seed=42,
        max_boards=None,
    )
    assert play.shape[0] >= 1
    outcome = train_imitation(
        boards=play,
        action_masks=masks,
        teacher_actions=tac,
        teacher_q=tq,
        train_cfg=TrainConfig(
            seed=1,
            device="cpu",
            model_dir=str(Path(tmp_path) / "m"),
            value_network="qnetwork",
            batch_size=2,
            learning_rate=1e-3,
        ),
        init_checkpoint_path=None,
        model_dir=Path(tmp_path) / "models_smoke",
        epochs=1,
        batch_size=2,
        device=torch.device("cpu"),
        soft_target_weight=0.0,
        save_step=1,
    )
    payload = torch.load(
        outcome.final_checkpoint, map_location="cpu", weights_only=False
    )
    assert payload["step"] == 1
    assert isinstance(payload["q_network_state_dict"], dict)


def test_boards_face_values_to_log2() -> None:
    face = np.array(
        [
            [
                [2, 0, 512, 8],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ],
        dtype=np.int64,
    )
    log2 = boards_face_values_to_log2(face)
    assert log2[0, 0, 0] == 1
    assert log2[0, 0, 2] == 9
    assert log2[0, 0, 3] == 3


def test_boards_face_values_rejects_odd_tile() -> None:
    face = np.array([[[3, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    with pytest.raises(ValueError, match="powers of 2"):
        boards_face_values_to_log2(face)


def test_train_imitation_run_dir_manifest_best_and_cosine(tmp_path: Path) -> None:
    """metrics.jsonl, checkpoint_best/final, cosine+warmup scheduler smoke."""
    row = np.array(
        [[[2, 0, 3, 0], [3, 0, 11, 0], [14, 0, 13, 0], [13, 0, 13, 0]]],
        dtype=np.int64,
    )
    boards = np.concatenate([row] * 8, axis=0)
    play, masks, tac, tq, _ = label_board_states(
        boards,
        stages=1,
        scenarios=1,
        seed=42,
        max_boards=None,
    )
    run_dir = Path(tmp_path) / "imit_run"
    outcome = train_imitation(
        boards=play[:6],
        action_masks=masks[:6],
        teacher_actions=tac[:6],
        teacher_q=tq[:6],
        train_cfg=TrainConfig(
            seed=1,
            device="cpu",
            model_dir=str(tmp_path / "m"),
            value_network="qnetwork",
            batch_size=2,
            learning_rate=1e-3,
        ),
        init_checkpoint_path=None,
        model_dir=tmp_path / "models_out",
        epochs=2,
        batch_size=2,
        device=torch.device("cpu"),
        soft_target_weight=0.0,
        save_step="test",
        val_boards=play[6:8],
        val_masks=masks[6:8],
        val_teacher_actions=tac[6:8],
        log_agreement_every_epoch=True,
        imitation_run_dir=run_dir,
        lr_schedule="cosine",
        warmup_epochs=1,
    )
    mf = run_dir / "metrics.jsonl"
    assert mf.is_file()
    lines = mf.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert (run_dir / "checkpoint_final.pt").is_file()
    assert outcome.best_checkpoint is not None
    assert outcome.best_checkpoint.is_file()
    assert outcome.final_checkpoint == (run_dir / "checkpoint_final.pt").resolve()


def test_train_imitation_epoch_checkpoints_pruned(tmp_path: Path) -> None:
    row = np.array(
        [[[2, 0, 3, 0], [3, 0, 11, 0], [14, 0, 13, 0], [13, 0, 13, 0]]],
        dtype=np.int64,
    )
    boards = np.concatenate([row] * 8, axis=0)
    play, masks, tac, tq, _ = label_board_states(
        boards,
        stages=1,
        scenarios=1,
        seed=99,
        max_boards=None,
    )
    run_dir = Path(tmp_path) / "imit_ep"
    train_imitation(
        boards=play[:6],
        action_masks=masks[:6],
        teacher_actions=tac[:6],
        teacher_q=tq[:6],
        train_cfg=TrainConfig(
            seed=2,
            device="cpu",
            model_dir=str(tmp_path / "m2"),
            value_network="qnetwork",
            batch_size=2,
            learning_rate=1e-3,
        ),
        init_checkpoint_path=None,
        model_dir=tmp_path / "models_ep",
        epochs=3,
        batch_size=2,
        device=torch.device("cpu"),
        soft_target_weight=0.0,
        save_step="ep",
        val_boards=play[6:8],
        val_masks=masks[6:8],
        val_teacher_actions=tac[6:8],
        log_agreement_every_epoch=True,
        imitation_run_dir=run_dir,
        epoch_checkpoints=True,
        keep_last_k=1,
    )
    eps = sorted(run_dir.glob("checkpoint_epoch*.pt"))
    assert len(eps) == 1
    assert eps[0].name == "checkpoint_epoch0003.pt"
