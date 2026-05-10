from __future__ import annotations

import numpy as np
from pathlib import Path

import pytest
import torch

from evaluation.diagnostics import (
    _resolve_checkpoint,
    evaluate_multihead_checkpoint,
    inspect_checkpoint_type,
)
from training.dqn import build_value_network


def test_inspect_checkpoint_type_detects_dqn_and_multihead(tmp_path: Path) -> None:
    dqn_ckpt = tmp_path / "checkpoint_dqn.pt"
    torch.save({"q_network_state_dict": {}}, dqn_ckpt)
    dqn_info = inspect_checkpoint_type(dqn_ckpt)
    assert dqn_info.model_type == "dqn"
    assert dqn_info.preferred_head is None

    multi_ckpt = tmp_path / "checkpoint_multi.pt"
    torch.save(
        {"multihead_state_dict": {}, "preferred_head": "policy"},
        multi_ckpt,
    )
    multi_info = inspect_checkpoint_type(multi_ckpt)
    assert multi_info.model_type == "multihead"
    assert multi_info.preferred_head == "policy"


def test_inspect_checkpoint_type_detects_td_npz(tmp_path: Path) -> None:
    td_ckpt = tmp_path / "checkpoint_td.npz"
    np.savez(td_ckpt, dummy=np.array([1]))
    td_info = inspect_checkpoint_type(td_ckpt)
    assert td_info.model_type == "td"


def test_inspect_checkpoint_type_rejects_unknown_payload(tmp_path: Path) -> None:
    unknown = tmp_path / "checkpoint_unknown.pt"
    torch.save({"not_a_model": 1}, unknown)
    with pytest.raises(ValueError, match="Unknown checkpoint payload format"):
        inspect_checkpoint_type(unknown)


def test_evaluate_multihead_checkpoint_uses_multihead_state_dict(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "multihead.pt"
    q_network = build_value_network(
        "qnetwork",
        4,
        max_exponent=15,
        embedding_dim=32,
        hidden_dim=256,
    )
    torch.save(
        {
            "config": {
                "value_network": "qnetwork",
                "max_exponent": 15,
                "embedding_dim": 32,
                "hidden_dim": 256,
                "seed": 7,
            },
            "multihead_state_dict": {
                "q_network_state_dict": q_network.state_dict(),
            },
        },
        checkpoint,
    )

    result = evaluate_multihead_checkpoint(
        checkpoint_path=checkpoint,
        episodes=1,
        device_name="cpu",
        eval_base_seed=123,
        head="q",
    )

    assert result.model_type == "multihead"
    assert result.head == "q"
    assert result.episodes == 1
    assert "mean_score" in result.metrics


def test_evaluate_multihead_checkpoint_requires_policy_state_for_policy_head(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "multihead_q_only.pt"
    q_network = build_value_network(
        "qnetwork",
        4,
        max_exponent=15,
        embedding_dim=32,
        hidden_dim=256,
    )
    torch.save(
        {
            "config": {
                "value_network": "qnetwork",
                "max_exponent": 15,
                "embedding_dim": 32,
                "hidden_dim": 256,
                "seed": 7,
            },
            "multihead_state_dict": {
                "q_network_state_dict": q_network.state_dict(),
            },
        },
        checkpoint,
    )

    with pytest.raises(
        ValueError,
        match="must include policy_network_state_dict",
    ):
        evaluate_multihead_checkpoint(
            checkpoint_path=checkpoint,
            episodes=1,
            device_name="cpu",
            eval_base_seed=123,
            head="policy",
        )


def test_resolve_checkpoint_multihead_auto_uses_preferred_head(tmp_path: Path) -> None:
    checkpoint = tmp_path / "multihead_preferred.pt"
    torch.save(
        {
            "config": {
                "value_network": "qnetwork",
                "max_exponent": 15,
                "embedding_dim": 32,
                "hidden_dim": 256,
                "seed": 7,
            },
            "preferred_head": "policy",
            "multihead_state_dict": {
                "q_network_state_dict": {},
                "policy_network_state_dict": {},
            },
        },
        checkpoint,
    )
    args = type(
        "Args",
        (),
        {
            "checkpoint": str(checkpoint),
            "model_type": "multihead",
            "head": "auto",
            "model_dir": "models",
        },
    )()

    model_type, resolved_path, head_mode = _resolve_checkpoint(args)

    assert model_type == "multihead"
    assert resolved_path == checkpoint
    assert head_mode == "policy"


def test_inspect_checkpoint_type_multihead_infers_q_when_only_q_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "multihead_q_only.pt"
    torch.save(
        {
            "config": {"seed": 7},
            "multihead_state_dict": {
                "q_network_state_dict": {},
            },
        },
        checkpoint,
    )

    info = inspect_checkpoint_type(checkpoint)

    assert info.model_type == "multihead"
    assert info.preferred_head == "q"
    assert info.available_heads == ("q",)


def test_evaluate_multihead_checkpoint_supports_flat_state_dict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "multihead_flat.pt"
    torch.save(
        {
            "config": {"seed": 7},
            "multihead_state_dict": {
                "conv.0.weight": torch.randn(64, 16, 2, 2),
                "conv.0.bias": torch.randn(64),
                "conv.2.weight": torch.randn(128, 64, 2, 2),
                "conv.2.bias": torch.randn(128),
                "trunk.0.weight": torch.randn(256, 512),
                "trunk.0.bias": torch.randn(256),
                "q_head.weight": torch.randn(4, 256),
                "q_head.bias": torch.randn(4),
                "policy_head.weight": torch.randn(4, 256),
                "policy_head.bias": torch.randn(4),
            },
        },
        checkpoint,
    )

    result = evaluate_multihead_checkpoint(
        checkpoint_path=checkpoint,
        episodes=1,
        device_name="cpu",
        eval_base_seed=123,
        head="q",
    )

    assert result.model_type == "multihead"
    assert result.head == "q"
