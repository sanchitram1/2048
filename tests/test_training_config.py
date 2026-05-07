from __future__ import annotations

from training.config import train_config_from_dict


def test_train_config_from_dict_ignores_unknown_keys() -> None:
    config = train_config_from_dict(
        {
            "steps": 123,
            "labels": "experiments/labels.npz",
            "non_config_key": True,
        }
    )

    assert config.steps == 123
