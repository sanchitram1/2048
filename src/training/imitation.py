from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from game2048.game import GameLogic
from training.config import TrainConfig, train_config_from_dict
from training.dqn import build_value_network, mask_illegal_actions
from training.env import Game2048Env
from training.planning import choose_n_step_mc
from training.train import resolve_device, seed_everything

_LOG = logging.getLogger("game2048.imitation")

ACTION_DIM = Game2048Env.action_space_n()


def load_board_dataset(path: str | Path) -> np.ndarray:
    """Load corpus shaped (N, 4, 4) with integer log2 tile exponents (env encoding)."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Board dataset not found: {resolved}")
    boards = np.load(resolved, allow_pickle=False)
    if boards.ndim != 3 or boards.shape[1:] != (4, 4):
        msg = f"Expected board array shaped (N, 4, 4); got shape {boards.shape}"
        raise ValueError(msg)
    if not np.issubdtype(boards.dtype, np.integer):
        raise ValueError(f"Expected integer dtype for boards; got {boards.dtype}")
    return np.asarray(boards, dtype=np.int64)


def game_from_board(grid: np.ndarray) -> GameLogic:
    """Rebuild rules state so legal moves reflect the frozen board snapshot."""
    logic = GameLogic()
    logical = np.asarray(grid, dtype=np.int16)
    if logical.shape != (logic.grid_size, logic.grid_size):
        raise ValueError(f"Board must be 4x4; got {logical.shape}")
    logic.grid = logical.astype(np.int16, copy=True)
    logic.score = 0
    logic.done = False
    return logic


def filter_usable_boards(boards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep boards with ≥1 slide that changes the grid."""
    usable: list[np.ndarray] = []
    row_idx: list[int] = []
    for i in range(boards.shape[0]):
        game = game_from_board(boards[i])
        if game.available_moves():
            usable.append(boards[i])
            row_idx.append(i)
    if not usable:
        return boards[:0].copy(), np.zeros(0, dtype=np.int64)
    stacked = np.stack(usable, axis=0).astype(np.int64, copy=False)
    indexes = np.asarray(row_idx, dtype=np.int64)
    return stacked, indexes


def label_board_states(
    boards: np.ndarray,
    *,
    stages: int,
    scenarios: int,
    seed: int,
    max_boards: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ``choose_n_step_mc`` teacher on usable boards."""
    usable, src_indexes = filter_usable_boards(boards)
    if max_boards is not None:
        cap = max(0, int(max_boards))
        usable = usable[:cap]
        src_indexes = src_indexes[: usable.shape[0]]

    action_dim = ACTION_DIM
    n = usable.shape[0]
    masks = np.zeros((n, action_dim), dtype=np.bool_)
    teacher_actions = np.zeros(n, dtype=np.int64)
    teacher_q = np.zeros((n, action_dim), dtype=np.float32)

    for row in range(n):
        game = game_from_board(usable[row])
        planned = choose_n_step_mc(
            game, stages=stages, scenarios=scenarios, rng=random.Random(seed + row)
        )
        for action_id in planned.legal_actions:
            masks[row, action_id] = True
        ta = planned.action
        if not masks[row, ta]:
            raise RuntimeError("Teacher chose illegal action")
        teacher_actions[row] = ta
        teacher_q[row] = np.array(planned.q_values, dtype=np.float32)

    return usable, masks, teacher_actions, teacher_q, src_indexes


def save_labels_npz(
    *,
    path: Path,
    boards: np.ndarray,
    action_masks: np.ndarray,
    teacher_actions: np.ndarray,
    teacher_q: np.ndarray,
    source_indexes: np.ndarray | None,
    stages: int,
    scenarios: int,
    seed: int,
    dataset_path: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "boards": boards,
        "action_masks": action_masks,
        "teacher_actions": teacher_actions,
        "teacher_q": teacher_q,
        "source_indexes": source_indexes.astype(np.int64, copy=False)
        if source_indexes is not None
        else np.zeros(0, dtype=np.int64),
        "stages": np.array([stages], dtype=np.int64),
        "scenarios": np.array([scenarios], dtype=np.int64),
        "seed": np.array([seed], dtype=np.int64),
        "dataset_path": np.array(dataset_path, dtype=np.str_),
    }
    np.savez_compressed(path, **arrays)


def load_labels_npz(path: str | Path) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Labels file not found: {resolved}")
    data = np.load(resolved, allow_pickle=False)
    boards = data["boards"].astype(np.int64, copy=False)
    masks = data["action_masks"]
    actions = data["teacher_actions"].astype(np.int64, copy=False)
    tq = data["teacher_q"].astype(np.float32, copy=False)
    if masks.dtype != np.bool_:
        masks = masks.astype(np.bool_, copy=False)
    src_raw = data.get("source_indexes")
    unique_src_index = isinstance(src_raw, np.ndarray) and src_raw.ndim == 1
    idx_list = np.asarray(src_raw, dtype=np.int64) if unique_src_index else None
    if idx_list is not None and idx_list.size == 0:
        idx_list = None
    return {
        "boards": boards,
        "action_masks": masks,
        "teacher_actions": actions,
        "teacher_q": tq,
        "source_indexes": idx_list,
        "stages": int(data["stages"][0]) if "stages" in data else None,
        "scenarios": int(data["scenarios"][0]) if "scenarios" in data else None,
        "seed": int(data["seed"][0]) if "seed" in data else None,
    }


def _teacher_probs_from_q(
    q: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Softmax restricted to legal actions (row vectors)."""
    masked = torch.where(mask, q, torch.full_like(q, -1.0e9))
    probs = torch.softmax(masked, dim=-1) * mask.float()
    return probs


def imitation_loss_batch(
    *,
    logits: torch.Tensor,
    action_masks: torch.Tensor,
    teacher_actions: torch.Tensor,
    teacher_q: torch.Tensor | None,
    soft_target_weight: float,
) -> torch.Tensor:
    if soft_target_weight < 0.0 or soft_target_weight > 1.0:
        raise ValueError("soft_target_weight must be in [0, 1]")

    masked_logits = mask_illegal_actions(logits, action_masks)
    log_probs = torch.log_softmax(masked_logits, dim=-1)

    illegal_target = (~action_masks.gather(1, teacher_actions.unsqueeze(1))).squeeze(1)
    if illegal_target.any():
        raise RuntimeError("Teacher indices must be legal")

    nll_hard = -log_probs.gather(1, teacher_actions.unsqueeze(1)).squeeze(1)

    if soft_target_weight == 0.0 or teacher_q is None:
        return nll_hard.mean()

    tgt = _teacher_probs_from_q(teacher_q, action_masks).clamp(min=1e-8)
    kl = (tgt * (torch.log(tgt) - log_probs)).sum(dim=-1)
    return (
        (1.0 - soft_target_weight) * nll_hard + soft_target_weight * kl
    ).mean()


class BoardLabelDataset(torch.utils.data.Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        boards: np.ndarray,
        action_masks: np.ndarray,
        teacher_actions: np.ndarray,
        teacher_q: np.ndarray | None,
    ) -> None:
        self.boards = boards
        self.action_masks = action_masks
        self.teacher_actions = teacher_actions
        self.teacher_q = teacher_q

    def __len__(self) -> int:
        return int(self.boards.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        bs = torch.as_tensor(self.boards[index], dtype=torch.long)
        mask = torch.as_tensor(self.action_masks[index], dtype=torch.bool)
        act = torch.as_tensor(self.teacher_actions[index], dtype=torch.long)
        if self.teacher_q is None:
            return bs, mask, act
        tq = torch.as_tensor(self.teacher_q[index], dtype=torch.float32)
        return bs, mask, act, tq


def merge_train_config_with_init(
    base: TrainConfig,
    *,
    init_checkpoint_path: Path | None,
    learning_rate_override: float | None,
    value_network_override: str | None,
) -> TrainConfig:
    merged_dict = asdict(train_config_from_dict(asdict(base)))
    if (
        init_checkpoint_path is not None
        and init_checkpoint_path.is_file()
    ):
        payload = torch.load(
            init_checkpoint_path, map_location="cpu", weights_only=False
        )
        ck_cfg = payload.get("config")
        if isinstance(ck_cfg, dict):
            merged_dict.update(asdict(train_config_from_dict(ck_cfg)))
    replacements: dict[str, object] = {
        "seed": base.seed,
        "model_dir": base.model_dir,
        "device": base.device,
    }
    if learning_rate_override is not None:
        replacements["learning_rate"] = learning_rate_override
    if value_network_override is not None:
        replacements["value_network"] = value_network_override

    merged_dict.update(replacements)
    field_names = tuple(TrainConfig.__dataclass_fields__.keys())
    return TrainConfig(**{name: merged_dict[name] for name in field_names})


def train_imitation(
    *,
    boards: np.ndarray,
    action_masks: np.ndarray,
    teacher_actions: np.ndarray,
    teacher_q: np.ndarray,
    train_cfg: TrainConfig,
    init_checkpoint_path: Path | None,
    model_dir: Path,
    epochs: int,
    batch_size: int,
    device: torch.device,
    soft_target_weight: float,
    save_step: int,
) -> Path:
    if boards.shape[0] == 0:
        raise ValueError("No labeled boards to train on")

    tq_np = teacher_q if soft_target_weight > 0.0 else None
    ds = BoardLabelDataset(boards, action_masks, teacher_actions, tq_np)
    sampler = torch.utils.data.RandomSampler(
        ds,
        generator=torch.Generator().manual_seed(train_cfg.seed),
    )
    drop_last = len(ds) >= batch_size and batch_size > 1 and len(ds) % batch_size == 1
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=0,
    )

    action_dim = ACTION_DIM
    q_network = build_value_network(
        train_cfg.value_network,
        action_dim,
        max_exponent=train_cfg.max_exponent,
        embedding_dim=train_cfg.embedding_dim,
        hidden_dim=train_cfg.hidden_dim,
    ).to(device)

    target_network = build_value_network(
        train_cfg.value_network,
        action_dim,
        max_exponent=train_cfg.max_exponent,
        embedding_dim=train_cfg.embedding_dim,
        hidden_dim=train_cfg.hidden_dim,
    ).to(device)

    if init_checkpoint_path is not None and init_checkpoint_path.is_file():
        full_ckpt = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        q_sd = full_ckpt.get("q_network_state_dict")
        if isinstance(q_sd, dict):
            q_network.load_state_dict(q_sd)

    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=train_cfg.learning_rate)
    if init_checkpoint_path is not None and init_checkpoint_path.is_file():
        ck = torch.load(init_checkpoint_path, map_location=device, weights_only=False)
        opt_sd = ck.get("optimizer_state_dict")
        if isinstance(opt_sd, dict):
            try:
                optimizer.load_state_dict(opt_sd)
            except (RuntimeError, ValueError):
                pass

    q_network.train()
    target_network.eval()

    for epoch in range(epochs):
        losses: list[float] = []
        for batch_tuple in loader:
            if len(batch_tuple) == 4:
                states_b, masks_b, ta_b, tq_b = batch_tuple
                tq_blob = tq_b.to(device)
            else:
                states_b, masks_b, ta_b = batch_tuple
                tq_blob = None

            states_b = states_b.to(device)
            masks_b = masks_b.to(device)
            ta_b = ta_b.to(device)

            logits = q_network(states_b)
            loss = imitation_loss_batch(
                logits=logits,
                action_masks=masks_b,
                teacher_actions=ta_b,
                teacher_q=tq_blob,
                soft_target_weight=soft_target_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(q_network.parameters(), train_cfg.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        epoch_mean = float(np.mean(losses)) if losses else 0.0
        _LOG.info(
            "[imitation] epoch=%s/%s mean_batch_loss=%.4f batches=%s",
            epoch + 1,
            epochs,
            epoch_mean,
            len(losses),
        )

    ckpt_dir = Path(model_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"checkpoint_{save_step}.pt"
    torch.save(
        {
            "step": save_step,
            "episodes_completed": 0,
            "config": asdict(train_cfg),
            "q_network_state_dict": q_network.cpu().state_dict(),
            "target_network_state_dict": target_network.cpu().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Label boards with N-step MC teacher (.npz) and/or supervised QCNN imitate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="NX4x4 boards .npy (required unless --train-only)",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("scripts/boards_mcts_labels.npz"),
        metavar="PATH",
        help="Teacher labels artifact (.npz); written when labeling, read when --train-only.",
    )
    p.add_argument(
        "--label-only",
        action="store_true",
        help="Compute labels then exit.",
    )
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Train from labels only (loads --labels-output).",
    )
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--scenarios", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-boards", type=int, default=None)
    p.add_argument("--init-checkpoint", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--learning-rate",
        type=float,
        default=TrainConfig.learning_rate,
    )
    p.add_argument(
        "--soft-target-weight",
        type=float,
        default=0.0,
    )
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--save-step", type=int, default=170_000)
    p.add_argument(
        "--value-network",
        choices=("qcnn", "qnetwork"),
        default="qcnn",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    return p.parse_args()


def configure_logging() -> None:
    if not _LOG.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        _LOG.addHandler(h)
        _LOG.propagate = False
    _LOG.setLevel(logging.INFO)


def resolve_labels_arg(args: argparse.Namespace) -> Path:
    return Path(args.labels)


def main() -> None:
    args = parse_args()
    configure_logging()
    seed_everything(int(args.seed))
    device = resolve_device(args.device)

    labels_path = resolve_labels_arg(args)

    if args.label_only and args.train_only:
        raise SystemExit("Cannot combine --label-only with --train-only")

    usable: np.ndarray
    masks: np.ndarray
    tac: np.ndarray
    tq: np.ndarray

    run_label = not args.train_only or args.label_only

    base_cfg = TrainConfig(
        seed=int(args.seed),
        learning_rate=float(args.learning_rate),
        batch_size=int(args.batch_size),
        value_network=args.value_network,  # type: ignore[arg-type]
        device=args.device,
        model_dir=str(args.model_dir),
    )
    train_cfg = merge_train_config_with_init(
        base_cfg,
        init_checkpoint_path=args.init_checkpoint,
        learning_rate_override=float(args.learning_rate),
        value_network_override=args.value_network,
    )

    if run_label:
        if args.dataset is None:
            raise SystemExit("--dataset required when labeling")
        boards = load_board_dataset(args.dataset)
        usable, masks, tac, tq, src_indexes = label_board_states(
            boards,
            stages=int(args.stages),
            scenarios=int(args.scenarios),
            seed=int(args.seed),
            max_boards=args.max_boards,
        )
        save_labels_npz(
            path=labels_path,
            boards=usable,
            action_masks=masks,
            teacher_actions=tac,
            teacher_q=tq,
            source_indexes=src_indexes,
            stages=int(args.stages),
            scenarios=int(args.scenarios),
            seed=int(args.seed),
            dataset_path=str(args.dataset.resolve()),
        )
        _LOG.info(
            "saved %s usable boards → %s", usable.shape[0], labels_path
        )

    if args.label_only:
        return

    if not run_label:
        payload_l = load_labels_npz(labels_path)
        usable = payload_l["boards"]
        masks = payload_l["action_masks"]
        tac = payload_l["teacher_actions"]
        tq = payload_l["teacher_q"]

    ck_out = train_imitation(
        boards=usable.astype(np.int64, copy=False),
        action_masks=masks.astype(np.bool_, copy=False),
        teacher_actions=tac.astype(np.int64, copy=False),
        teacher_q=tq.astype(np.float32, copy=False),
        train_cfg=train_cfg,
        init_checkpoint_path=args.init_checkpoint,
        model_dir=args.model_dir,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        device=device,
        soft_target_weight=float(args.soft_target_weight),
        save_step=int(args.save_step),
    )
    _LOG.info("saved checkpoint %s", ck_out)


if __name__ == "__main__":
    main()
