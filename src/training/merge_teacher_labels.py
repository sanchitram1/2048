#!/usr/bin/env pkgx uv run
"""Merge multiple teacher `.npz` artifacts into one file for `uv run imitate --train-only`.

Console entrypoint: ``uv run merge-mcts-datasets …``.

Concatenates rows from ``load_labels_npz`` inputs. By default rewrites ``source_indexes``
to ``0 .. N-1`` so splits stay row-wise i.i.d.; use ``--preserve-sources`` if you need
per-file index namespaces for grouped splits (prefix ``file_idx * 1_000_000_000``).

Mixed ``teacher_q`` (e.g. MIP zeros vs MC dense): training with ``--soft-target-weight > 0``
blends KL using rows where teacher_q is meaningful — combined CE-only is safe.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

from training.imitation import load_labels_for_training, load_labels_npz, save_labels_npz

_LOG = logging.getLogger("game2048.merge_teacher_labels")
SOURCE_INDEX_NAMESPACE = 1_000_000_000


def merge_teacher_npz_files(
    paths: list[Path],
    *,
    preserve_sources: bool,
    output_path: Path,
) -> None:
    if len(paths) < 1:
        raise ValueError("need at least one input .npz")

    boards_parts: list[np.ndarray] = []
    masks_parts: list[np.ndarray] = []
    actions_parts: list[np.ndarray] = []
    tq_parts: list[np.ndarray] = []
    src_parts: list[np.ndarray] = []

    meta_paths: list[str] = []

    for idx, path in enumerate(paths):
        resolved = path.expanduser().resolve()
        payload = load_labels_npz(resolved)
        meta_paths.append(str(resolved))

        b = payload["boards"]  # type: ignore[assignment]
        m = payload["action_masks"]  # type: ignore[assignment]
        a = payload["teacher_actions"]  # type: ignore[assignment]
        tq = payload["teacher_q"]  # type: ignore[assignment]
        si = payload["source_indexes"]

        n = int(b.shape[0])
        if m.shape[0] != n or a.shape[0] != n or tq.shape[0] != n:
            msg = f"Length mismatch in {resolved}"
            raise ValueError(msg)

        boards_parts.append(b.astype(np.int64, copy=False))
        masks_parts.append(m.astype(np.bool_, copy=False))
        actions_parts.append(a.astype(np.int64, copy=False))
        tq_parts.append(tq.astype(np.float32, copy=False))

        if preserve_sources:
            prefix = int(idx) * 1_000_000_000
            if si is None or len(si) != n:
                si_arr = np.arange(n, dtype=np.int64) + prefix
            else:
                si_arr = si.astype(np.int64, copy=False) + prefix
            src_parts.append(si_arr)
        else:
            pass  # filled below

    boards = np.concatenate(boards_parts, axis=0)
    masks = np.concatenate(masks_parts, axis=0)
    actions = np.concatenate(actions_parts, axis=0)
    teacher_q = np.concatenate(tq_parts, axis=0)

    if preserve_sources:
        source_indexes = np.concatenate(src_parts, axis=0)
    else:
        source_indexes = np.arange(boards.shape[0], dtype=np.int64)

    ds_meta = "merged|" + "|".join(meta_paths)

    save_labels_npz(
        path=output_path,
        boards=boards,
        action_masks=masks,
        teacher_actions=actions,
        teacher_q=teacher_q,
        source_indexes=source_indexes,
        stages=0,
        scenarios=0,
        seed=0,
        dataset_path=ds_meta,
    )
    _LOG.info(
        "[merge-labels] wrote %s rows from %s inputs → %s (preserve_sources=%s)",
        boards.shape[0],
        len(paths),
        output_path,
        preserve_sources,
    )


def _parse_blend_inputs(specs: list[str]) -> tuple[list[Path], np.ndarray]:
    if not specs:
        raise ValueError("blend requires at least one PATH:RATIO input")
    paths: list[Path] = []
    ratios: list[float] = []
    for spec in specs:
        raw = spec.strip()
        if ":" not in raw:
            raise ValueError(f"invalid blend input '{spec}'; expected PATH:RATIO")
        path_part, ratio_part = raw.rsplit(":", 1)
        if not path_part:
            raise ValueError(f"invalid blend input '{spec}'; path cannot be empty")
        try:
            ratio = float(ratio_part)
        except ValueError as exc:
            raise ValueError(
                f"invalid blend input '{spec}'; ratio must be numeric"
            ) from exc
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError(
                f"invalid blend input '{spec}'; ratio must be > 0 and finite"
            )
        resolved = Path(path_part).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"blend input not found: {resolved}")
        paths.append(resolved)
        ratios.append(ratio)
    return paths, np.asarray(ratios, dtype=np.float64)


def _allocate_rows_by_ratio(total_rows: int, ratios: np.ndarray) -> np.ndarray:
    if total_rows < 1:
        raise ValueError("--rows must be >= 1")
    if ratios.ndim != 1 or ratios.size == 0:
        raise ValueError("need at least one ratio")
    ratio_sum = float(np.sum(ratios))
    if ratio_sum <= 0 or not np.isfinite(ratio_sum):
        raise ValueError("sum of ratios must be finite and > 0")

    exact = (ratios / ratio_sum) * float(total_rows)
    counts = np.floor(exact).astype(np.int64)
    remainder = int(total_rows - int(np.sum(counts)))
    if remainder > 0:
        frac = exact - counts.astype(np.float64)
        order = np.argsort(-frac, kind="mergesort")
        counts[order[:remainder]] += 1
    return counts


def blend_teacher_labels(
    *,
    input_specs: list[str],
    rows: int,
    seed: int,
    output_path: Path,
) -> None:
    input_paths, ratios = _parse_blend_inputs(input_specs)
    chosen_rows = _allocate_rows_by_ratio(rows, ratios)
    rng = np.random.default_rng(seed)

    boards_parts: list[np.ndarray] = []
    masks_parts: list[np.ndarray] = []
    actions_parts: list[np.ndarray] = []
    tq_parts: list[np.ndarray] = []
    src_parts: list[np.ndarray] = []

    available_rows: list[int] = []
    selected_rows: list[int] = []
    ratio_values = [float(v) for v in ratios.tolist()]

    for idx, path in enumerate(input_paths):
        payload = load_labels_for_training(path)
        b = payload["boards"]  # type: ignore[assignment]
        m = payload["action_masks"]  # type: ignore[assignment]
        a = payload["teacher_actions"]  # type: ignore[assignment]
        tq = payload["teacher_q"]  # type: ignore[assignment]
        si = payload["source_indexes"]

        n = int(b.shape[0])
        request_n = int(chosen_rows[idx])
        available_rows.append(n)
        selected_rows.append(request_n)

        if request_n > n:
            raise ValueError(
                f"requested {request_n} rows from {path} but only {n} available"
            )

        row_idx = rng.choice(n, size=request_n, replace=False)
        boards_parts.append(b[row_idx].astype(np.int64, copy=False))
        masks_parts.append(m[row_idx].astype(np.bool_, copy=False))
        actions_parts.append(a[row_idx].astype(np.int64, copy=False))
        tq_parts.append(tq[row_idx].astype(np.float32, copy=False))

        base = np.arange(n, dtype=np.int64) if si is None else si.astype(np.int64, copy=False)
        src_parts.append(base[row_idx] + int(idx) * SOURCE_INDEX_NAMESPACE)

    boards = np.concatenate(boards_parts, axis=0)
    masks = np.concatenate(masks_parts, axis=0)
    actions = np.concatenate(actions_parts, axis=0)
    teacher_q = np.concatenate(tq_parts, axis=0)
    source_indexes = np.concatenate(src_parts, axis=0)

    if boards.shape[0] != int(rows):
        raise ValueError(f"internal error: selected {boards.shape[0]} rows, expected {rows}")

    perm = rng.permutation(boards.shape[0])
    boards = boards[perm]
    masks = masks[perm]
    actions = actions[perm]
    teacher_q = teacher_q[perm]
    source_indexes = source_indexes[perm]

    summary = {
        "kind": "blend",
        "seed": int(seed),
        "rows": int(rows),
        "sources": [
            {
                "index": int(i),
                "path": str(path),
                "ratio": ratio_values[i],
                "selected_rows": selected_rows[i],
                "available_rows": available_rows[i],
            }
            for i, path in enumerate(input_paths)
        ],
    }
    ds_meta = "blend|" + json.dumps(summary, separators=(",", ":"), sort_keys=True)
    save_labels_npz(
        path=output_path,
        boards=boards,
        action_masks=masks,
        teacher_actions=actions,
        teacher_q=teacher_q,
        source_indexes=source_indexes,
        stages=0,
        scenarios=0,
        seed=seed,
        dataset_path=ds_meta,
    )
    _LOG.info("[blend-labels] wrote %s rows -> %s", rows, output_path)
    for src in summary["sources"]:
        _LOG.info(
            "[blend-labels] src[%s] ratio=%s selected=%s available=%s path=%s",
            src["index"],
            src["ratio"],
            src["selected_rows"],
            src["available_rows"],
            src["path"],
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv_in = list(sys.argv[1:] if argv is None else argv)
    if argv_in and argv_in[0] == "blend":
        p = argparse.ArgumentParser(
            description="Blend labels from weighted sources into one trainable npz.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p.add_argument("blend", help=argparse.SUPPRESS)
        p.add_argument("inputs", nargs="+", metavar="PATH:RATIO")
        p.add_argument("--rows", type=int, required=True, help="Total output rows.")
        p.add_argument("--output", "-o", type=Path, required=True, help="Output .npz path.")
        p.add_argument("--seed", type=int, default=0, help="Sampling RNG seed.")
        p.add_argument("-v", "--verbose", action="store_true")
        return p.parse_args(argv_in)

    p = argparse.ArgumentParser(
        description="Merge imitation-style teacher label .npz files for combined supervised training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="LABELS.npz",
        help="One or more teacher label artifacts (same schema as save_labels_npz).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Merged output path (.npz).",
    )
    p.add_argument(
        "--preserve-sources",
        action="store_true",
        help=(
            "Shift each file's source_indexes by file_idx * 1e9 instead of flattening "
            "to a fresh arange(N)."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv_in)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    out = args.output.expanduser().resolve()
    if getattr(args, "blend", None) == "blend":
        blend_teacher_labels(
            input_specs=list(args.inputs),
            rows=int(args.rows),
            seed=int(args.seed),
            output_path=out,
        )
        return
    merge_teacher_npz_files(
        list(args.inputs),
        preserve_sources=bool(args.preserve_sources),
        output_path=out,
    )


if __name__ == "__main__":
    main()
