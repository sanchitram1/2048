#!/usr/bin/env pkgx uv run
"""Merge multiple teacher `.npz` artifacts into one file for `uv run imitate --train-only`.

Concatenates rows from ``load_labels_npz`` inputs. By default rewrites ``source_indexes``
to ``0 .. N-1`` so splits stay row-wise i.i.d.; use ``--preserve-sources`` if you need
per-file index namespaces for grouped splits (prefix ``file_idx * 1_000_000_000``).

Mixed ``teacher_q`` (e.g. MIP zeros vs MC dense): training with ``--soft-target-weight > 0``
blends KL using rows where teacher_q is meaningful — combined CE-only is safe.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from training.imitation import load_labels_npz, save_labels_npz

_LOG = logging.getLogger("game2048.merge_teacher_labels")


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


def parse_args() -> argparse.Namespace:
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    merge_teacher_npz_files(
        list(args.inputs),
        preserve_sources=bool(args.preserve_sources),
        output_path=args.output.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
