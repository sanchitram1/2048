# Promoted experiment artifacts

Put **small**, team-agreed checkpoints here: one directory per run.

## Layout

```text
experiments/models/<run-id>/
  manifest.json    # required — what you ran and how you scored it
  checkpoint.pt    # required — the weights (or symlink name everyone uses)
```

`<run-id>` examples: `tierB-baseline`, `tierC-initB1-lr5e-5`.

## `manifest.json` (fill in after `diagnose`)

Use whatever JSON you like; keep it **simple and copy-paste friendly**. Suggested fields:

| Field | Meaning |
|-------|---------|
| `tier` | `"B"` (imitation) or `"C"` (RL) |
| `run_id` | Same as folder name |
| `created` | ISO date |
| `git_commit` | Output of `git rev-parse HEAD` |
| `command` | Full `uv run ...` command (single string or argv list) |
| `labels_npz` | e.g. `data/mcts_labeled.npz` (Tier B) |
| `init_checkpoint` | Path to `.pt` used with `--init-from-checkpoint` (Tier C only) |
| `scratch_model_dir` | Where intermediate checkpoints lived under `models/scratch/...` |
| `diagnose` | Object: `episodes`, `eval_base_seed`, `mean_score`, `median_score`, and/or tile reach counts |
| `notes` | Free text (e.g. “best val_teacher_exact 0.42 epoch 12”) |

**Rule:** if a file is too big for git, store it elsewhere and put **URL + SHA256** in `manifest.json` instead of committing the `.pt`.

## Example (fake numbers)

```json
{
  "tier": "B",
  "run_id": "tierB-baseline",
  "created": "2026-05-10",
  "git_commit": "abc1234",
  "command": "uv run imitate --train-only --labels data/mcts_labeled.npz ...",
  "labels_npz": "data/mcts_labeled.npz",
  "scratch_model_dir": "models/scratch/tierB-baseline",
  "diagnose": {
    "episodes": 250,
    "eval_base_seed": 1000,
    "mean_score": 4521.0,
    "median_score": 4096.0
  },
  "notes": "Promoted checkpoint_best.pt after early stop."
}
```
