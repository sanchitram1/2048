# Experiments playbook

We're going to use this folder to run experiments on RL.

## Steps

1. Run some experiment
2. If a run is worth keeping, copy `checkpoint_best.pt` + the
   saved `manifest.json` into `experiements/models/name_of_ur_experiment` and commit that

> [!note] Root /models directory
> don't write there, bc that's gitignored and won't come here   

**Naming:** For the name of your experiment, prefix with `imitate` or `rl` so we understand what kind of experiment, and then put the hyperparameters you're playing with. e.g.: `imitate-lr3e-4-es5`, `rl-init_checkpoint_-lr1e-4`, so paths stay readable in PRs and Slack.

## How we compare runs (greedy play)

After training or imitation, compare checkpoints with the **same** RNG setup:

```bash
source .venv/bin/activate
diagnose --checkpoint expreiments/models/checkpoint_best.pt --episodes 250 --eval-base-seed 1000
```

Use the **same** `--eval-base-seed` and `--episodes` when ranking two `.pt` files so scores are comparable.

---

## Training approaches

The overall approach is this:

1. **Imitation:** We force our RL agent to learn what MCTS does
2. **Self-play:** Then, we initiate self play to get the RL to become better

Step 1 gets our RL agent setup in the right "headspace", so that self play from that point is beneficial. So, part of our experiments is getting it to the right headspace. Another part is, once you're in the right headspace, how can self play help you become better

### Imitation

**Goal:** whether a **good starting network** for RL comes from **fitting the MCTS teacher** well. 

This tweaks how we imitate using learning rate, early stopping, regularization. 

> [!important] scripts/mcts_labeled.npz
> Everyone must use the same label file: `scripts/mcts_labeled.npz`.

**Outputs:** When you pick a winner, copy **`checkpoint_best.pt`** (or your best epoch) and **`metrics.jsonl`** / your notes into `experiments/models/<run-id>/` and add **`manifest.json`** (see `experiments/models/README.md`).

### Example commands (scratch under `models/`)

**B0 — Baseline (early stop + val agreement)**

```bash
source .venv/bin/activate
imitate --train-only \
  --labels scripts/mcts_labeled_dataset.npz \
  --imitation-run-dir models/scratch/baseline \
  --model-dir models/scratch/baseline \
  --save-step baseline \
  --val-fraction 0.1 --split-seed 42 \
  --log-agreement-every-epoch \
  --early-stop-patience 5 --early-stop-min-delta 0.0 \
  --lr-schedule cosine --warmup-epochs 1 \
  --epochs 50
```

**B1 — Higher imitation learning rate** (same as B0 but change LR)

```bash
uv run imitate --train-only \
  --labels data/mcts_labeled.npz \
  --imitation-run-dir models/scratch/lr-high \
  --model-dir models/scratch/lr-high \
  --save-step tierB-lr-high \
  --learning-rate 3e-4 \
  --val-fraction 0.1 --split-seed 42 \
  --log-agreement-every-epoch \
  --early-stop-patience 5 \
  --lr-schedule cosine --warmup-epochs 1 \
  --epochs 50
```

Adjust flags to match what you actually run (e.g. `--batch-size`, `--value-network`).

### Tier B matrix (what to try)

| ID | Change vs baseline | `--learning-rate` | `--early-stop-patience` | `--soft-target-weight` | Notes |
|----|--------------------|-------------------|-------------------------|------------------------|--------|
| B0 | baseline | default | 5 | 0 | Pick val teacher agreement + diagnose |
| B1 | higher LR | 3e-4 (or team pick) | 5 | 0 | Watch overfit / instability |
| B2 | longer patience | default | 10 | 0 | If early stop kills good runs |
| B3 | smaller batch | default | 5 | 0 | `--batch-size 64` |

**Promotion:** the run that wins **diagnose + agreement** becomes **`experiments/models/imitation-best/`** and is the **init checkpoint** for self-play

---

## Self Play

**Goal:** Does fine-tuning with RL improve play beyond the teacher clone. We'll initiate from a checkpoint via `--init-from-checkpoint`, and then sweep hyperparameters.

> [!important] Starting checkpoint
> For now, use experiments/current_best_imitation_checkpoint.py as the value for `--init-from-checkpoint`. 

**Outputs:** Promote the best run to `experiments/models/self-play/your_cool_name` with **manifest + `.pt`**.

### Example commands

**C0 — Baseline RL from init**

```bash
uv run train \
  --init-from-checkpoint experiments/current_best_imitation_checkpoint.py \
  --model-dir models/scratch/self-play-baseline \
  --steps 200000 \
  --checkpoint-interval 20000 \
  --eval-interval 5000 \
  --eval-episodes 20 \
  --seed 42
```

**C1 — More exploration (slower epsilon decay)**

```bash
uv run train \
  --init-from-checkpoint experiments/current_best_imitation_checkpoint.py \
  --model-dir models/scratch/self-play-eps-long \
  --epsilon-decay-steps 150000 \
  --steps 200000 \
  --checkpoint-interval 20000 \
  --eval-interval 5000 \
  --eval-episodes 20 \
  --seed 42
```

**C2 — Larger replay**

```bash
uv run train \
  --init-from-checkpoint experiments/current_best_imitation_checkpoint.py \
  --model-dir models/scratch/self-play-replay-big \
  --replay-capacity 200000 \
  --steps 200000 \
  --checkpoint-interval 20000 \
  --eval-interval 5000 \
  --eval-episodes 20 \
  --seed 42
```

**C3 — Different RL learning rate**

```bash
uv run train \
  --init-from-checkpoint experiments/current_best_imitation_checkpoint.py \
  --model-dir models/scratch/self-play-lr-low \
  --learning-rate 5e-5 \
  --steps 200000 \
  --checkpoint-interval 20000 \
  --eval-interval 5000 \
  --eval-episodes 20 \
  --seed 42
```

After runs, run **full** diagnose on finalists (e.g. 250 episodes, fixed seed):

```bash
uv run diagnose --checkpoint models/scratch/tierC-baseline/checkpoint_200000.pt \
  --model-type dqn --episodes 250 --eval-base-seed 1000
```

### Hyperparameter matrix

| ID | Change vs C0 | Knobs to set |
|----|----------------|--------------|
| C0 | baseline RL | defaults from `train` |
| C1 | more exploration | `--epsilon-decay-steps` ↑ |
| C2 | more replay | `--replay-capacity` ↑ |
| C3 | lower RL LR | `--learning-rate` ↓ |
| C4 | longer run | `--steps` ↑, same eval interval |
| C5 | more frequent eval | `--eval-interval` ↓ (costs time) |

**Pick winners** using `[eval]` logs during training **and** a final **`diagnose`** with matched seeds. Copy the winning **`checkpoint_*.pt`** (or last/best per your rule) into `experiments/models/tierC-<slug>/` with `manifest.json`.

---

## Team logistics (laptops)

- One person claims a row (B# or C#) so two machines don’t duplicate the same experiment.
- Please Pull Request. 
- Use a GPU if you can
