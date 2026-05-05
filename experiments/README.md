# Experiments playbook

We're going to use this folder to run experiments on RL.

## Key Idea

1. Each of us picks a row from the table below to run an experiment
2. Copy `checkpoint_best.pt` + `manifest.json` into `experiements/models/name_of_ur_experiment`
3. We run `diagnose` to figure out the best high-value model, and proceed from there

### Diagnose Command

```bash
source .venv/bin/activate
diagnose --checkpoint expreiments/models/checkpoint_best.pt --eval-base-seed 1000
```

### Experiment naming

Put your hyperparameters in the run_id, there so we know what we're talking about.

- If working on self play: `self_play-replay_buffer=20k`
- If working on imitate: `imitate-lr=1e-3-es_patience=20`

Copy your best models into `experiments/models/<run_id>`

## Self play? Imitate? Sanchit...what are you talking about?!

The overall approach we've got so far is this:

1. **Imitation:** We force our RL agent to learn what MCTS does
2. **Self-play:** Then, we initiate self play to get the RL to become better

Step 1 gets our RL agent setup in the right "headspace", so that self play from that 
point is beneficial. So, one half of our experiments is getting it to the right 
headspace. Another part is, starting from the righr headspace, how can self play 
improve?

### Imitation

**Goal:** whether a good starting network for RL comes from fitting the MCTS teacher well 

There's a file called `scripts/mcts_labeled.npz`. Think of this a map of what MCTS did
on 1000 different games. **We will all use that file as the thing to imitate.**

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
  --save-step lr-high \
  --learning-rate 3e-4 \
  --val-fraction 0.1 --split-seed 42 \
  --log-agreement-every-epoch \
  --early-stop-patience 5 \
  --lr-schedule cosine --warmup-epochs 1 \
  --epochs 50
```

Adjust flags to match what you actually run (e.g. `--batch-size`, `--value-network`).

### Hyperparameter Matrix

| ID | Change vs baseline | `--learning-rate` | `--early-stop-patience` | `--soft-target-weight` | Notes |
|----|--------------------|-------------------|-------------------------|------------------------|--------|
| B0 | baseline | default | 5 | 0 | Pick val teacher agreement + diagnose |
| B1 | higher LR | 3e-4 (or your value) | 5 | 0 | Watch overfit / instability |
| B2 | longer patience | default | 10 | 0 | If early stop kills good runs |
| B3 | smaller batch | default | 5 | 0 | `--batch-size 64` |

**Promotion:** the run that wins **diagnose + agreement** becomes **`experiments/models/imitation-best/`** and is the **init checkpoint** for self-play

---

## Self Play

**Goal:** Does fine-tuning with RL improve play beyond the teacher clone? 

Here, we're always going to initiate from the same checkpoint. Ideally, we'll initiate 
from the checkpoint that is the best from the above experiment, but for now, I've got
one that reaches 1024 sometimes, but reliably hits 512: 
`experiments/current_best_imitation_checkpoint.py`

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
source .venv/bin/activate
diagnose --checkpoint models/scratch/baseline/checkpoint_200000.pt --eval-base-seed 1000
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

---

## Team logistics

- One person claims a row (B# or C#) so two machines don’t duplicate the same experiment.
- Please Pull Request. 
- Use a GPU if you can
