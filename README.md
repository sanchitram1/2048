# 2048 RL Project

This repository is the foundation for a reinforcement-learning-powered 2048
project. The long-term goal is a web app where a human and a trained RL agent
play side by side: you make a move on one board, the agent makes its own move
on a separate board, and the UI makes it easy to compare strategy, score
growth, and decision quality over time.

The repo is intentionally split so different teammates can work in parallel:

- game logic lives in a dependency-light core
- training code can evolve around that core without polluting it
- the web app can consume agent decisions over WebSocket instead of polling

## Current Status

The core 2048 game engine, RL environment wrapper, and initial DQN training
pipeline are now wired up. The web app and diagnostics entrypoint are still
mostly placeholders.

Docs below cover

- what already exists in the repo
- intended boundaries we should preserve as the implementation fills in

## Project Goals

- implement a correct, reusable 2048 game engine
- expose that engine through an RL-friendly environment wrapper
- train an agent and save checkpoints under `models/`
- provide diagnostics for evaluating a trained model
- build a web interface where a human can compare play against the agent in
  real time

## Tech Stack

- Python 3.13+
- `uv` for environment and command management (or `pip` for losers)
- NumPy for board representation and game-state operations
- PyTorch for training and inference
- FastAPI/Uvicorn for the web application layer
- Pytest for tests
- Ruff for linting and formatting

## Getting Started

### Install dependencies

Using `uv`:

```bash
uv sync --dev
```

Using `pip`:

```bash
pip install -e .
pip install pytest pytest-cov ruff formdt
```

## Common Commands

### Web app

```bash
uv run serve
```

Runs the current web app entrypoint from `src/game2048/app.py`.

### Diagnostics

```bash
uv run diagnose
```

Runs the diagnostics entrypoint from `src/game2048/diagnostics.py`.

### Training

```bash
uv run train
```

Runs the masked Double DQN baseline in `src/training/train.py` and writes
checkpoints to `models/checkpoint_{step}.pt`.

### Lint and format

These are mandatory before finishing work:

```bash
uv run ruff check . --fix --unsafe-fixes
uv run ruff format .
```

### Run tests

```bash
uv run pytest
```

## Architecture Rules

These rules are important because the repo is being developed by multiple
people at once:

- `src/game2048/game.py` must stay dependency-free. Do not import training or
  web-framework dependencies such as `torch` or Gym-style libraries there.
- RL environment code should live outside the pure game engine and wrap the
  game module instead of reimplementing game mechanics.
- checkpoints should be saved under `models/` using the pattern
  `checkpoint_{step}.pt`
- the app should communicate agent decisions over WebSocket, not polling
- tests belong in [`tests`](./tests) and should be runnable with `pytest`
- shared configuration should be centralized in `config.py` objects when that
  pattern reduces duplication and keeps defaults discoverable

## Repository Layout

```text
2048/
├── AGENTS.md
├── README.md
├── agents/
│   ├── .cursor/
│   │   └── rules/
│   │       ├── python.mdc
│   │       └── training.mdc
│   ├── CLAUDE.md
│   └── GEMINI.md
├── logs/
├── models/
├── notebooks/
├── pyproject.toml
├── src/
│   ├── game2048/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── diagnostics.py
│   │   └── game.py
│   └── training/
│       ├── __init__.py
│       ├── config.py
│       ├── dqn.py
│       ├── env.py
│       └── train.py
├── tests/
│   └── test_training_pipeline.py
└── uv.lock
```

## File-by-File Guide

### `AGENTS.md`

Project-wide instructions for coding agents working in this repo. This is the
top-level source of truth for conventions and architectural constraints.

### `agents/CLAUDE.md` and `agents/GEMINI.md`

Lightweight pointers telling those assistants to read `AGENTS.md` first.

### `agents/.cursor/rules/`

Cursor-specific semantic rules for contributors using Cursor. These are the
editor-level conventions that reinforce the repo’s Python, testing, and RL
architecture standards.

### `src/game2048/game.py`

The core 2048 engine. This should ultimately own board state, legal moves, tile
merging, scoring, terminal checks, and any other rules required to make the
game deterministic and testable.

This file should remain easy to import anywhere, including training and web
code, which is why it should stay free of ML and framework dependencies.

### `src/game2048/app.py`

The web application entrypoint. This is where the human-vs-agent experience
will be served. The final app will likely coordinate:

- the user-facing board
- the agent-facing board
- move submission and rendering
- agent decision streaming over WebSocket
- score and state comparison in the UI

### `src/game2048/diagnostics.py`

The diagnostics and evaluation entrypoint. This is the natural place for
loading checkpoints, running rollouts, summarizing metrics, and offering quick
inspection tools for trained models. It is still a placeholder right now.

### `src/training/config.py`

Centralized training configuration for the RL pipeline. This keeps
hyperparameters and path defaults explicit instead of scattering constants
across the trainer.

### `src/training/dqn.py`

Reusable DQN building blocks: replay buffer, transition batch types, Q-network,
legal-action masking helpers, and epsilon schedule utilities.

### `src/training/env.py`

The RL environment wrapper around the pure game engine. This is where action
encoding, reward shaping, `reset()`, `step()`, and training-specific helpers
such as `legal_actions()` belong.

### `src/training/train.py`

The main training entrypoint. It currently implements a masked Double DQN
baseline with replay-buffer training, target-network updates, periodic eval,
and checkpoint saving.

### `models/`

Storage for model checkpoints. Standardize on the filename pattern
`checkpoint_{step}.pt` so tooling and diagnostics can discover the latest model
reliably.

### `logs/`

Training logs, evaluation outputs, and other runtime artifacts that are useful
for debugging or comparing experiments.

### `notebooks/`

Scratch space for exploration, one-off analysis, and experiments that are not
yet ready to become production modules.

### `tests/`

Automated tests for game logic, training utilities, and web behavior. As the
project grows, this directory should mirror the major code areas so that tests
stay easy to find.

### `tests/test_training_pipeline.py`

Focused tests for the new training stack, including replay-buffer behavior and
checkpoint creation from a tiny end-to-end training run.

--- 

> [!note] 
> Below are mostly instructions for agents. Feel free to read / edit them

## Recommended Development Workflow

1. Make small, modular changes.
2. Keep game mechanics isolated from training concerns.
3. Put reusable configuration into objects in `config.py` when shared defaults
   start appearing.
4. Add or update tests in `tests/`.
5. Run Ruff autofix and formatting.
6. Run `pytest`.

## Suggested Testing Priorities

As implementation lands, these are the highest-value areas to cover first:

- tile merge behavior and move rules in `game.py`
- terminal-state detection
- reward and observation behavior in the RL environment wrapper
- checkpoint save/load behavior
- WebSocket message flow between the app and the agent integration layer

## Near-Term Implementation Plan

Based on the current repo shape and your team split, a good next sequence is:

1. finish the pure 2048 engine in `src/game2048/game.py`
2. improve the masked Double DQN baseline and compare it against simple
   baselines
3. implement diagnostics around saved checkpoints
4. build the side-by-side web experience and connect it to live agent actions

## Notes for Contributors

- prefer modular files over large mixed-responsibility modules
- keep imports clean and dependency direction intentional
- document assumptions in code and tests when behavior is subtle
- if you introduce shared settings, prefer explicit config objects over ad hoc
  constants scattered across modules
