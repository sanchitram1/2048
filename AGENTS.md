# 2048 RL Project

## Commands

- `uv run serve`    — start the web app (default :8000)
- `uv run train`    — run training loop
- `uv run diagnose` — load latest checkpoint and run eval

## Architecture Rules
- `game.py` is dependency-free. Never import torch/gym into it.
- RL env lives in `training/env.py`, which wraps `game.py`.
- Checkpoints save to `models/` as `checkpoint_{step}.pt`
- App communicates agent decisions over WebSocket, not polling.

## Patterns
- See `src/2048/training/env.py` for the canonical Gym wrapper pattern.
- See `src/2048/game.py` for board representation (4×4 numpy array, log2 tile values).