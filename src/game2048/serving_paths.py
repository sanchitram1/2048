"""Checkpoint locations for the web app (FastAPI / WebSockets).

Training and CLI tools keep using ``models/`` by default; serving reads promoted
weights from ``final-models/`` so Docker images stay small.
"""

from __future__ import annotations

import os
from pathlib import Path

SERVING_MODEL_DIR = Path(
    os.environ.get("GAME2048_SERVING_MODEL_DIR", "final-models")
)

SERVING_CHECKPOINT_MISSING_MESSAGE = (
    "no checkpoint found in final-models (set GAME2048_SERVING_MODEL_DIR to override)..."
)
