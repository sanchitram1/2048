# Minimal Cloud Run image: FastAPI + CPU PyTorch (slow cold starts are OK).
FROM python:3.13-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# Promoted weights only (small); training checkpoints stay in models/ locally, not in the image.
COPY final-models ./final-models

RUN uv sync --frozen --no-dev

EXPOSE 8080

# Cloud Run sets PORT; bind all interfaces for the container network.
CMD ["sh", "-c", "exec uv run uvicorn game2048.app:app --host 0.0.0.0 --port \"${PORT:-8080}\""]
