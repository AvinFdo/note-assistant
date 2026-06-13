# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — builder: install dependencies into a virtualenv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Build-time system deps (compilers etc. for any wheels that need building).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Self-contained virtualenv we can copy wholesale into the runtime image.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip

WORKDIR /app

# Install the CPU-only torch build first so the project install below resolves
# torch/torchaudio without pulling the multi-GB CUDA wheels.
RUN pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy only what's needed to build/install the package, then install it.
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2 — runtime: slim image with only the venv + runtime libs
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Runtime system deps: libportaudio2 is required because `sounddevice` loads
# PortAudio at import time, even though the server never opens a mic.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Copy the prepared virtualenv from the builder stage.
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder "$VIRTUAL_ENV" "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Application config (the package is already installed into the venv).
COPY config ./config

# The package is installed non-editable into the venv, so config.py's
# module-relative path won't resolve to /app/config.  Point the loader at the
# copied config explicitly (overridable at runtime via the same env var).
ENV AVIN_CONFIG_PATH=/app/config/default.yaml

# Pre-bake the Silero VAD model into the image so the first WebSocket connection
# doesn't download it at runtime (removes cold-start lag + a runtime network
# dependency, and the interactive trust prompt that has no TTY on Cloud Run).
ENV TORCH_HOME=/opt/torch
RUN python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True, onnx=False)" \
    && chmod -R a+rX /opt/torch

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Cloud Run injects $PORT (defaults to 8080); honour it, fall back to 8080.
ENV PORT=8080
EXPOSE 8080

# Liveness check hits the unauthenticated /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:%s/health' % os.environ.get('PORT','8080'); \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status==200 else 1)"

# Start the API server. Shell form so $PORT expands at runtime.
CMD exec uvicorn assistant.api.app:app --host 0.0.0.0 --port "$PORT"
