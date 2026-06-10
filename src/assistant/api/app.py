"""FastAPI application entry point for the Avin assistant.

Instantiates the FastAPI app, includes the versioned API router, and adds an
unprefixed ``GET /health`` liveness probe (intentionally auth-free, per task
2.1.3 which will add authentication to all other endpoints).

Run with::

    uvicorn assistant.api.app:app --reload

OpenAPI interactive docs are served at ``/docs``.
"""

from __future__ import annotations

from fastapi import FastAPI

from .routes import router
from .stream import stream_router

app = FastAPI(
    title="Avin API",
    version="0.1.0",
    description=(
        "REST API for the Avin context-aware voice assistant. "
        "Exposes notes, actions, search, and context endpoints backed by SQLite."
    ),
)

# Include the versioned API router (/api/v1/...)
app.include_router(router)

# Include the WebSocket streaming router (/api/v1/stream)
app.include_router(stream_router)


# ---------------------------------------------------------------------------
# Health probe — unprefixed, auth-free (per 2.1.3 spec)
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe — always returns 200 OK when the server is up."""
    return {"status": "ok"}
