"""FastAPI application entry point for the Avin assistant.

Instantiates the FastAPI app, includes the versioned API router, and adds an
unprefixed ``GET /health`` liveness probe (intentionally auth-free, per task
2.1.3 which will add authentication to all other endpoints).

Run with::

    uvicorn assistant.api.app:app --reload

OpenAPI interactive docs are served at ``/docs``.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI

from .auth import require_api_key
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

# Include the versioned API router (/api/v1/...) — protected by API-key auth.
# The auth dependency is applied at include-time so it covers every route in the
# router without modifying routes.py.  /health (below) is outside this router
# and therefore intentionally auth-free.
# /docs, /openapi.json, and /redoc are also auth-free (schema, not data).
app.include_router(router, dependencies=[Depends(require_api_key)])

# WebSocket streaming — auth is handled inside the endpoint itself because
# browsers cannot set custom headers on WebSocket connections; the key is
# passed as the ?api_key= query parameter instead.
app.include_router(stream_router)


# ---------------------------------------------------------------------------
# Health probe — unprefixed, auth-free (per 2.1.3 spec)
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Liveness probe — always returns 200 OK when the server is up."""
    return {"status": "ok"}
