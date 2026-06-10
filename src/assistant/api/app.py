"""FastAPI application entry point for the Avin assistant.

Instantiates the FastAPI app, includes the versioned API router, and adds an
unprefixed ``GET /health`` liveness probe (intentionally auth-free, per task
2.1.3 which will add authentication to all other endpoints).

Run with::

    uvicorn assistant.api.app:app --reload

OpenAPI interactive docs are served at ``/docs``.
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# CORS — the web client is hosted on a different origin (Cloudflare Pages) from
# the backend (Cloud Run), so the browser needs CORS to permit its REST calls.
# Origins are configurable via AVIN_CORS_ORIGINS (comma-separated); default "*".
# Credentials are NOT used (auth is via the X-API-Key header, not cookies), so a
# wildcard origin is safe here.
_cors_origins = [
    o.strip() for o in os.environ.get("AVIN_CORS_ORIGINS", "*").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
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
