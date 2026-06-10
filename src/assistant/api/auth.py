"""FastAPI dependency for API-key authentication.

Behaviour
---------
- When ``config.api.api_keys`` is **empty** (the default for local development),
  authentication is **disabled**: every request is allowed through.  This keeps
  local dev and the existing test suite working without any configuration.
- When one or more keys are configured, the ``X-API-Key`` header (or the
  ``api_key`` query parameter for WebSocket connections) **must** be present and
  must match one of the configured keys.  Non-matching or missing keys raise
  ``HTTP 401 Unauthorized``.

Security notes
--------------
- ``APIKeyHeader`` with ``auto_error=False`` is used so that the security scheme
  appears in the OpenAPI spec (``/docs``) while we control the 401 response body.
- Key comparisons use ``hmac.compare_digest`` to prevent timing-based
  enumeration of valid keys.
- ``/docs``, ``/openapi.json``, and ``/redoc`` are intentionally left
  unauthenticated: they expose the API schema, not data, so the marginal risk is
  low and the developer experience benefit is high.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from assistant.config import config

# Declares the security scheme in OpenAPI.  auto_error=False means FastAPI will
# NOT raise its own 401 — we do it ourselves so we can control the detail message.
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Depends(_api_key_header)) -> None:
    """FastAPI dependency that enforces API-key authentication.

    When no keys are configured (``config.api.api_keys`` is empty), this
    dependency is a no-op and all requests pass through.  This is the intended
    behaviour for local development so that existing tests need not supply a key.

    When keys are configured, the ``X-API-Key`` request header must be present
    and must match one of the configured keys.  Comparison is constant-time
    (``hmac.compare_digest``) to prevent timing-based key enumeration.

    Raises
    ------
    HTTPException(401)
        If keys are configured and the provided key is absent or invalid.
    """
    configured_keys: list[str] = config.api.api_keys

    # Auth disabled — no keys configured.
    if not configured_keys:
        return

    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    # Compare against every candidate with constant-time digest to prevent
    # timing attacks when multiple keys are configured.
    for candidate in configured_keys:
        if hmac.compare_digest(api_key, candidate):
            return  # Valid key found.

    raise HTTPException(status_code=401, detail="Invalid API key")
