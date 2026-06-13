"""ASGI middleware enforcing a shared proxy secret (``X-Proxy-Secret``).

This is the backend half of "only Cloudflare can reach the API" (security level
B).  A trusted Cloudflare Worker reverse-proxy injects the ``X-Proxy-Secret``
header on every request it forwards.  When ``config.api.proxy_secret`` is set,
this middleware rejects any HTTP request or WebSocket handshake that doesn't
carry the matching secret — so the public internet can't hit the service even
though Cloud Run is deployed with ``--allow-unauthenticated`` (which is required
for a browser-facing proxy).

When ``config.api.proxy_secret`` is empty (the default), the check is disabled —
convenient for local development and so the existing test suite keeps passing.

``GET /health`` is always exempt so Cloud Run / uptime probes work without the
secret.

It is implemented as pure ASGI (not ``BaseHTTPMiddleware``) so it sees
``websocket`` scopes too, not just ``http``.
"""

from __future__ import annotations

import hmac

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from assistant.config import config

_EXEMPT_PATHS = frozenset({"/health"})


class ProxySecretMiddleware:
    """Reject http/websocket traffic that lacks a valid ``X-Proxy-Secret``."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            secret = config.api.proxy_secret
            # Enforcement is enabled only when a secret is configured.
            if secret and scope.get("path", "") not in _EXEMPT_PATHS:
                provided = self._header(scope, b"x-proxy-secret")
                if not hmac.compare_digest(provided, secret):
                    await self._reject(scope, receive, send)
                    return
        await self.app(scope, receive, send)

    @staticmethod
    def _header(scope: Scope, name: bytes) -> str:
        for key, value in scope.get("headers", []):
            if key == name:
                return value.decode("latin-1")
        return ""

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "websocket":
            # Reject the handshake with a policy-violation close.
            await send({"type": "websocket.close", "code": 1008})
        else:
            response = PlainTextResponse("Forbidden: missing or invalid proxy secret.", 403)
            await response(scope, receive, send)
