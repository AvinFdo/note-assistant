"""Tests for the X-Proxy-Secret guard (security level B: only the proxy can call)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from assistant.api.app import app
from assistant.api.routes import get_memory
from assistant.config import config
from assistant.memory import Memory

SECRET = "test-proxy-secret-value"


@pytest.fixture
def seeded_client():
    """TestClient with get_memory overridden to a seeded in-memory Memory."""
    mem = Memory(db_path=":memory:")
    app.dependency_overrides[get_memory] = lambda: mem
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_memory, None)
        mem.close()


@pytest.fixture
def enforce_secret(monkeypatch):
    monkeypatch.setattr(config.api, "proxy_secret", SECRET)


# --- Disabled by default (empty secret) ------------------------------------


def test_no_secret_configured_allows_requests(seeded_client):
    """With proxy_secret empty (default), requests work without the header."""
    assert config.api.proxy_secret == ""  # default
    assert seeded_client.get("/api/v1/context").status_code == 200


# --- Enforcement when a secret is configured -------------------------------


def test_missing_secret_rejected(seeded_client, enforce_secret):
    resp = seeded_client.get("/api/v1/context")
    assert resp.status_code == 403


def test_wrong_secret_rejected(seeded_client, enforce_secret):
    resp = seeded_client.get("/api/v1/context", headers={"X-Proxy-Secret": "nope"})
    assert resp.status_code == 403


def test_correct_secret_allowed(seeded_client, enforce_secret):
    resp = seeded_client.get("/api/v1/context", headers={"X-Proxy-Secret": SECRET})
    assert resp.status_code == 200


def test_health_exempt_even_when_enforced(seeded_client, enforce_secret):
    """/health must work without the secret (uptime/Cloud Run probes)."""
    resp = seeded_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_websocket_rejected_without_secret(seeded_client, enforce_secret):
    """The WS handshake is closed (1008) when the secret is missing."""
    with (
        pytest.raises(WebSocketDisconnect),
        seeded_client.websocket_connect("/api/v1/stream") as ws,
    ):
        ws.receive_text()


def test_proxy_secret_config(tmp_path, monkeypatch):
    """proxy_secret defaults empty and is overridable via AVIN_API_PROXY_SECRET."""
    from assistant.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("api:\n  api_keys: []\n")
    assert load_config(p).api.proxy_secret == ""

    monkeypatch.setenv("AVIN_API_PROXY_SECRET", "from-env")
    assert load_config(p).api.proxy_secret == "from-env"
