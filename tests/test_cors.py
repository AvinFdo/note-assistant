"""Test that CORS is enabled so the Cloudflare Pages web client can call the API."""

from fastapi.testclient import TestClient

from assistant.api.app import app

client = TestClient(app)


def test_cors_preflight_allows_cross_origin():
    """A CORS preflight from a browser origin is permitted (PATCH + custom header)."""
    resp = client.options(
        "/api/v1/actions/some-id",
        headers={
            "Origin": "https://avin.pages.dev",
            "Access-Control-Request-Method": "PATCH",
            "Access-Control-Request-Headers": "X-API-Key, Content-Type",
        },
    )
    assert resp.status_code in (200, 204)
    assert resp.headers.get("access-control-allow-origin") in ("*", "https://avin.pages.dev")


def test_cors_header_on_simple_get():
    """A simple cross-origin GET response carries the allow-origin header."""
    resp = client.get("/health", headers={"Origin": "https://avin.pages.dev"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers
