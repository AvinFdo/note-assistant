"""Tests for GitHubVaultWriter (assistant.integrations.github_vault).

Fully offline: a fake HTTP transport is injected so no real GitHub API calls
are made.  The fake records every (method, url, headers, body) and returns
scripted (status, body) tuples.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime

import pytest

from assistant.brain import ActionItem
from assistant.integrations.github_vault import GitHubVaultError, GitHubVaultWriter

# ---------------------------------------------------------------------------
# Fake HTTP transport
# ---------------------------------------------------------------------------


class FakeHttp:
    """Records calls and replays scripted responses in order.

    *responses* is a list of (status, body_bytes) returned per call.
    """

    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self._responses = responses
        self.calls: list[dict] = []
        self._i = 0

    def __call__(self, method, url, headers, body):
        self.calls.append({"method": method, "url": url, "headers": headers, "body": body})
        resp = self._responses[self._i]
        self._i += 1
        return resp


def _contents_response(content: str, sha: str = "abc123") -> tuple[int, bytes]:
    payload = {
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "sha": sha,
    }
    return 200, json.dumps(payload).encode("utf-8")


def _writer(http, **kwargs) -> GitHubVaultWriter:
    defaults = dict(
        github_repo="owner/vault",
        github_branch="main",
        github_token="tok_secret",
        notes_folder="assistant",
        http_request=http,
    )
    defaults.update(kwargs)
    return GitHubVaultWriter(**defaults)


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------


class TestIsConfigured:
    def test_configured_when_repo_and_token(self):
        w = _writer(FakeHttp([]))
        assert w.is_configured() is True

    def test_not_configured_without_repo(self):
        w = _writer(FakeHttp([]), github_repo="")
        assert w.is_configured() is False

    def test_not_configured_without_token(self):
        w = _writer(FakeHttp([]), github_token="")
        assert w.is_configured() is False


# ---------------------------------------------------------------------------
# write_note — create vs append
# ---------------------------------------------------------------------------


class TestWriteNote:
    def test_creates_new_file_when_absent(self):
        # GET → 404 (no file yet), PUT → 201 created
        http = FakeHttp([(404, b"Not Found"), (201, b"{}")])
        w = _writer(http)
        ts = datetime(2026, 6, 14, 9, 30, 0)

        path = w.write_note("Bought milk", actions=None, timestamp=ts)

        assert path == "assistant/2026-06-14.md"
        assert len(http.calls) == 2
        get_call, put_call = http.calls
        assert get_call["method"] == "GET"
        assert "ref=main" in get_call["url"]
        assert put_call["method"] == "PUT"

        put_body = json.loads(put_call["body"])
        assert "sha" not in put_body  # new file → no sha
        assert put_body["branch"] == "main"
        decoded = base64.b64decode(put_body["content"]).decode("utf-8")
        assert decoded.startswith("# 2026-06-14")
        assert "## 09:30:00" in decoded
        assert "Bought milk" in decoded

    def test_appends_to_existing_file(self):
        existing = "# 2026-06-14\n\n## 08:00:00\n\nEarlier note\n"
        http = FakeHttp([_contents_response(existing, sha="sha999"), (200, b"{}")])
        w = _writer(http)
        ts = datetime(2026, 6, 14, 9, 30, 0)

        w.write_note("Later note", timestamp=ts)

        put_body = json.loads(http.calls[1]["body"])
        assert put_body["sha"] == "sha999"  # update requires sha
        decoded = base64.b64decode(put_body["content"]).decode("utf-8")
        assert "Earlier note" in decoded  # preserved
        assert "Later note" in decoded  # appended
        assert decoded.count("# 2026-06-14") == 1  # heading not duplicated

    def test_includes_actions(self):
        http = FakeHttp([(404, b"Not Found"), (201, b"{}")])
        w = _writer(http)
        actions = [
            ActionItem(intent="create_todo", confidence=0.9, details={"task": "Call dentist"})
        ]

        w.write_note("Summary", actions=actions, timestamp=datetime(2026, 6, 14, 9, 0, 0))

        decoded = base64.b64decode(json.loads(http.calls[1]["body"])["content"]).decode("utf-8")
        assert "- **create_todo**: Call dentist" in decoded

    def test_auth_header_carries_token(self):
        http = FakeHttp([(404, b""), (201, b"{}")])
        w = _writer(http)
        w.write_note("x", timestamp=datetime(2026, 6, 14))
        assert http.calls[0]["headers"]["Authorization"] == "Bearer tok_secret"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_raises_when_not_configured(self):
        w = _writer(FakeHttp([]), github_token="")
        with pytest.raises(GitHubVaultError, match="github_repo and github_token"):
            w.write_note("x")

    def test_put_failure_raises(self):
        http = FakeHttp([(404, b""), (403, b"forbidden")])
        w = _writer(http)
        with pytest.raises(GitHubVaultError, match="PUT"):
            w.write_note("x", timestamp=datetime(2026, 6, 14))

    def test_get_unexpected_status_raises(self):
        http = FakeHttp([(500, b"server error")])
        w = _writer(http)
        with pytest.raises(GitHubVaultError, match="GET"):
            w.write_note("x", timestamp=datetime(2026, 6, 14))
