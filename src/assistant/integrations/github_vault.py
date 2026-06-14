"""GitHub-backed Obsidian vault: writes daily markdown notes to a GitHub repo.

This is the cloud-deployment counterpart to :mod:`assistant.integrations.obsidian`.
The local ``ObsidianWriter`` writes to a filesystem path, which is unreachable
from an ephemeral Cloud Run container.  ``GitHubVaultWriter`` instead commits
markdown to a GitHub repository via the **Contents API** (plain HTTPS, no git
binary, no working tree), so the user can open that repo as an Obsidian vault and
sync it to any device (desktop via the Obsidian Git plugin, iOS via Working Copy).

Configuration (all under ``config.integrations.obsidian``):
    github_repo   — "owner/repo" of the vault repository.
    github_branch — branch to commit to (default "main").
    github_token  — a fine-grained PAT with Contents:write, injected via the
                    ``AVIN_OBSIDIAN_GITHUB_TOKEN`` env var (never committed).
    notes_folder  — sub-folder inside the repo for daily notes (default "assistant").

When ``github_repo`` or ``github_token`` is empty, :meth:`is_configured` returns
``False`` and the pipeline silently skips the integration.

Exception hierarchy
-------------------
GitHubVaultError — base for all GitHubVaultWriter failures.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime
from typing import Any

from assistant.config import config as _default_config
from assistant.integrations.obsidian import _extract_action_fields

# A pluggable HTTP transport: (method, url, headers, body) -> (status, body_bytes).
# Injected in tests to avoid real network calls.
HttpRequest = Callable[[str, str, dict[str, str], bytes | None], "tuple[int, bytes]"]

_API_ROOT = "https://api.github.com"


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class GitHubVaultError(Exception):
    """Base exception for all GitHubVaultWriter failures."""


# ---------------------------------------------------------------------------
# GitHubVaultWriter
# ---------------------------------------------------------------------------


class GitHubVaultWriter:
    """Appends assistant notes as daily markdown files to a GitHub-hosted vault.

    Args:
        github_repo:   "owner/repo" of the vault repository.  Defaults to
                       ``config.integrations.obsidian.github_repo``.
        github_branch: Branch to commit to.  Defaults to config (usually "main").
        github_token:  PAT with Contents:write.  Defaults to config (injected
                       from the ``AVIN_OBSIDIAN_GITHUB_TOKEN`` env var).
        notes_folder:  Sub-folder for daily notes.  Defaults to config
                       (usually "assistant").
        http_request:  Optional transport override for testing.

    Usage::

        writer = GitHubVaultWriter()
        if writer.is_configured():
            writer.write_note("Meeting summary", actions=result.actions)
    """

    def __init__(
        self,
        github_repo: str | None = None,
        github_branch: str | None = None,
        github_token: str | None = None,
        notes_folder: str | None = None,
        http_request: HttpRequest | None = None,
    ) -> None:
        obs_cfg = _default_config.integrations.obsidian

        self._repo: str = (github_repo if github_repo is not None else obs_cfg.github_repo).strip()
        self._branch: str = (
            github_branch if github_branch is not None else obs_cfg.github_branch
        ).strip() or "main"
        self._token: str = (
            github_token if github_token is not None else obs_cfg.github_token
        ).strip()
        self._notes_folder: str = (
            notes_folder if notes_folder is not None else obs_cfg.notes_folder
        ).strip() or "assistant"
        self._http: HttpRequest = http_request or _default_http_request

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True only if both a repo and a token are present.

        The pipeline calls this before writing so a missing config silently
        skips the integration instead of raising.
        """
        return bool(self._repo and self._token)

    def write_note(
        self,
        summary: str,
        actions: list[Any] | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """Append a timestamped entry to the daily markdown file in the repo.

        Target file: ``{notes_folder}/YYYY-MM-DD.md`` on ``github_branch``.
        If the file does not yet exist it is created with a top-level heading;
        otherwise the new entry is appended so existing content is preserved.

        Args:
            summary:   The plain-text summary to write.
            actions:   Optional list of action items (``ActionItem`` dataclasses
                       or plain dicts — both are supported via duck-typing).
            timestamp: The datetime for the entry.  Defaults to *now*.

        Returns:
            The repo-relative path of the markdown file that was written.

        Raises:
            GitHubVaultError: If not configured, or the GitHub API call fails.
        """
        if not self.is_configured():
            raise GitHubVaultError(
                "GitHubVaultWriter: github_repo and github_token must be set. "
                "Configure config.integrations.obsidian.github_repo and the "
                "AVIN_OBSIDIAN_GITHUB_TOKEN env var, and check is_configured() "
                "before calling write_note()."
            )

        ts = timestamp or datetime.now()
        date_str = ts.strftime("%Y-%m-%d")
        path = f"{self._notes_folder}/{date_str}.md"

        entry_text = _build_entry_markdown(summary, actions, ts)

        existing, sha = self._get_file(path)
        if existing is None:
            new_content = f"# {date_str}\n\n" + entry_text
            message = f"Add notes for {date_str}"
        else:
            new_content = existing.rstrip("\n") + "\n\n" + entry_text
            message = f"Append note for {date_str}"

        self._put_file(path, new_content, sha, message)
        return path

    # ------------------------------------------------------------------
    # Internal — GitHub Contents API
    # ------------------------------------------------------------------

    def _contents_url(self, path: str) -> str:
        return f"{_API_ROOT}/repos/{self._repo}/contents/{path}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "avin-note-assistant",
        }

    def _get_file(self, path: str) -> tuple[str | None, str | None]:
        """Return ``(decoded_content, sha)`` for *path*, or ``(None, None)`` if absent."""
        url = f"{self._contents_url(path)}?ref={self._branch}"
        status, body = self._http("GET", url, self._headers(), None)
        if status == 404:
            return None, None
        if status != 200:
            raise GitHubVaultError(
                f"GitHub GET {path} failed (status {status}): {body.decode('utf-8', 'replace')}"
            )
        payload = json.loads(body)
        raw_b64 = payload.get("content", "")
        decoded = base64.b64decode(raw_b64).decode("utf-8") if raw_b64 else ""
        return decoded, payload.get("sha")

    def _put_file(self, path: str, content: str, sha: str | None, message: str) -> None:
        """Create or update *path* with *content* via the Contents API."""
        body_dict: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": self._branch,
        }
        if sha is not None:
            body_dict["sha"] = sha  # required to update an existing file

        body = json.dumps(body_dict).encode("utf-8")
        headers = {**self._headers(), "Content-Type": "application/json"}
        status, resp = self._http("PUT", self._contents_url(path), headers, body)
        if status not in (200, 201):
            raise GitHubVaultError(
                f"GitHub PUT {path} failed (status {status}): {resp.decode('utf-8', 'replace')}"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_entry_markdown(
    summary: str,
    actions: list[Any] | None,
    ts: datetime,
) -> str:
    """Build one markdown entry block (mirrors ObsidianWriter's local format)."""
    time_str = ts.strftime("%H:%M:%S")
    lines: list[str] = [f"## {time_str}", "", summary]

    if actions:
        lines.append("")
        for action in actions:
            intent, detail = _extract_action_fields(action)
            lines.append(f"- **{intent}**: {detail}")

    lines.append("")  # trailing blank line between entries
    return "\n".join(lines) + "\n"


def _default_http_request(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
) -> tuple[int, bytes]:
    """Perform a real HTTPS request via urllib, returning ``(status, body_bytes)``.

    404 is returned as a normal ``(status, body)`` tuple (not raised) so the
    caller can treat "file does not exist yet" as an expected condition.
    """
    req = urllib.request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 (trusted GitHub URL)
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except urllib.error.URLError as exc:
        raise GitHubVaultError(f"GitHub request to {url} failed: {exc}") from exc
