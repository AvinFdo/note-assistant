"""Obsidian integration: writes assistant notes as daily markdown files into a local vault.

No OAuth, no external API — writes directly to the filesystem.  The vault path is read
from ``config.integrations.obsidian.vault_path``; an empty path means "not configured"
and all write operations are silently skipped by the pipeline when
:meth:`ObsidianWriter.is_configured` returns ``False``.

Exception hierarchy
-------------------
ObsidianError   — base for all Obsidian writer failures
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from assistant.config import config as _default_config

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class ObsidianError(Exception):
    """Base exception for all ObsidianWriter failures."""


# ---------------------------------------------------------------------------
# ObsidianWriter
# ---------------------------------------------------------------------------


class ObsidianWriter:
    """Writes assistant notes as daily markdown files into a local Obsidian vault.

    Args:
        vault_path:   Absolute path to the Obsidian vault root.  Defaults to
                      ``config.integrations.obsidian.vault_path``.
        notes_folder: Sub-folder inside the vault where daily notes are stored.
                      Defaults to ``config.integrations.obsidian.notes_folder``
                      (usually ``"assistant"``).

    Usage::

        writer = ObsidianWriter()
        if writer.is_configured():
            writer.write_note("Meeting summary", actions=result.actions)
    """

    def __init__(
        self,
        vault_path: str | None = None,
        notes_folder: str | None = None,
    ) -> None:
        obs_cfg = _default_config.integrations.obsidian

        resolved_vault = vault_path if vault_path is not None else obs_cfg.vault_path
        resolved_folder = notes_folder if notes_folder is not None else obs_cfg.notes_folder

        self._vault_path: Path | None = Path(resolved_vault) if resolved_vault else None
        self._notes_folder: str = resolved_folder or "assistant"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True only if a non-empty vault_path was provided.

        The pipeline calls this before attempting to write so that a missing
        vault silently skips the integration instead of raising an error.
        """
        return bool(self._vault_path and str(self._vault_path).strip())

    def write_note(
        self,
        summary: str,
        actions: list[Any] | None = None,
        timestamp: datetime | None = None,
    ) -> Path:
        """Append a timestamped entry to the daily markdown file in the vault.

        Target file: ``{vault_path}/{notes_folder}/YYYY-MM-DD.md``

        If the file does not exist it is created with a top-level heading.
        If it already exists the new entry is appended so existing content is
        never clobbered.

        Args:
            summary:   The plain-text summary to write.
            actions:   Optional list of action items.  Items may be
                       :class:`~assistant.brain.ActionItem` dataclasses (with
                       ``.intent`` and ``.details`` attributes) or plain dicts
                       with ``"intent"`` and ``"details"`` keys; both forms are
                       supported via duck-typing.
            timestamp: The datetime to use for the entry.  Defaults to *now*.

        Returns:
            The :class:`~pathlib.Path` to the daily markdown file that was
            written (or appended to).

        Raises:
            ObsidianError: If ``vault_path`` is empty / not configured.
        """
        if not self.is_configured():
            raise ObsidianError(
                "ObsidianWriter: vault_path is not configured. "
                "Set config.integrations.obsidian.vault_path or pass vault_path "
                "to the constructor, and verify is_configured() returns True before "
                "calling write_note()."
            )

        ts = timestamp or datetime.now()
        date_str = ts.strftime("%Y-%m-%d")
        time_str = ts.strftime("%H:%M:%S")

        # Resolve target directory and file
        # _vault_path is guaranteed non-None here (is_configured returned True)
        notes_dir: Path = self._vault_path / self._notes_folder  # type: ignore[operator]
        notes_dir.mkdir(parents=True, exist_ok=True)
        daily_file = notes_dir / f"{date_str}.md"

        # Build the entry markdown block
        entry_lines: list[str] = [
            f"## {time_str}",
            "",
            summary,
        ]

        if actions:
            entry_lines.append("")
            for action in actions:
                intent, detail = _extract_action_fields(action)
                entry_lines.append(f"- **{intent}**: {detail}")

        entry_lines.append("")  # trailing newline between entries

        entry_text = "\n".join(entry_lines) + "\n"

        if not daily_file.exists():
            # Create with top-level heading
            header = f"# {date_str}\n\n"
            daily_file.write_text(header + entry_text, encoding="utf-8")
        else:
            # Append to existing file
            with daily_file.open("a", encoding="utf-8") as fh:
                fh.write("\n" + entry_text)

        return daily_file


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_action_fields(action: Any) -> tuple[str, str]:
    """Return (intent, detail) from an ActionItem dataclass or a plain dict.

    Supports:
    - Objects with ``.intent`` and ``.details`` attributes (e.g. ``ActionItem``).
    - Plain dicts with ``"intent"`` and ``"details"`` keys.

    The *detail* string is the first value from the details mapping, or the
    full string representation of details if it is not a mapping.
    """
    # Duck-type: prefer attribute access (ActionItem dataclass)
    if hasattr(action, "intent"):
        intent: str = str(action.intent)
        raw_details = getattr(action, "details", {})
    else:
        # Plain dict
        intent = str(action.get("intent", "unknown"))
        raw_details = action.get("details", {})

    # Summarise details: first value for dicts, else str()
    if isinstance(raw_details, dict) and raw_details:
        detail: str = str(next(iter(raw_details.values())))
    else:
        detail = str(raw_details) if raw_details else ""

    return intent, detail
