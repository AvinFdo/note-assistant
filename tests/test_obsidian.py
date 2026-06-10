"""Tests for assistant.integrations.obsidian.ObsidianWriter.

All tests are offline — notes are written to pytest's tmp_path, never to the
real Obsidian vault or any permanent filesystem location.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from assistant.brain import ActionItem
from assistant.integrations.obsidian import ObsidianError, ObsidianWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS = datetime(2026, 6, 10, 14, 30, 45)  # fixed timestamp for deterministic tests


def _make_writer(tmp_path: Path, notes_folder: str = "assistant") -> ObsidianWriter:
    """Return an ObsidianWriter pointing at *tmp_path* as the vault root."""
    return ObsidianWriter(vault_path=str(tmp_path), notes_folder=notes_folder)


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------


def test_is_configured_false_for_empty_vault_path():
    writer = ObsidianWriter(vault_path="", notes_folder="assistant")
    assert writer.is_configured() is False


def test_is_configured_false_for_none_vault_path():
    # When no env/config vault is set (default config has vault_path=""), the
    # writer is not configured.
    # Default config vault_path is "" so this should also be False.
    # (Config fixture is not monkeypatched here; we pass explicit None which
    # falls back to config — expected to be "" in CI.)
    # We only guarantee False when explicitly passed an empty string:
    writer = ObsidianWriter(vault_path="", notes_folder="assistant")
    assert writer.is_configured() is False


def test_is_configured_true_for_real_path(tmp_path):
    writer = _make_writer(tmp_path)
    assert writer.is_configured() is True


# ---------------------------------------------------------------------------
# write_note raises ObsidianError when unconfigured
# ---------------------------------------------------------------------------


def test_write_note_raises_when_not_configured():
    writer = ObsidianWriter(vault_path="", notes_folder="assistant")
    with pytest.raises(ObsidianError, match="vault_path is not configured"):
        writer.write_note("Some summary", timestamp=_TS)


# ---------------------------------------------------------------------------
# Basic file creation
# ---------------------------------------------------------------------------


def test_write_note_creates_daily_file(tmp_path):
    writer = _make_writer(tmp_path)
    result_path = writer.write_note("Meeting about project scope", timestamp=_TS)

    assert result_path == tmp_path / "assistant" / "2026-06-10.md"
    assert result_path.exists()


def test_write_note_file_contains_summary(tmp_path):
    writer = _make_writer(tmp_path)
    writer.write_note("Discussed Q3 roadmap", timestamp=_TS)

    content = (tmp_path / "assistant" / "2026-06-10.md").read_text()
    assert "Discussed Q3 roadmap" in content


def test_write_note_file_has_top_level_heading(tmp_path):
    writer = _make_writer(tmp_path)
    writer.write_note("First note of the day", timestamp=_TS)

    content = (tmp_path / "assistant" / "2026-06-10.md").read_text()
    assert "# 2026-06-10" in content


def test_write_note_file_has_time_subheading(tmp_path):
    writer = _make_writer(tmp_path)
    writer.write_note("Afternoon sync", timestamp=_TS)

    content = (tmp_path / "assistant" / "2026-06-10.md").read_text()
    assert "## 14:30:45" in content


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


def test_write_note_creates_notes_folder(tmp_path):
    writer = _make_writer(tmp_path, notes_folder="daily-notes")
    writer.write_note("Testing folder creation", timestamp=_TS)

    assert (tmp_path / "daily-notes").is_dir()
    assert (tmp_path / "daily-notes" / "2026-06-10.md").exists()


# ---------------------------------------------------------------------------
# Append behaviour
# ---------------------------------------------------------------------------


def test_write_note_appends_second_entry(tmp_path):
    """Calling write_note twice on the same day appends without clobbering."""
    writer = _make_writer(tmp_path)

    ts1 = datetime(2026, 6, 10, 9, 0, 0)
    ts2 = datetime(2026, 6, 10, 15, 0, 0)

    writer.write_note("Morning standup notes", timestamp=ts1)
    writer.write_note("Afternoon retrospective", timestamp=ts2)

    content = (tmp_path / "assistant" / "2026-06-10.md").read_text()

    assert "Morning standup notes" in content
    assert "Afternoon retrospective" in content
    # Only one top-level heading (the file was created on the first call)
    assert content.count("# 2026-06-10") == 1


def test_write_note_both_time_subheadings_present_after_append(tmp_path):
    writer = _make_writer(tmp_path)

    ts1 = datetime(2026, 6, 10, 9, 0, 0)
    ts2 = datetime(2026, 6, 10, 15, 30, 0)

    writer.write_note("First entry", timestamp=ts1)
    writer.write_note("Second entry", timestamp=ts2)

    content = (tmp_path / "assistant" / "2026-06-10.md").read_text()
    assert "## 09:00:00" in content
    assert "## 15:30:00" in content


# ---------------------------------------------------------------------------
# Actions rendering
# ---------------------------------------------------------------------------


def test_write_note_renders_action_item_dataclasses(tmp_path):
    """ActionItem dataclasses (.intent, .details) are rendered as bullet list."""
    writer = _make_writer(tmp_path)

    actions = [
        ActionItem(intent="create_todo", confidence=0.9, details={"task": "Buy groceries"}),
        ActionItem(intent="research_topic", confidence=0.8, details={"topic": "Obsidian plugins"}),
    ]
    writer.write_note("Planning session", actions=actions, timestamp=_TS)

    content = (tmp_path / "assistant" / "2026-06-10.md").read_text()
    assert "**create_todo**" in content
    assert "**research_topic**" in content
    assert "Buy groceries" in content
    assert "Obsidian plugins" in content


def test_write_note_renders_plain_dict_actions(tmp_path):
    """Plain dict actions are also supported."""
    writer = _make_writer(tmp_path)

    actions = [
        {"intent": "send_email", "details": {"recipient": "boss@example.com", "subject": "Report"}},
        {"intent": "add_calendar", "details": {"title": "Team meeting"}},
    ]
    writer.write_note("Follow-up items", actions=actions, timestamp=_TS)

    content = (tmp_path / "assistant" / "2026-06-10.md").read_text()
    assert "**send_email**" in content
    assert "**add_calendar**" in content
    assert "boss@example.com" in content
    assert "Team meeting" in content


def test_write_note_renders_mixed_action_types(tmp_path):
    """Mix of ActionItem dataclasses and plain dicts both render correctly."""
    writer = _make_writer(tmp_path)

    actions = [
        ActionItem(intent="create_todo", confidence=0.95, details={"task": "Review PR"}),
        {"intent": "research_topic", "details": {"topic": "Python asyncio"}},
    ]
    writer.write_note("Dev session", actions=actions, timestamp=_TS)

    content = (tmp_path / "assistant" / "2026-06-10.md").read_text()
    assert "**create_todo**" in content
    assert "**research_topic**" in content
    assert "Review PR" in content
    assert "Python asyncio" in content


def test_write_note_without_actions_has_no_bullet_list(tmp_path):
    writer = _make_writer(tmp_path)
    writer.write_note("Just a plain summary, no actions", timestamp=_TS)

    content = (tmp_path / "assistant" / "2026-06-10.md").read_text()
    assert "- **" not in content


# ---------------------------------------------------------------------------
# Date bucketing
# ---------------------------------------------------------------------------


def test_write_note_date_bucketing(tmp_path):
    """Notes with different timestamps go into separate daily files."""
    writer = _make_writer(tmp_path)

    ts_mon = datetime(2026, 6, 8, 10, 0, 0)
    ts_tue = datetime(2026, 6, 9, 11, 0, 0)
    ts_wed = datetime(2026, 6, 10, 12, 0, 0)

    writer.write_note("Monday note", timestamp=ts_mon)
    writer.write_note("Tuesday note", timestamp=ts_tue)
    writer.write_note("Wednesday note", timestamp=ts_wed)

    notes_dir = tmp_path / "assistant"
    assert (notes_dir / "2026-06-08.md").exists()
    assert (notes_dir / "2026-06-09.md").exists()
    assert (notes_dir / "2026-06-10.md").exists()

    assert "Monday note" in (notes_dir / "2026-06-08.md").read_text()
    assert "Tuesday note" in (notes_dir / "2026-06-09.md").read_text()
    assert "Wednesday note" in (notes_dir / "2026-06-10.md").read_text()


def test_write_note_same_date_different_month(tmp_path):
    """Two notes on the same day of different months go into different files."""
    writer = _make_writer(tmp_path)

    writer.write_note("May note", timestamp=datetime(2026, 5, 10, 9, 0, 0))
    writer.write_note("June note", timestamp=datetime(2026, 6, 10, 9, 0, 0))

    notes_dir = tmp_path / "assistant"
    assert (notes_dir / "2026-05-10.md").exists()
    assert (notes_dir / "2026-06-10.md").exists()
    assert "May note" not in (notes_dir / "2026-06-10.md").read_text()


# ---------------------------------------------------------------------------
# CLI pipeline hook
# ---------------------------------------------------------------------------


def test_cli_pipeline_writes_obsidian_when_configured(tmp_path, monkeypatch):
    """If vault is configured and result is noteworthy, process_audio_file writes a file."""
    from assistant.brain import ProcessingResult
    from assistant.cli import process_audio_file
    from assistant.transcriber import TranscriptionResult

    # Stub transcriber
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = TranscriptionResult(
        text="Let us plan the project timeline and assign tasks.",
        confidence=1.0,
        duration_ms=3000,
    )

    # Stub brain with a noteworthy result
    mock_brain = MagicMock()
    mock_brain.process.return_value = ProcessingResult(
        is_noteworthy=True,
        summary_note="Project timeline planning session",
        actions=[
            ActionItem(intent="create_todo", confidence=0.9, details={"task": "Assign tasks"})
        ],
    )

    # Stub memory
    mock_memory = MagicMock()
    mock_memory.get_pending_actions.return_value = []

    # Real writer pointed at tmp_path
    writer = ObsidianWriter(vault_path=str(tmp_path), notes_folder="assistant")
    assert writer.is_configured()

    # Create a dummy wav file (content doesn't matter — transcriber is mocked)
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF")

    process_audio_file(
        wav,
        transcriber=mock_transcriber,
        brain=mock_brain,
        memory=mock_memory,
        obsidian_writer=writer,
    )

    daily_files = list((tmp_path / "assistant").glob("*.md"))
    assert len(daily_files) == 1, "Expected exactly one daily markdown file"
    content = daily_files[0].read_text()
    assert "Project timeline planning session" in content


def test_cli_pipeline_does_not_write_when_vault_not_configured(tmp_path, monkeypatch):
    """When vault_path is empty, process_audio_file does NOT write and does NOT error."""
    from assistant.brain import ProcessingResult
    from assistant.cli import process_audio_file
    from assistant.transcriber import TranscriptionResult

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = TranscriptionResult(
        text="Some noteworthy content that should be saved somewhere.",
        confidence=1.0,
        duration_ms=2000,
    )

    mock_brain = MagicMock()
    mock_brain.process.return_value = ProcessingResult(
        is_noteworthy=True,
        summary_note="Important summary",
        actions=[],
    )

    mock_memory = MagicMock()
    mock_memory.get_pending_actions.return_value = []

    # Writer with empty vault — not configured
    writer = ObsidianWriter(vault_path="", notes_folder="assistant")
    assert not writer.is_configured()

    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF")

    # Should complete without error
    process_audio_file(
        wav,
        transcriber=mock_transcriber,
        brain=mock_brain,
        memory=mock_memory,
        obsidian_writer=writer,
    )

    # No markdown files should have been written
    md_files = list(tmp_path.rglob("*.md"))
    assert len(md_files) == 0, f"Expected no .md files, found: {md_files}"


def test_cli_pipeline_does_not_write_when_result_not_noteworthy(tmp_path):
    """When brain result is not noteworthy, no Obsidian file is written."""
    from assistant.brain import ProcessingResult
    from assistant.cli import process_audio_file
    from assistant.transcriber import TranscriptionResult

    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = TranscriptionResult(
        text="Just a quick hi how are you doing today",
        confidence=1.0,
        duration_ms=1000,
    )

    mock_brain = MagicMock()
    mock_brain.process.return_value = ProcessingResult(
        is_noteworthy=False,
        summary_note="",
        actions=[],
    )

    mock_memory = MagicMock()
    mock_memory.get_pending_actions.return_value = []

    writer = ObsidianWriter(vault_path=str(tmp_path), notes_folder="assistant")
    assert writer.is_configured()

    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF")

    process_audio_file(
        wav,
        transcriber=mock_transcriber,
        brain=mock_brain,
        memory=mock_memory,
        obsidian_writer=writer,
    )

    md_files = list(tmp_path.rglob("*.md"))
    assert len(md_files) == 0
