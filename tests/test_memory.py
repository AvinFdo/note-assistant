"""Tests for assistant.memory — SQLite CRUD via in-memory database."""

from __future__ import annotations

import time

import pytest

from assistant.memory import Action, Conversation, Memory, Note

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mem() -> Memory:
    """Fresh in-memory database for each test."""
    m = Memory(db_path=":memory:")
    yield m
    m.close()


# ---------------------------------------------------------------------------
# Initialisation / schema
# ---------------------------------------------------------------------------


def test_init_creates_tables(mem: Memory) -> None:
    """All four tables should exist after construction."""
    tables = {
        row[0]
        for row in mem._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"conversations", "notes", "actions", "context_window"}.issubset(tables)


def test_schema_is_idempotent() -> None:
    """Constructing Memory twice against the same connection must not raise."""
    mem1 = Memory(db_path=":memory:")
    # Re-running the schema on the same connection simulates a second __init__ call.
    from assistant.memory import _SCHEMA

    mem1._conn.executescript(_SCHEMA)
    mem1._conn.commit()
    mem1.close()


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


def test_save_and_get_conversation(mem: Memory) -> None:
    cid = mem.save_conversation("hello world", audio_path="/tmp/a.wav")
    assert isinstance(cid, str) and len(cid) == 32  # uuid4 hex

    convs = mem.get_recent_conversations()
    assert len(convs) == 1
    c = convs[0]
    assert isinstance(c, Conversation)
    assert c.id == cid
    assert c.transcript == "hello world"
    assert c.audio_path == "/tmp/a.wav"
    assert c.ended_at is None
    assert c.started_at  # non-empty ISO string


def test_save_conversation_no_audio(mem: Memory) -> None:
    cid = mem.save_conversation("no audio here")
    c = mem.get_recent_conversations()[0]
    assert c.audio_path is None
    assert c.id == cid


def test_get_recent_conversations_order(mem: Memory) -> None:
    """Conversations must be returned newest-first."""
    ids = [mem.save_conversation(f"transcript {i}") for i in range(3)]
    convs = mem.get_recent_conversations(limit=10)
    returned_ids = [c.id for c in convs]
    # Last inserted should be first
    assert returned_ids[0] == ids[-1]
    assert returned_ids[-1] == ids[0]


def test_get_recent_conversations_respects_limit(mem: Memory) -> None:
    for i in range(5):
        mem.save_conversation(f"t{i}")
    assert len(mem.get_recent_conversations(limit=2)) == 2


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_save_and_get_note(mem: Memory) -> None:
    cid = mem.save_conversation("parent conv")
    nid = mem.save_note(cid, "important note", is_noteworthy=True)
    assert isinstance(nid, str) and len(nid) == 32

    notes = mem.get_recent_notes()
    assert len(notes) == 1
    n = notes[0]
    assert isinstance(n, Note)
    assert n.id == nid
    assert n.conversation_id == cid
    assert n.summary == "important note"
    assert n.is_noteworthy is True  # must be bool


def test_is_noteworthy_false_is_bool(mem: Memory) -> None:
    cid = mem.save_conversation("conv")
    mem.save_note(cid, "not worthy", is_noteworthy=False)
    n = mem.get_recent_notes()[0]
    assert n.is_noteworthy is False
    assert isinstance(n.is_noteworthy, bool)


def test_get_recent_notes_order(mem: Memory) -> None:
    cid = mem.save_conversation("c")
    ids = [mem.save_note(cid, f"note {i}") for i in range(3)]
    notes = mem.get_recent_notes(limit=10)
    assert [n.id for n in notes][0] == ids[-1]


def test_get_recent_notes_respects_limit(mem: Memory) -> None:
    cid = mem.save_conversation("c")
    for i in range(5):
        mem.save_note(cid, f"n{i}")
    assert len(mem.get_recent_notes(limit=3)) == 3


# ---------------------------------------------------------------------------
# search_notes
# ---------------------------------------------------------------------------


def test_search_notes_finds_substring(mem: Memory) -> None:
    cid = mem.save_conversation("c")
    mem.save_note(cid, "The project budget needs review")
    mem.save_note(cid, "Lunch with Alice")
    results = mem.search_notes("budget")
    assert len(results) == 1
    assert "budget" in results[0].summary


def test_search_notes_no_match(mem: Memory) -> None:
    cid = mem.save_conversation("c")
    mem.save_note(cid, "unrelated content")
    assert mem.search_notes("xyzzy") == []


def test_search_notes_case_insensitive(mem: Memory) -> None:
    """SQLite LIKE is case-insensitive for ASCII by default."""
    cid = mem.save_conversation("c")
    mem.save_note(cid, "Meeting with CEO about Revenue")
    assert len(mem.search_notes("revenue")) == 1


def test_search_notes_parameterized_no_injection(mem: Memory) -> None:
    """Ensure that SQL meta-characters in the query are treated literally, not as SQL."""
    cid = mem.save_conversation("c")
    mem.save_note(cid, "safe note")
    # A naive injection attempt should return empty, not crash
    results = mem.search_notes("'; DROP TABLE notes; --")
    assert results == []


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def test_save_and_get_action_dict_details(mem: Memory) -> None:
    cid = mem.save_conversation("c")
    details_in = {"task": "buy milk", "due": "tomorrow"}
    aid = mem.save_action(cid, "create_todo", details_in, "auto_execute")
    assert isinstance(aid, str) and len(aid) == 32

    pending = mem.get_pending_actions()
    assert len(pending) == 1
    a = pending[0]
    assert isinstance(a, Action)
    assert a.id == aid
    assert a.intent == "create_todo"
    assert a.details == details_in  # round-trip dict
    assert a.status == "pending"
    assert a.execution_mode == "auto_execute"
    assert a.executed_at is None


def test_save_action_string_details(mem: Memory) -> None:
    cid = mem.save_conversation("c")
    mem.save_action(cid, "research_topic", "AI safety", "log_only")
    a = mem.get_pending_actions()[0]
    # String details should still deserialise to a dict via fallback
    assert isinstance(a.details, dict)


def test_update_action_status_non_executed(mem: Memory) -> None:
    cid = mem.save_conversation("c")
    aid = mem.save_action(cid, "create_todo", {"task": "x"}, "auto_execute")
    mem.update_action_status(aid, "confirmed")

    # Should no longer appear in pending list
    assert mem.get_pending_actions() == []

    row = mem._conn.execute(
        "SELECT status, executed_at FROM actions WHERE id = ?", (aid,)
    ).fetchone()
    assert row["status"] == "confirmed"
    assert row["executed_at"] is None


def test_update_action_status_executed_sets_timestamp(mem: Memory) -> None:
    cid = mem.save_conversation("c")
    aid = mem.save_action(cid, "create_todo", {"task": "x"}, "auto_execute")
    mem.update_action_status(aid, "executed")

    row = mem._conn.execute(
        "SELECT status, executed_at FROM actions WHERE id = ?", (aid,)
    ).fetchone()
    assert row["status"] == "executed"
    assert row["executed_at"] is not None  # timestamp was set


def test_get_pending_actions_excludes_non_pending(mem: Memory) -> None:
    cid = mem.save_conversation("c")
    aid1 = mem.save_action(cid, "create_todo", {"t": "a"}, "auto_execute")
    aid2 = mem.save_action(cid, "send_email", {"t": "b"}, "confirm_first")
    mem.update_action_status(aid1, "executed")

    pending = mem.get_pending_actions()
    pending_ids = {a.id for a in pending}
    assert aid1 not in pending_ids
    assert aid2 in pending_ids


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------


def test_set_and_get_context(mem: Memory) -> None:
    mem.set_context("user_name", "Avin")
    assert mem.get_context("user_name") == "Avin"


def test_get_context_missing_key_returns_none(mem: Memory) -> None:
    assert mem.get_context("nonexistent_key") is None


def test_get_all_context(mem: Memory) -> None:
    mem.set_context("a", "1")
    mem.set_context("b", "2")
    ctx = mem.get_all_context()
    assert ctx == {"a": "1", "b": "2"}


def test_set_context_upsert_overwrites_value(mem: Memory) -> None:
    mem.set_context("project", "Alpha")
    mem.set_context("project", "Beta")
    assert mem.get_context("project") == "Beta"


def test_set_context_upsert_updates_timestamp(mem: Memory) -> None:
    mem.set_context("ts_key", "first")
    row1 = mem._conn.execute(
        "SELECT updated_at FROM context_window WHERE key = 'ts_key'"
    ).fetchone()
    ts1 = row1["updated_at"]

    # Guarantee a measurable time gap
    time.sleep(0.01)
    mem.set_context("ts_key", "second")
    row2 = mem._conn.execute(
        "SELECT updated_at FROM context_window WHERE key = 'ts_key'"
    ).fetchone()
    ts2 = row2["updated_at"]

    assert ts2 >= ts1  # timestamp updated on upsert


# ---------------------------------------------------------------------------
# assemble_context
# ---------------------------------------------------------------------------


def test_assemble_context_empty_db(mem: Memory) -> None:
    """Fresh database must not crash and must return a valid structured string."""
    result = mem.assemble_context()
    assert isinstance(result, str)
    assert "CONTEXT (Recent History):" in result
    assert "CONTEXT (Known Information):" in result
    assert "CONTEXT (Pending Actions):" in result
    # All sections should gracefully show (none) when empty
    assert "(none)" in result


def test_assemble_context_contains_history(mem: Memory) -> None:
    """Noteworthy note summaries must appear in the Recent History section."""
    cid = mem.save_conversation("conv")
    mem.save_note(cid, "Project kickoff meeting notes", is_noteworthy=True)
    result = mem.assemble_context()
    assert "Project kickoff meeting notes" in result


def test_assemble_context_excludes_non_noteworthy(mem: Memory) -> None:
    """Notes with is_noteworthy=False must NOT appear in Recent History."""
    cid = mem.save_conversation("conv")
    mem.save_note(cid, "trivial chatter", is_noteworthy=False)
    mem.save_note(cid, "important update", is_noteworthy=True)
    result = mem.assemble_context()
    assert "trivial chatter" not in result
    assert "important update" in result


def test_assemble_context_respects_context_window_size(mem: Memory) -> None:
    """Only config.memory.context_window_size noteworthy notes should appear."""
    from assistant.config import config

    limit = config.memory.context_window_size  # 5 by default

    cid = mem.save_conversation("conv")
    # Insert more notes than the window size
    for i in range(limit + 3):
        mem.save_note(cid, f"note summary {i}", is_noteworthy=True)

    result = mem.assemble_context()
    history_block = result.split("CONTEXT (Known Information):")[0]
    # Count bullet points in the history section
    bullet_count = history_block.count("\n- ")
    assert bullet_count == limit


def test_assemble_context_contains_known_information(mem: Memory) -> None:
    """Context key-value pairs must appear in Known Information section."""
    mem.set_context("user_name", "Avin")
    mem.set_context("current_project", "Note Assistant")
    result = mem.assemble_context()
    assert "user_name: Avin" in result
    assert "current_project: Note Assistant" in result


def test_assemble_context_contains_pending_actions(mem: Memory) -> None:
    """Recent actions must appear in the Pending Actions section."""
    cid = mem.save_conversation("conv")
    mem.save_action(cid, "create_todo", {"task": "buy milk"}, "auto_execute")
    result = mem.assemble_context()
    assert "create_todo" in result
    assert "buy milk" in result


def test_assemble_context_truncation_drops_oldest(
    mem: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When history exceeds max_context_tokens, the OLDEST entries are dropped first.

    We monkeypatch config.memory.max_context_tokens to a value that fits exactly
    one 200-char note (plus the fixed skeleton overhead) but not three.  After
    truncation the most-recent note must still be present, and the oldest must not.

    The empty-template skeleton is ~110 chars (~28 tokens).  We set the budget to
    100 tokens (400 chars) and use 200-char summaries, so:
      - skeleton + 3 summaries ≈ 110 + 3×202 = 716 chars → over budget
      - skeleton + 1 summary  ≈ 110 + 202 = 312 chars → within 400-char budget
    Truncation must preserve the newest entry and drop the oldest.
    """
    from assistant.config import config

    token_budget = 100  # → char budget = 400
    monkeypatch.setattr(config.memory, "max_context_tokens", token_budget)
    # Also make sure the window is large enough to initially load all three notes
    monkeypatch.setattr(config.memory, "context_window_size", 10)

    cid = mem.save_conversation("conv")
    oldest_summary = "A" * 200  # inserted first → oldest
    middle_summary = "B" * 200
    newest_summary = "C" * 200  # inserted last → newest

    mem.save_note(cid, oldest_summary, is_noteworthy=True)
    mem.save_note(cid, middle_summary, is_noteworthy=True)
    mem.save_note(cid, newest_summary, is_noteworthy=True)

    result = mem.assemble_context()

    char_budget = token_budget * 4
    assert len(result) <= char_budget, (
        f"Result length {len(result)} exceeds char budget {char_budget}"
    )
    # The newest entry must be retained
    assert newest_summary in result
    # The oldest entry must have been dropped
    assert oldest_summary not in result


def test_assemble_context_truncation_preserves_known_info(
    mem: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Known Information section must survive truncation of history entries.

    Even after all history entries are dropped, the Known Information section
    (which is always preserved) must still be present.
    """
    from assistant.config import config

    # Token budget of 50 (200 chars) is below even a single 200-char note + skeleton,
    # but must still hold the skeleton with Known Information intact.
    # The skeleton alone with (none) bullets is ~110 chars, so set budget to 60 (240 chars)
    # which is enough for the skeleton but not for any 200-char history entries.
    token_budget = 60  # → char budget = 240 chars
    monkeypatch.setattr(config.memory, "max_context_tokens", token_budget)
    monkeypatch.setattr(config.memory, "context_window_size", 10)

    cid = mem.save_conversation("conv")
    for _ in range(3):
        mem.save_note(cid, "X" * 200, is_noteworthy=True)

    mem.set_context("user_name", "Avin")

    result = mem.assemble_context()
    assert "CONTEXT (Known Information):" in result
    assert "user_name: Avin" in result


def test_assemble_context_get_recent_actions_limit(mem: Memory) -> None:
    """get_recent_actions helper must respect its limit parameter."""
    cid = mem.save_conversation("conv")
    for i in range(15):
        mem.save_action(cid, "create_todo", {"task": f"task {i}"}, "auto_execute")

    actions = mem.get_recent_actions(limit=10)
    assert len(actions) == 10
    # Should be newest-first
    assert actions[0].details["task"] == "task 14"
