"""Tests for the action framework: registry, routing, and guardrails (tasks 1.6.1 & 1.6.2).

All tests use in-memory SQLite and stub callables — no live APIs, no real genai calls.

Task 1.6.2 additions
--------------------
- Status persistence: route_action updates the action row's status in SQLite when
  action_id is supplied.
- Confirmed status transitions for each mode:
    auto_execute  → "executed"
    confirm_first + True  → "executed"
    confirm_first + False → "dismissed"
    confirm_first + None  → "pending" (awaiting — no DB update)
    log_only              → "logged"
- Each action's execute() prints the documented prefix to stdout.
"""

from __future__ import annotations

import pytest

from assistant.actions import (
    ACTION_REGISTRY,
    UnknownActionError,
    route_action,
)
from assistant.actions.calendar import AddCalendarAction
from assistant.actions.email import SendEmailAction
from assistant.actions.research import ResearchTopicAction
from assistant.actions.todo import CreateTodoAction
from assistant.brain import ActionItem
from assistant.memory import Memory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory() -> Memory:
    """In-memory SQLite instance — no files created, no state shared between tests."""
    return Memory(db_path=":memory:")


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_registry_has_all_four_intents() -> None:
    """ACTION_REGISTRY must contain exactly the four known intents."""
    assert "create_todo" in ACTION_REGISTRY
    assert "send_email" in ACTION_REGISTRY
    assert "add_calendar" in ACTION_REGISTRY
    assert "research_topic" in ACTION_REGISTRY


def test_registry_maps_to_correct_classes() -> None:
    """Each registry entry must be an instance of the correct Action subclass."""
    assert isinstance(ACTION_REGISTRY["create_todo"], CreateTodoAction)
    assert isinstance(ACTION_REGISTRY["send_email"], SendEmailAction)
    assert isinstance(ACTION_REGISTRY["add_calendar"], AddCalendarAction)
    assert isinstance(ACTION_REGISTRY["research_topic"], ResearchTopicAction)


# ---------------------------------------------------------------------------
# Routing — auto_execute (create_todo default)
# ---------------------------------------------------------------------------


def test_route_create_todo_auto_execute(memory: Memory) -> None:
    """create_todo in auto_execute mode must call execute() and return its message."""
    item = ActionItem(intent="create_todo", confidence=0.9, details={"task": "Buy milk"})
    result = route_action(item, memory)
    assert result == "[TODO] Buy milk"


def test_route_research_topic_auto_execute(memory: Memory) -> None:
    """research_topic in auto_execute mode must dispatch to ResearchTopicAction.execute()."""
    item = ActionItem(
        intent="research_topic", confidence=0.85, details={"topic": "quantum computing"}
    )
    result = route_action(item, memory)
    assert result == "[RESEARCH] quantum computing"


# ---------------------------------------------------------------------------
# Routing — unknown intent raises UnknownActionError
# ---------------------------------------------------------------------------


def test_unknown_intent_raises(memory: Memory) -> None:
    """An intent not in the registry must raise UnknownActionError."""
    item = ActionItem(intent="fly_to_moon", confidence=0.99, details={})
    with pytest.raises(UnknownActionError, match="fly_to_moon"):
        route_action(item, memory)


# ---------------------------------------------------------------------------
# Routing — confirm_first (send_email)
# ---------------------------------------------------------------------------


def test_send_email_confirm_none_not_executed(memory: Memory) -> None:
    """send_email with confirm=None must NOT execute — guardrail against auto-send."""
    item = ActionItem(
        intent="send_email",
        confidence=0.95,
        details={"recipient": "boss@example.com", "subject": "Report"},
    )
    result = route_action(item, memory, confirm=None)
    # Must not contain the email execution marker
    assert "[EMAIL]" not in result
    # Must signal that execution did not happen
    assert "not executed" in result.lower() or "awaiting" in result.lower()


def test_send_email_confirm_true_executes(memory: Memory) -> None:
    """send_email with confirm=lambda: True must execute and return the email message."""
    item = ActionItem(
        intent="send_email",
        confidence=0.95,
        details={"recipient": "friend@example.com", "subject": "Hello"},
    )
    result = route_action(item, memory, confirm=lambda _desc: True)
    assert result == "[EMAIL] To: friend@example.com, Subject: Hello"


def test_send_email_confirm_false_not_executed(memory: Memory) -> None:
    """send_email with confirm=lambda: False must NOT execute — user dismissed it."""
    item = ActionItem(
        intent="send_email",
        confidence=0.95,
        details={"recipient": "x@example.com", "subject": "Nope"},
    )
    result = route_action(item, memory, confirm=lambda _desc: False)
    assert "[EMAIL]" not in result
    assert "dismissed" in result.lower() or "not confirmed" in result.lower()


# ---------------------------------------------------------------------------
# Routing — log_only mode (monkeypatched)
# ---------------------------------------------------------------------------


def test_log_only_not_executed(memory: Memory, monkeypatch: pytest.MonkeyPatch) -> None:
    """When config mode is log_only, the action must not execute."""
    import assistant.actions as actions_pkg

    # Temporarily override create_todo mode to log_only
    monkeypatch.setattr(
        actions_pkg.config.actions.create_todo,
        "mode",
        "log_only",
    )

    item = ActionItem(intent="create_todo", confidence=0.9, details={"task": "Secret task"})
    result = route_action(item, memory)

    assert "[TODO]" not in result
    assert "log" in result.lower()


# ---------------------------------------------------------------------------
# 1.6.2 — SQLite status persistence via action_id
# ---------------------------------------------------------------------------


def _save_action(memory: Memory, intent: str, details: dict, mode: str) -> str:
    """Helper: create a conversation then save an action row; return the action_id."""
    cid = memory.save_conversation("test transcript")
    return memory.save_action(cid, intent, details, mode)


def _get_action_status(memory: Memory, action_id: str) -> str:
    """Helper: retrieve a single action's status from memory by action_id."""
    actions = memory.get_recent_actions(limit=50)
    for action in actions:
        if action.id == action_id:
            return action.status
    raise AssertionError(f"action {action_id!r} not found in recent actions")


# --- auto_execute + action_id → "executed" ---


def test_auto_execute_sets_status_executed(memory: Memory, capsys: pytest.CaptureFixture) -> None:
    """auto_execute with action_id must set status 'executed' in SQLite and print [TODO]."""
    action_id = _save_action(memory, "create_todo", {"task": "Buy milk"}, "auto_execute")
    item = ActionItem(intent="create_todo", confidence=0.9, details={"task": "Buy milk"})

    result = route_action(item, memory, action_id=action_id)

    # Status persisted
    assert _get_action_status(memory, action_id) == "executed"
    # Execute was called and returned the right message
    assert result == "[TODO] Buy milk"
    # print() was called
    captured = capsys.readouterr()
    assert "[TODO]" in captured.out


# --- confirm_first + True → "executed" ---


def test_confirm_true_sets_status_executed(memory: Memory, capsys: pytest.CaptureFixture) -> None:
    """confirm_first + confirm=True must execute the action and set status 'executed'."""
    action_id = _save_action(
        memory,
        "send_email",
        {"recipient": "friend@example.com", "subject": "Hello"},
        "confirm_first",
    )
    item = ActionItem(
        intent="send_email",
        confidence=0.95,
        details={"recipient": "friend@example.com", "subject": "Hello"},
    )

    result = route_action(item, memory, action_id=action_id, confirm=lambda _desc: True)

    assert _get_action_status(memory, action_id) == "executed"
    assert result == "[EMAIL] To: friend@example.com, Subject: Hello"
    captured = capsys.readouterr()
    assert "[EMAIL]" in captured.out


# --- confirm_first + False → "dismissed" (no execution) ---


def test_confirm_false_sets_status_dismissed(memory: Memory, capsys: pytest.CaptureFixture) -> None:
    """confirm_first + confirm=False must NOT execute the action and set status 'dismissed'."""
    action_id = _save_action(
        memory,
        "send_email",
        {"recipient": "x@example.com", "subject": "Nope"},
        "confirm_first",
    )
    item = ActionItem(
        intent="send_email",
        confidence=0.95,
        details={"recipient": "x@example.com", "subject": "Nope"},
    )

    result = route_action(item, memory, action_id=action_id, confirm=lambda _desc: False)

    assert _get_action_status(memory, action_id) == "dismissed"
    # No email output — action was NOT executed
    captured = capsys.readouterr()
    assert "[EMAIL]" not in captured.out
    assert "dismissed" in result.lower() or "not confirmed" in result.lower()


# --- confirm_first + None → status stays "pending" (awaiting) ---


def test_confirm_none_status_remains_pending(memory: Memory, capsys: pytest.CaptureFixture) -> None:
    """confirm_first + confirm=None must NOT execute; status must remain 'pending'."""
    action_id = _save_action(
        memory,
        "send_email",
        {"recipient": "boss@example.com", "subject": "Report"},
        "confirm_first",
    )
    item = ActionItem(
        intent="send_email",
        confidence=0.95,
        details={"recipient": "boss@example.com", "subject": "Report"},
    )

    result = route_action(item, memory, action_id=action_id, confirm=None)

    # Status must NOT have been changed — remains "pending" (awaiting user input)
    assert _get_action_status(memory, action_id) == "pending"
    # No email output
    captured = capsys.readouterr()
    assert "[EMAIL]" not in captured.out
    assert "not executed" in result.lower() or "awaiting" in result.lower()


# --- log_only → "logged" ---


def test_log_only_sets_status_logged(
    memory: Memory,
    capsys: pytest.CaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """log_only mode with action_id must set status 'logged' and NOT execute the action."""
    import assistant.actions as actions_pkg

    monkeypatch.setattr(actions_pkg.config.actions.create_todo, "mode", "log_only")

    action_id = _save_action(memory, "create_todo", {"task": "Secret task"}, "log_only")
    item = ActionItem(intent="create_todo", confidence=0.9, details={"task": "Secret task"})

    result = route_action(item, memory, action_id=action_id)

    assert _get_action_status(memory, action_id) == "logged"
    captured = capsys.readouterr()
    assert "[TODO]" not in captured.out
    assert "log" in result.lower()


# ---------------------------------------------------------------------------
# 1.6.2 — Each action's execute() prints the correct prefix
# ---------------------------------------------------------------------------


def test_todo_execute_prints_prefix(capsys: pytest.CaptureFixture) -> None:
    """CreateTodoAction.execute() must print a line containing '[TODO]'."""
    from assistant.actions.todo import CreateTodoAction

    action = CreateTodoAction()
    result = action.execute({"task": "Write tests"})

    captured = capsys.readouterr()
    assert "[TODO]" in captured.out
    assert "[TODO]" in result


def test_email_execute_prints_prefix(capsys: pytest.CaptureFixture) -> None:
    """SendEmailAction.execute() must print a line containing '[EMAIL]'."""
    from assistant.actions.email import SendEmailAction

    action = SendEmailAction()
    result = action.execute({"recipient": "a@b.com", "subject": "Hi"})

    captured = capsys.readouterr()
    assert "[EMAIL]" in captured.out
    assert "[EMAIL]" in result


def test_calendar_execute_prints_prefix(capsys: pytest.CaptureFixture) -> None:
    """AddCalendarAction.execute() must print a line containing '[CALENDAR]'."""
    from assistant.actions.calendar import AddCalendarAction

    action = AddCalendarAction()
    result = action.execute({"title": "Team sync", "time": "3pm"})

    captured = capsys.readouterr()
    assert "[CALENDAR]" in captured.out
    assert "[CALENDAR]" in result


def test_research_execute_prints_prefix(capsys: pytest.CaptureFixture) -> None:
    """ResearchTopicAction.execute() must print a line containing '[RESEARCH]'."""
    from assistant.actions.research import ResearchTopicAction

    action = ResearchTopicAction()
    result = action.execute({"topic": "quantum computing"})

    captured = capsys.readouterr()
    assert "[RESEARCH]" in captured.out
    assert "[RESEARCH]" in result


# ---------------------------------------------------------------------------
# 1.6.2 — action_id=None (default) — no DB write, backward-compatible
# ---------------------------------------------------------------------------


def test_auto_execute_no_action_id_still_works(memory: Memory) -> None:
    """route_action with action_id=None (default) must work without touching the DB."""
    item = ActionItem(intent="create_todo", confidence=0.9, details={"task": "No DB"})
    result = route_action(item, memory)  # action_id defaults to None
    assert result == "[TODO] No DB"
