"""Tests for the action framework: registry, routing, and guardrails (task 1.6.1).

All tests use in-memory SQLite and stub callables — no live APIs, no real genai calls.
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
