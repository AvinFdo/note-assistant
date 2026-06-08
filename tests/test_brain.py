"""Tests for assistant.brain — context-aware prompting and structured output.

All tests inject a mock genai.Client — no live API calls are made.
Uses a real in-memory Memory(:memory:) for persistence assertions.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from assistant.brain import ActionItem, Brain, BrainError, ProcessingResult
from assistant.memory import Memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(response_data: dict) -> MagicMock:
    """Return a mock genai.Client whose generate_content returns *response_data* as JSON."""
    response = MagicMock()
    response.text = json.dumps(response_data)
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def _make_response(
    is_noteworthy: bool = True,
    summary: str = "Test summary",
    actions: list[dict] | None = None,
) -> dict:
    """Build a well-formed model response dict."""
    return {
        "is_noteworthy": is_noteworthy,
        "summary_note": summary,
        "actions": actions or [],
    }


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
# Happy path: noteworthy + high-confidence action
# ---------------------------------------------------------------------------


def test_happy_path_high_confidence(mem: Memory) -> None:
    """Mock returns noteworthy result with one high-confidence action.

    Verifies:
    - correct ProcessingResult returned
    - conversation, note, and action all persisted
    - high-confidence action status == 'pending' (appears in get_pending_actions)
    """
    action_payload = {
        "intent": "create_todo",
        "confidence": 0.9,
        "details": {"task": "buy groceries"},
    }
    client = _make_mock_client(
        _make_response(
            is_noteworthy=True,
            summary="User needs to buy groceries.",
            actions=[action_payload],
        )
    )

    brain = Brain(memory=mem, client=client)
    result = brain.process("I need to buy groceries today.")

    # ProcessingResult shape
    assert isinstance(result, ProcessingResult)
    assert result.is_noteworthy is True
    assert result.summary_note == "User needs to buy groceries."
    assert len(result.actions) == 1
    action = result.actions[0]
    assert isinstance(action, ActionItem)
    assert action.intent == "create_todo"
    assert action.confidence == pytest.approx(0.9)
    assert action.details == {"task": "buy groceries"}

    # Persistence — conversation saved
    convs = mem.get_recent_conversations()
    assert len(convs) == 1
    assert convs[0].transcript == "I need to buy groceries today."

    # Persistence — note saved (noteworthy)
    notes = mem.get_recent_notes()
    assert len(notes) == 1
    assert notes[0].summary == "User needs to buy groceries."
    assert notes[0].is_noteworthy is True

    # Persistence — action is pending (high confidence)
    pending = mem.get_pending_actions()
    assert len(pending) == 1
    assert pending[0].intent == "create_todo"
    assert pending[0].status == "pending"


# ---------------------------------------------------------------------------
# Low-confidence action: saved but status == 'low_confidence'
# ---------------------------------------------------------------------------


def test_low_confidence_action_not_pending(mem: Memory) -> None:
    """Action with confidence < 0.7 is saved but marked 'low_confidence', not 'pending'."""
    action_payload = {
        "intent": "send_email",
        "confidence": 0.4,
        "details": {"recipient": "alice@example.com", "subject": "Hi"},
    }
    client = _make_mock_client(
        _make_response(
            is_noteworthy=True,
            summary="User might want to email Alice.",
            actions=[action_payload],
        )
    )

    brain = Brain(memory=mem, client=client)
    result = brain.process("Maybe I should email Alice about that.")

    # Action appears in result
    assert len(result.actions) == 1
    assert result.actions[0].confidence == pytest.approx(0.4)

    # NOT in pending actions
    pending = mem.get_pending_actions()
    assert pending == []

    # But is saved with low_confidence status
    all_actions = mem.get_recent_actions(limit=10)
    assert len(all_actions) == 1
    assert all_actions[0].status == "low_confidence"
    assert all_actions[0].intent == "send_email"


# ---------------------------------------------------------------------------
# is_noteworthy=false: conversation saved, no note
# ---------------------------------------------------------------------------


def test_not_noteworthy_no_note_saved(mem: Memory) -> None:
    """When is_noteworthy=False, conversation is persisted but no note is created."""
    client = _make_mock_client(_make_response(is_noteworthy=False, summary="", actions=[]))

    brain = Brain(memory=mem, client=client)
    result = brain.process("Hey, how's it going?")

    assert result.is_noteworthy is False

    # Conversation must be saved
    convs = mem.get_recent_conversations()
    assert len(convs) == 1

    # No note
    notes = mem.get_recent_notes()
    assert notes == []


# ---------------------------------------------------------------------------
# Context influence: prior note appears in prompt
# ---------------------------------------------------------------------------


def test_context_influence_prior_note_in_prompt(mem: Memory) -> None:
    """Prior noteworthy note must appear in the prompt passed to generate_content."""
    prior_note_text = "User is working on the quarterly report."

    # Seed memory with a prior conversation and noteworthy note
    cid = mem.save_conversation("We should wrap up the quarterly report.")
    mem.save_note(cid, prior_note_text, is_noteworthy=True)

    client = _make_mock_client(_make_response(is_noteworthy=False, summary="", actions=[]))

    brain = Brain(memory=mem, client=client)
    brain.process("What did we discuss earlier?")

    # Capture the prompt passed to generate_content
    call_args = client.models.generate_content.call_args
    contents = call_args.kwargs.get("contents") or call_args.args[1]
    prompt_text = " ".join(str(c) for c in contents)

    assert prior_note_text in prompt_text, (
        f"Expected prior note text '{prior_note_text}' in prompt, got: {prompt_text!r}"
    )


# ---------------------------------------------------------------------------
# Malformed JSON → BrainError
# ---------------------------------------------------------------------------


def test_malformed_json_raises_brain_error(mem: Memory) -> None:
    """When the model returns invalid JSON, Brain must raise BrainError."""
    response = MagicMock()
    response.text = "This is definitely not JSON {{{"
    client = MagicMock()
    client.models.generate_content.return_value = response

    brain = Brain(memory=mem, client=client)

    with pytest.raises(BrainError, match="malformed JSON"):
        brain.process("Some transcript.")


# ---------------------------------------------------------------------------
# execution_mode read from config per intent
# ---------------------------------------------------------------------------


def test_execution_mode_send_email_is_confirm_first(mem: Memory) -> None:
    """send_email action must use 'confirm_first' execution_mode from config."""
    action_payload = {
        "intent": "send_email",
        "confidence": 0.95,
        "details": {"recipient": "bob@example.com", "subject": "Meeting"},
    }
    client = _make_mock_client(
        _make_response(
            is_noteworthy=True,
            summary="User wants to email Bob about a meeting.",
            actions=[action_payload],
        )
    )

    brain = Brain(memory=mem, client=client)
    brain.process("Send Bob an email about the meeting.")

    all_actions = mem.get_recent_actions(limit=10)
    assert len(all_actions) == 1
    assert all_actions[0].intent == "send_email"
    assert all_actions[0].execution_mode == "confirm_first"


def test_execution_mode_create_todo_is_auto_execute(mem: Memory) -> None:
    """create_todo action must use 'auto_execute' execution_mode from config."""
    action_payload = {
        "intent": "create_todo",
        "confidence": 0.85,
        "details": {"task": "review PR"},
    }
    client = _make_mock_client(
        _make_response(
            is_noteworthy=True,
            summary="User needs to review a PR.",
            actions=[action_payload],
        )
    )

    brain = Brain(memory=mem, client=client)
    brain.process("I need to review that pull request.")

    all_actions = mem.get_recent_actions(limit=10)
    assert len(all_actions) == 1
    assert all_actions[0].execution_mode == "auto_execute"


# ---------------------------------------------------------------------------
# Mixed confidence: one high, one low in same response
# ---------------------------------------------------------------------------


def test_mixed_confidence_correct_statuses(mem: Memory) -> None:
    """One action above threshold (pending), one below (low_confidence) in same call."""
    actions_payload = [
        {
            "intent": "create_todo",
            "confidence": 0.85,
            "details": {"task": "call dentist"},
        },
        {
            "intent": "add_calendar",
            "confidence": 0.3,
            "details": {"title": "dentist appointment"},
        },
    ]
    client = _make_mock_client(
        _make_response(
            is_noteworthy=True,
            summary="User needs to call dentist and maybe schedule appointment.",
            actions=actions_payload,
        )
    )

    brain = Brain(memory=mem, client=client)
    brain.process("I should call the dentist and book an appointment.")

    pending = mem.get_pending_actions()
    all_actions = mem.get_recent_actions(limit=10)

    assert len(all_actions) == 2
    # Only the high-confidence one is pending
    assert len(pending) == 1
    assert pending[0].intent == "create_todo"

    # The low-confidence one has low_confidence status
    low_conf = [a for a in all_actions if a.intent == "add_calendar"]
    assert len(low_conf) == 1
    assert low_conf[0].status == "low_confidence"
