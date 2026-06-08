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
    result = brain.process("I need to buy groceries today when I finish work this evening.")

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
    assert convs[0].transcript == "I need to buy groceries today when I finish work this evening."

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
    result = brain.process("Maybe I should send an email to Alice about that project update.")

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
    """When is_noteworthy=False, conversation is persisted but no note or actions are created."""
    action_payload = {
        "intent": "create_todo",
        "confidence": 0.9,
        "details": {"task": "irrelevant"},
    }
    client = _make_mock_client(
        _make_response(is_noteworthy=False, summary="", actions=[action_payload])
    )

    brain = Brain(memory=mem, client=client)
    result = brain.process("Hey, how's it going today? Anything interesting happen recently?")

    assert result.is_noteworthy is False

    # Conversation must be saved
    convs = mem.get_recent_conversations()
    assert len(convs) == 1

    # No note
    notes = mem.get_recent_notes()
    assert notes == []

    # No actions persisted when not noteworthy
    all_actions = mem.get_recent_actions(limit=10)
    assert all_actions == []


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
    brain.process("What did we discuss earlier about the quarterly report work?")

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
        brain.process("Some transcript that has enough words to pass the length filter.")


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
    brain.process("Please send Bob an email about the upcoming meeting this Thursday afternoon.")

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
    brain.process("I need to review that pull request before the end of today.")

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
    brain.process("I should call the dentist office tomorrow and book an appointment soon.")

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


# ---------------------------------------------------------------------------
# Relevance filter: length pre-filter (< min_transcript_words)
# ---------------------------------------------------------------------------


def test_short_transcript_skips_llm(mem: Memory) -> None:
    """Transcript with fewer words than min_transcript_words never calls generate_content.

    Verifies:
    - LLM client is NOT called
    - conversation IS saved for context continuity
    - no note created
    - no actions persisted
    - returns ProcessingResult(is_noteworthy=False)
    """
    client = _make_mock_client(_make_response(is_noteworthy=True, summary="Should not appear"))

    brain = Brain(memory=mem, client=client)
    result = brain.process("Sure thing.")  # 2 words — below threshold of 10

    # LLM must NOT have been called
    client.models.generate_content.assert_not_called()

    # Result must signal not noteworthy
    assert result.is_noteworthy is False
    assert result.summary_note == ""
    assert result.actions == []

    # Conversation must be saved for context continuity
    convs = mem.get_recent_conversations()
    assert len(convs) == 1
    assert convs[0].transcript == "Sure thing."

    # No note or actions created
    assert mem.get_recent_notes() == []
    assert mem.get_recent_actions(limit=10) == []


# ---------------------------------------------------------------------------
# Relevance filter: is_noteworthy=False from LLM — no note, no actions persisted
# ---------------------------------------------------------------------------


def test_llm_not_noteworthy_no_actions_persisted(mem: Memory) -> None:
    """Long transcript where LLM returns is_noteworthy=False: actions are NOT persisted."""
    action_payload = {
        "intent": "create_todo",
        "confidence": 0.95,
        "details": {"task": "follow up on report"},
    }
    client = _make_mock_client(
        _make_response(is_noteworthy=False, summary="", actions=[action_payload])
    )

    brain = Brain(memory=mem, client=client)
    result = brain.process(
        "I was thinking about maybe following up on that report we discussed yesterday afternoon."
    )

    assert result.is_noteworthy is False

    # Conversation is saved
    convs = mem.get_recent_conversations()
    assert len(convs) == 1

    # No note
    assert mem.get_recent_notes() == []

    # No actions persisted when not noteworthy
    assert mem.get_pending_actions() == []
    assert mem.get_recent_actions(limit=10) == []


# ---------------------------------------------------------------------------
# Relevance filter: config threshold respected (monkeypatch)
# ---------------------------------------------------------------------------


def test_min_transcript_words_threshold_from_config(
    mem: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Threshold is read from config.memory.min_transcript_words.

    A transcript at exactly the threshold (10 words) passes; one word below (9) is filtered.
    Monkeypatches the config value to verify the code reads it dynamically.
    """
    import assistant.brain as brain_mod
    from assistant.config import MemoryConfig

    # Build a new config with threshold=5 and monkeypatch it
    new_memory_cfg = MemoryConfig(
        db_path=":memory:",
        context_window_size=5,
        max_context_tokens=4000,
        min_transcript_words=5,
    )
    monkeypatch.setattr(brain_mod.config, "memory", new_memory_cfg)

    # 5-word transcript — exactly at threshold → should call LLM
    client_above = _make_mock_client(_make_response(is_noteworthy=True, summary="Pass"))
    brain_above = Brain(memory=mem, client=client_above)
    brain_above.process("One two three four five")  # 5 words
    client_above.models.generate_content.assert_called_once()

    # 4-word transcript — below threshold → must NOT call LLM
    client_below = _make_mock_client(
        _make_response(is_noteworthy=True, summary="Should not appear")
    )
    brain_below = Brain(memory=mem, client=client_below)
    brain_below.process("One two three four")  # 4 words
    client_below.models.generate_content.assert_not_called()
