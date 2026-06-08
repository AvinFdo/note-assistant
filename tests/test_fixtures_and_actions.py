"""Tests exercising the shared conftest fixtures and the concrete mock actions.

These complement the per-feature suites: they verify the reusable fixtures
(`memory`, `test_config`, `mock_genai_client`) work as intended and cover the
``execute``/``describe`` bodies of every concrete Action so coverage of the
``actions/`` package stays comfortably above the 80% target.
"""

from __future__ import annotations

from assistant.actions.calendar import AddCalendarAction
from assistant.actions.email import SendEmailAction
from assistant.actions.research import ResearchTopicAction
from assistant.actions.todo import CreateTodoAction
from assistant.brain import Brain

# ---------------------------------------------------------------------------
# Fixture sanity checks
# ---------------------------------------------------------------------------


def test_memory_fixture_is_usable_and_isolated(memory):
    cid = memory.save_conversation("hello there general")
    assert memory.get_conversation(cid) is not None
    # Fresh DB — nothing else leaked in.
    assert len(memory.get_recent_conversations()) == 1


def test_test_config_fixture_loads_values(test_config):
    assert test_config.gcp.project_id == "test-project"
    assert test_config.memory.min_transcript_words == 10
    assert test_config.actions.send_email.mode == "confirm_first"


def test_mock_genai_client_drives_brain(memory, mock_genai_client):
    client = mock_genai_client(
        {"is_noteworthy": True, "summary_note": "Met with Sam about Q3 roadmap.", "actions": []}
    )
    brain = Brain(memory, client=client)

    result = brain.process("we discussed the third quarter roadmap with sam in detail today")

    assert result.is_noteworthy is True
    assert "roadmap" in result.summary_note.lower()
    notes = memory.get_recent_notes()
    assert len(notes) == 1


# ---------------------------------------------------------------------------
# Concrete action execute/describe coverage
# ---------------------------------------------------------------------------


def test_create_todo_execute_and_describe(capsys):
    action = CreateTodoAction()
    assert action.execute({"task": "buy milk"}) == "[TODO] buy milk"
    assert "[TODO] buy milk" in capsys.readouterr().out
    assert "buy milk" in action.describe({"task": "buy milk"})
    assert "no task" in action.describe({})


def test_send_email_execute_and_describe(capsys):
    action = SendEmailAction()
    msg = action.execute({"recipient": "a@b.com", "subject": "Hi"})
    assert "a@b.com" in msg and "Hi" in msg
    assert "[EMAIL]" in capsys.readouterr().out
    assert "a@b.com" in action.describe({"recipient": "a@b.com", "subject": "Hi"})
    assert "no recipient" in action.describe({})


def test_add_calendar_execute_and_describe(capsys):
    action = AddCalendarAction()
    msg = action.execute({"title": "Standup", "time": "9am"})
    assert "Standup" in msg and "9am" in msg
    assert "[CALENDAR]" in capsys.readouterr().out
    assert "Standup" in action.describe({"title": "Standup", "time": "9am"})
    assert "no title" in action.describe({})


def test_research_topic_execute_and_describe(capsys):
    action = ResearchTopicAction()
    assert action.execute({"topic": "rust async"}) == "[RESEARCH] rust async"
    assert "[RESEARCH] rust async" in capsys.readouterr().out
    assert "rust async" in action.describe({"topic": "rust async"})
    assert "no topic" in action.describe({})
