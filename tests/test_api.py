"""Tests for the Avin FastAPI application (task 2.1.1).

Uses ``fastapi.testclient.TestClient`` with the ``get_memory`` dependency
overridden to supply a seeded in-memory :class:`~assistant.memory.Memory`
instance.  No live API calls are made; all tests are fully offline.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from assistant.api.app import app
from assistant.api.routes import get_memory
from assistant.memory import Memory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mem():
    """In-memory Memory instance seeded with one conversation, note, and two actions."""
    m = Memory(db_path=":memory:")

    # Seed a conversation
    conv_id = m.save_conversation(transcript="Discussed the Q3 budget review.")

    # Seed a note
    m.save_note(conversation_id=conv_id, summary="Q3 budget review discussed", is_noteworthy=True)

    # Seed two actions: one pending create_todo, one confirmed send_email
    m.save_action(
        conversation_id=conv_id,
        intent="create_todo",
        details={"task": "Follow up on budget"},
        execution_mode="auto_execute",
    )
    m.save_action(
        conversation_id=conv_id,
        intent="send_email",
        details={"recipient": "boss@example.com", "subject": "Q3 Budget"},
        execution_mode="confirm_first",
    )

    # Confirm the email action
    all_actions = m.get_recent_actions(limit=100)
    email_action = next(a for a in all_actions if a.intent == "send_email")
    m.update_action_status(email_action.id, "confirmed")

    # Seed context
    m.set_context("user_name", "Avin")
    m.set_context("current_project", "budget_review")

    yield m
    m.close()


@pytest.fixture
def client(mem):
    """TestClient with the get_memory dependency overridden."""

    def override_get_memory():
        yield mem

    app.dependency_overrides[get_memory] = override_get_memory
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health probe
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /api/v1/notes
# ---------------------------------------------------------------------------


def test_list_notes_contains_seeded_note(client, mem):
    resp = client.get("/api/v1/notes")
    assert resp.status_code == 200
    data = resp.json()
    assert "notes" in data
    assert "total" in data
    assert data["total"] == 1
    assert len(data["notes"]) == 1
    assert data["notes"][0]["summary"] == "Q3 budget review discussed"


def test_list_notes_pagination_limit(client, mem):
    """Seeding extra notes and using limit=1 should return only 1 note."""
    conv_id = mem.save_conversation(transcript="Another conversation")
    mem.save_note(conversation_id=conv_id, summary="Second note", is_noteworthy=True)

    resp = client.get("/api/v1/notes?limit=1&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["notes"]) == 1
    assert data["total"] == 2


def test_list_notes_pagination_offset(client, mem):
    """Using offset=1 with 2 total notes should return the older note."""
    conv_id = mem.save_conversation(transcript="Another conversation")
    mem.save_note(conversation_id=conv_id, summary="Second note", is_noteworthy=True)

    resp = client.get("/api/v1/notes?limit=10&offset=1")
    assert resp.status_code == 200
    data = resp.json()
    # Newest first: offset=1 skips the most-recent note
    assert len(data["notes"]) == 1
    assert data["total"] == 2


# ---------------------------------------------------------------------------
# GET /api/v1/notes/{id}
# ---------------------------------------------------------------------------


def test_get_note_detail(client, mem):
    note_id = mem.get_recent_notes(limit=1)[0].id
    resp = client.get(f"/api/v1/notes/{note_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["note"]["id"] == note_id
    assert data["conversation"] is not None
    assert isinstance(data["actions"], list)
    # The conversation holds the two actions we seeded
    assert len(data["actions"]) == 2


def test_get_note_unknown_id_returns_404(client):
    resp = client.get("/api/v1/notes/nonexistent-id-xyz")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/v1/notes/{id}
# ---------------------------------------------------------------------------


def test_delete_note_removes_it(client, mem):
    note_id = mem.get_recent_notes(limit=1)[0].id
    resp = client.delete(f"/api/v1/notes/{note_id}")
    assert resp.status_code == 204
    # Gone from the store and the API.
    assert mem.count_notes() == 0
    assert client.get(f"/api/v1/notes/{note_id}").status_code == 404


def test_delete_note_unknown_id_returns_404(client):
    resp = client.delete("/api/v1/notes/nonexistent-id-xyz")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/actions
# ---------------------------------------------------------------------------


def test_list_actions_returns_all(client):
    resp = client.get("/api/v1/actions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["actions"]) == 2


def test_list_actions_filter_by_status(client):
    resp = client.get("/api/v1/actions?status=pending")
    assert resp.status_code == 200
    data = resp.json()
    assert all(a["status"] == "pending" for a in data["actions"])
    assert len(data["actions"]) == 1
    assert data["actions"][0]["intent"] == "create_todo"


def test_list_actions_filter_by_intent(client):
    resp = client.get("/api/v1/actions?intent=send_email")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["actions"]) == 1
    assert data["actions"][0]["intent"] == "send_email"


def test_list_actions_filter_by_status_and_intent(client):
    resp = client.get("/api/v1/actions?status=confirmed&intent=send_email")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["actions"]) == 1
    assert data["actions"][0]["status"] == "confirmed"
    assert data["actions"][0]["intent"] == "send_email"


# ---------------------------------------------------------------------------
# PATCH /api/v1/actions/{id}
# ---------------------------------------------------------------------------


def test_patch_action_status(client, mem):
    todo_action = next(a for a in mem.get_recent_actions(limit=100) if a.intent == "create_todo")
    resp = client.patch(f"/api/v1/actions/{todo_action.id}", json={"status": "confirmed"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"]["status"] == "confirmed"
    assert data["action"]["id"] == todo_action.id


def test_patch_action_unknown_id_returns_404(client):
    resp = client.patch("/api/v1/actions/nonexistent-id-xyz", json={"status": "confirmed"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /api/v1/search
# ---------------------------------------------------------------------------


def test_search_notes_matching(client):
    resp = client.post("/api/v1/search", json={"query": "budget"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert "budget" in data["results"][0]["summary"].lower()


def test_search_notes_no_match(client):
    resp = client.post("/api/v1/search", json={"query": "zxqwerty_nomatche"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"] == []


# ---------------------------------------------------------------------------
# GET /api/v1/context
# ---------------------------------------------------------------------------


def test_get_context(client):
    resp = client.get("/api/v1/context")
    assert resp.status_code == 200
    data = resp.json()
    assert data["context"]["user_name"] == "Avin"
    assert data["context"]["current_project"] == "budget_review"


# ---------------------------------------------------------------------------
# OpenAPI / docs
# ---------------------------------------------------------------------------


def test_openapi_json(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert "paths" in resp.json()


def test_docs_page(client):
    resp = client.get("/docs")
    assert resp.status_code == 200
