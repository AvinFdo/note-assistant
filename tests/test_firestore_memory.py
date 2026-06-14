"""Offline tests for FirestoreMemory using an in-memory fake Firestore client.

The fake implements only the subset of the Firestore API that FirestoreMemory
uses (nested collection/document, set with merge, get, stream), so no real
Firestore connection is ever made and the suite stays fast and offline.
"""

from __future__ import annotations

import pytest

from assistant.firestore_memory import CONVERSATION_TTL_DAYS, FirestoreMemory
from assistant.memory import Action, Conversation, Note

# ---------------------------------------------------------------------------
# Minimal in-memory fake Firestore
# ---------------------------------------------------------------------------


class _FakeSnapshot:
    def __init__(self, doc_id: str, data: dict | None) -> None:
        self.id = doc_id
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict | None:
        return dict(self._data) if self._data is not None else None


class _FakeDocRef:
    def __init__(self) -> None:
        self._data: dict | None = None
        self._subcollections: dict[str, _FakeCollection] = {}
        self._id = ""

    def collection(self, name: str) -> _FakeCollection:
        return self._subcollections.setdefault(name, _FakeCollection())

    def set(self, data: dict, merge: bool = False) -> None:
        if merge and self._data is not None:
            self._data.update(data)
        else:
            self._data = dict(data)

    def update(self, data: dict) -> None:
        if self._data is None:
            self._data = {}
        self._data.update(data)

    def get(self) -> _FakeSnapshot:
        return _FakeSnapshot(self._id, self._data)


class _FakeCollection:
    def __init__(self) -> None:
        self._docs: dict[str, _FakeDocRef] = {}

    def document(self, doc_id: str) -> _FakeDocRef:
        ref = self._docs.get(doc_id)
        if ref is None:
            ref = _FakeDocRef()
            ref._id = doc_id
            self._docs[doc_id] = ref
        return ref

    def stream(self):
        for doc_id, ref in self._docs.items():
            if ref._data is not None:
                yield _FakeSnapshot(doc_id, ref._data)


class _FakeFirestore:
    def __init__(self) -> None:
        self._collections: dict[str, _FakeCollection] = {}
        self.closed = False

    def collection(self, name: str) -> _FakeCollection:
        return self._collections.setdefault(name, _FakeCollection())

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def mem() -> FirestoreMemory:
    return FirestoreMemory(client=_FakeFirestore(), user_id="test-user")


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


def test_save_and_get_conversation(mem):
    cid = mem.save_conversation("hello world transcript", audio_path="/tmp/a.wav")
    conv = mem.get_conversation(cid)
    assert isinstance(conv, Conversation)
    assert conv.id == cid
    assert conv.transcript == "hello world transcript"
    assert conv.audio_path == "/tmp/a.wav"
    assert conv.started_at and conv.created_at


def test_get_conversation_missing_returns_none(mem):
    assert mem.get_conversation("nope") is None


def test_save_conversation_writes_ttl_expiry(mem):
    from datetime import datetime

    cid = mem.save_conversation("t")
    # Reach into the fake to read the stored doc.
    raw = mem._conversations.document(cid).get().to_dict()
    assert "expires_at" in raw
    created = datetime.fromisoformat(raw["created_at"])
    expires = datetime.fromisoformat(raw["expires_at"])
    assert (expires - created).days == CONVERSATION_TTL_DAYS


def test_recent_conversations_newest_first_and_limit(mem):
    ids = [mem.save_conversation(f"c{i}") for i in range(5)]
    recent = mem.get_recent_conversations(limit=3)
    assert len(recent) == 3
    assert recent[0].id == ids[-1]  # newest first


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------


def test_save_and_get_notes(mem):
    cid = mem.save_conversation("c")
    nid = mem.save_note(cid, "a useful summary", is_noteworthy=True)
    notes = mem.get_recent_notes()
    assert len(notes) == 1
    assert isinstance(notes[0], Note)
    assert notes[0].id == nid
    assert notes[0].is_noteworthy is True
    assert notes[0].conversation_id == cid


def test_search_notes_substring(mem):
    cid = mem.save_conversation("c")
    mem.save_note(cid, "Budget review for Q3")
    mem.save_note(cid, "Lunch with Sam")
    hits = mem.search_notes("budget")
    assert len(hits) == 1
    assert "Budget" in hits[0].summary
    assert mem.search_notes("nonexistent") == []


def test_notes_for_conversation_and_count(mem):
    c1 = mem.save_conversation("c1")
    c2 = mem.save_conversation("c2")
    mem.save_note(c1, "n1")
    mem.save_note(c1, "n2")
    mem.save_note(c2, "n3")
    assert mem.count_notes() == 3
    assert len(mem.get_notes_for_conversation(c1)) == 2


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def test_save_action_dict_and_roundtrip(mem):
    cid = mem.save_conversation("c")
    aid = mem.save_action(cid, "create_todo", {"task": "buy milk"}, "auto_execute")
    action = mem.get_action(aid)
    assert isinstance(action, Action)
    assert action.intent == "create_todo"
    assert action.details == {"task": "buy milk"}
    assert action.status == "pending"
    assert action.execution_mode == "auto_execute"


def test_save_action_json_string_details(mem):
    cid = mem.save_conversation("c")
    aid = mem.save_action(cid, "send_email", '{"to": "a@b.com"}', "confirm_first")
    assert mem.get_action(aid).details == {"to": "a@b.com"}


def test_update_action_status_executed_sets_executed_at(mem):
    cid = mem.save_conversation("c")
    aid = mem.save_action(cid, "create_todo", {"task": "x"}, "auto_execute")
    mem.update_action_status(aid, "executed")
    action = mem.get_action(aid)
    assert action.status == "executed"
    assert action.executed_at is not None


def test_pending_actions_filter(mem):
    cid = mem.save_conversation("c")
    a1 = mem.save_action(cid, "create_todo", {"task": "x"}, "auto_execute")
    a2 = mem.save_action(cid, "create_todo", {"task": "y"}, "auto_execute")
    mem.update_action_status(a1, "executed")
    pending = mem.get_pending_actions()
    assert [a.id for a in pending] == [a2]


def test_get_recent_actions_and_for_conversation(mem):
    c1 = mem.save_conversation("c1")
    c2 = mem.save_conversation("c2")
    mem.save_action(c1, "create_todo", {"task": "1"}, "auto_execute")
    mem.save_action(c2, "research_topic", {"topic": "2"}, "auto_execute")
    assert len(mem.get_recent_actions(limit=10)) == 2
    assert len(mem.get_actions_for_conversation(c1)) == 1


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------


def test_context_set_get_upsert(mem):
    mem.set_context("user_name", "Avin")
    assert mem.get_context("user_name") == "Avin"
    mem.set_context("user_name", "Avin F")  # upsert
    assert mem.get_context("user_name") == "Avin F"
    assert mem.get_context("missing") is None


def test_get_all_context(mem):
    mem.set_context("a", "1")
    mem.set_context("b", "2")
    assert mem.get_all_context() == {"a": "1", "b": "2"}


# ---------------------------------------------------------------------------
# assemble_context (must match SQLite Memory format)
# ---------------------------------------------------------------------------


def test_assemble_context_sections(mem):
    cid = mem.save_conversation("c")
    mem.save_note(cid, "Discussed roadmap", is_noteworthy=True)
    mem.save_action(cid, "create_todo", {"task": "follow up"}, "auto_execute")
    mem.set_context("user_name", "Avin")

    ctx = mem.assemble_context()
    assert "CONTEXT (Recent History):" in ctx
    assert "Discussed roadmap" in ctx
    assert "CONTEXT (Known Information):" in ctx
    assert "user_name: Avin" in ctx
    assert "CONTEXT (Pending Actions):" in ctx
    assert "create_todo: follow up" in ctx


def test_assemble_context_empty_db(mem):
    ctx = mem.assemble_context()
    assert "- (none)" in ctx


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_calls_client_close():
    fake = _FakeFirestore()
    m = FirestoreMemory(client=fake, user_id="u")
    m.close()
    assert fake.closed is True


# ---------------------------------------------------------------------------
# Scored-retrieval fields (2.3.2): importance + embedding
# ---------------------------------------------------------------------------


def test_save_note_roundtrips_importance_and_embedding(mem):
    cid = mem.save_conversation("transcript")
    mem.save_note(cid, "summary", is_noteworthy=True, importance=0.8, embedding=[0.1, 0.2, 0.3])
    note = mem.get_recent_notes(limit=1)[0]
    assert note.importance == 0.8
    assert note.embedding == [0.1, 0.2, 0.3]


def test_get_notes_without_embedding(mem):
    cid = mem.save_conversation("transcript")
    mem.save_note(cid, "no vector", embedding=None)
    mem.save_note(cid, "has vector", embedding=[1.0, 2.0])
    summaries = [n.summary for n in mem.get_notes_without_embedding()]
    assert "no vector" in summaries
    assert "has vector" not in summaries


def test_update_note_embedding(mem):
    cid = mem.save_conversation("transcript")
    nid = mem.save_note(cid, "summary")
    mem.update_note_embedding(nid, [0.5, 0.6, 0.7])
    note = mem.get_recent_notes(limit=1)[0]
    assert note.embedding == [0.5, 0.6, 0.7]
    assert mem.get_notes_without_embedding() == []
