"""FirestoreMemory: a Firestore-backed implementation of the Memory interface.

Mirrors the public surface of :class:`assistant.memory.Memory` exactly (same
method names, signatures, and dataclass return types) so the storage backend can
be swapped via ``config.memory.backend`` with no change to consumers.  This is the
serverless-friendly backend for Cloud Run, where the local filesystem (and thus
SQLite) is ephemeral.

Schema (PROJECT_BRIEF §4), user-scoped under ``users/{user_id}``::

    conversations/{id}: started_at, ended_at, transcript, audio_path, created_at, expires_at
    notes/{id}:         conversation_id, summary, is_noteworthy, created_at
    actions/{id}:       conversation_id, intent, details, status, execution_mode, executed_at, created_at
    context/{key}:      value, updated_at

Design notes
------------
- Filtering / ordering / limiting are done **client-side** in Python (fetch a
  collection, then sort/filter/slice).  At personal-assistant scale this is
  simple and keeps the unit tests offline (the test fake only needs
  set/get/stream).  Semantic search is task 2.3.2.
- Timestamps are ISO 8601 strings (``datetime.now().isoformat()``) so the
  returned dataclasses are byte-compatible with the SQLite backend.  ISO strings
  also sort lexicographically, which keeps "newest-first" ordering correct.

TTL
---
``save_conversation`` writes an ``expires_at`` field (created_at + 90 days).  The
actual Firestore TTL **policy** must be enabled out-of-band (it cannot be set
from application code), e.g.::

    gcloud firestore fields ttl update expires_at \\
        --collection-group=conversations

Once enabled, Firestore auto-deletes expired conversation documents.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

from assistant.config import config as _default_config
from assistant.context_assembly import assemble_context_string
from assistant.memory import Action, Conversation, Note, compute_note_expiry

#: Conversations are retained this long; an ``expires_at`` field is written so a
#: Firestore TTL policy (enabled out-of-band) can auto-delete old documents.
CONVERSATION_TTL_DAYS = 90


def _now() -> str:
    return datetime.now().isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


class FirestoreMemory:
    """Firestore-backed persistence with the same interface as :class:`Memory`.

    Parameters
    ----------
    client:
        An optional pre-built ``google.cloud.firestore.Client``.  When *None*,
        a real client is constructed from ``config.gcp.project_id`` using ADC.
        Tests MUST inject a fake client so no real Firestore call is made.
    user_id:
        Namespacing key — all data lives under ``users/{user_id}``.
    """

    def __init__(self, client: Any | None = None, user_id: str = "default") -> None:
        if client is None:
            from google.cloud import firestore  # local import — avoids hard dep at import time

            client = firestore.Client(project=_default_config.gcp.project_id)
        self._client = client
        self._user_id = user_id
        base = client.collection("users").document(user_id)
        self._conversations = base.collection("conversations")
        self._notes = base.collection("notes")
        self._actions = base.collection("actions")
        self._context = base.collection("context")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _docs(collection) -> list[dict]:
        """Return every document in *collection* as a dict (id injected)."""
        out: list[dict] = []
        for snap in collection.stream():
            data = snap.to_dict() or {}
            data["id"] = snap.id
            out.append(data)
        return out

    @staticmethod
    def _to_conversation(d: dict) -> Conversation:
        return Conversation(
            id=d["id"],
            started_at=d.get("started_at", ""),
            ended_at=d.get("ended_at"),
            transcript=d.get("transcript", ""),
            audio_path=d.get("audio_path"),
            created_at=d.get("created_at", ""),
        )

    @staticmethod
    def _to_note(d: dict) -> Note:
        embedding = d.get("embedding")
        return Note(
            id=d["id"],
            conversation_id=d.get("conversation_id", ""),
            summary=d.get("summary", ""),
            is_noteworthy=bool(d.get("is_noteworthy", True)),
            created_at=d.get("created_at", ""),
            importance=d.get("importance"),
            embedding=list(embedding) if embedding is not None else None,
        )

    @staticmethod
    def _to_action(d: dict) -> Action:
        details = d.get("details", {})
        if not isinstance(details, dict):
            details = {"raw": details}
        return Action(
            id=d["id"],
            conversation_id=d.get("conversation_id", ""),
            intent=d.get("intent", ""),
            details=details,
            status=d.get("status", "pending"),
            execution_mode=d.get("execution_mode", ""),
            executed_at=d.get("executed_at"),
            created_at=d.get("created_at", ""),
        )

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def save_conversation(self, transcript: str, audio_path: str | None = None) -> str:
        conversation_id = _new_id()
        now_dt = datetime.now()
        now = now_dt.isoformat()
        expires_at = (now_dt + timedelta(days=CONVERSATION_TTL_DAYS)).isoformat()
        self._conversations.document(conversation_id).set(
            {
                "started_at": now,
                "ended_at": None,
                "transcript": transcript,
                "audio_path": audio_path,
                "created_at": now,
                "expires_at": expires_at,
            }
        )
        return conversation_id

    def get_recent_conversations(self, limit: int = 5) -> list[Conversation]:
        docs = self._docs(self._conversations)
        docs.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return [self._to_conversation(d) for d in docs[:limit]]

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        snap = self._conversations.document(conversation_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        data["id"] = snap.id
        return self._to_conversation(data)

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    def save_note(
        self,
        conversation_id: str,
        summary: str,
        is_noteworthy: bool = True,
        importance: float | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        note_id = _new_id()
        created = _now()
        self._notes.document(note_id).set(
            {
                "conversation_id": conversation_id,
                "summary": summary,
                "is_noteworthy": bool(is_noteworthy),
                "created_at": created,
                "importance": importance,
                "embedding": embedding,
                "expires_at": compute_note_expiry(created, importance),
            }
        )
        return note_id

    def get_recent_notes(self, limit: int = 20) -> list[Note]:
        docs = self._docs(self._notes)
        docs.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return [self._to_note(d) for d in docs[:limit]]

    def search_notes(self, query: str) -> list[Note]:
        """Case-insensitive substring search on summaries (client-side).

        Firestore has no LIKE operator; semantic search is task 2.3.2.
        """
        needle = query.lower()
        docs = [d for d in self._docs(self._notes) if needle in d.get("summary", "").lower()]
        docs.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return [self._to_note(d) for d in docs]

    def get_notes_for_conversation(self, conversation_id: str) -> list[Note]:
        docs = [d for d in self._docs(self._notes) if d.get("conversation_id") == conversation_id]
        docs.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return [self._to_note(d) for d in docs]

    def count_notes(self) -> int:
        return len(self._docs(self._notes))

    def get_notes_without_embedding(self, limit: int = 500) -> list[Note]:
        """Return up to *limit* notes lacking an embedding (oldest first)."""
        docs = [d for d in self._docs(self._notes) if not d.get("embedding")]
        docs.sort(key=lambda d: d.get("created_at", ""))
        return [self._to_note(d) for d in docs[:limit]]

    def update_note_embedding(self, note_id: str, embedding: list[float]) -> None:
        """Set the embedding vector for an existing note."""
        self._notes.document(note_id).update({"embedding": embedding})

    def delete_note(self, note_id: str) -> bool:
        """Delete the note with *note_id*. Returns True if it existed."""
        ref = self._notes.document(note_id)
        if not ref.get().exists:
            return False
        ref.delete()
        return True

    def purge_expired_notes(self, now: str | None = None) -> int:
        """Delete notes whose ``expires_at`` is in the past. Returns the count removed.

        Client-side filter (consistent with the other Firestore queries). For
        scale, enable a native Firestore TTL policy on the notes ``expires_at``
        field instead — this method then becomes a manual top-up.
        """
        cutoff = now or _now()
        removed = 0
        for d in self._docs(self._notes):
            exp = d.get("expires_at")
            if exp is not None and exp < cutoff:
                self._notes.document(d["id"]).delete()
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def save_action(
        self,
        conversation_id: str,
        intent: str,
        details: dict | str,
        execution_mode: str,
    ) -> str:
        import json

        action_id = _new_id()
        details_dict = details if isinstance(details, dict) else json.loads(details)
        self._actions.document(action_id).set(
            {
                "conversation_id": conversation_id,
                "intent": intent,
                "details": details_dict,
                "status": "pending",
                "execution_mode": execution_mode,
                "executed_at": None,
                "created_at": _now(),
            }
        )
        return action_id

    def update_action_status(self, action_id: str, status: str) -> None:
        update: dict[str, Any] = {"status": status}
        if status == "executed":
            update["executed_at"] = _now()
        self._actions.document(action_id).set(update, merge=True)

    def get_pending_actions(self) -> list[Action]:
        docs = [d for d in self._docs(self._actions) if d.get("status") == "pending"]
        docs.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return [self._to_action(d) for d in docs]

    def get_recent_actions(self, limit: int = 10) -> list[Action]:
        docs = self._docs(self._actions)
        docs.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return [self._to_action(d) for d in docs[:limit]]

    def get_actions_for_conversation(self, conversation_id: str) -> list[Action]:
        docs = [d for d in self._docs(self._actions) if d.get("conversation_id") == conversation_id]
        docs.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return [self._to_action(d) for d in docs]

    def get_action(self, action_id: str) -> Action | None:
        snap = self._actions.document(action_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        data["id"] = snap.id
        return self._to_action(data)

    # ------------------------------------------------------------------
    # Context window (key-value store)
    # ------------------------------------------------------------------

    def set_context(self, key: str, value: str) -> None:
        self._context.document(key).set({"value": value, "updated_at": _now()})

    def get_context(self, key: str) -> str | None:
        snap = self._context.document(key).get()
        if not snap.exists:
            return None
        return (snap.to_dict() or {}).get("value")

    def get_all_context(self) -> dict:
        result: dict[str, str] = {}
        for snap in self._context.stream():
            result[snap.id] = (snap.to_dict() or {}).get("value", "")
        return result

    # ------------------------------------------------------------------
    # Context assembly (shared with SQLite Memory)
    # ------------------------------------------------------------------

    def get_recent_noteworthy_notes(self, limit: int = 200) -> list[Note]:
        """Return up to *limit* noteworthy notes, newest-first (with embeddings)."""
        docs = [d for d in self._docs(self._notes) if d.get("is_noteworthy", True)]
        docs.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return [self._to_note(d) for d in docs[:limit]]

    def assemble_context(self, query_embedding: list[float] | None = None) -> str:
        """Assemble the LLM prompt context — identical format to SQLite Memory."""
        from assistant.retrieval import select_context_summaries

        cfg = _default_config.memory
        rcfg = cfg.retrieval

        # 1. Recent History — recency or scored selection over the candidate pool.
        pool_limit = rcfg.candidate_pool if rcfg.mode == "scored" else cfg.context_window_size
        notes = self.get_recent_noteworthy_notes(limit=pool_limit)
        history: list[str] = select_context_summaries(
            notes, query_embedding, rcfg, cfg.context_window_size, datetime.now()
        )

        # 2. Recent actions (last 10).
        action_lines: list[str] = []
        for action in self.get_recent_actions(limit=10):
            if isinstance(action.details, dict) and action.details:
                short_detail = next(iter(action.details.values()))
                if not isinstance(short_detail, str):
                    short_detail = str(short_detail)
            else:
                short_detail = str(action.details)
            action_lines.append(f"{action.intent}: {short_detail}")

        # 3. Known information.
        ctx = self.get_all_context()

        return assemble_context_string(history, action_lines, ctx, cfg.max_context_tokens)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying Firestore client if it exposes a close()."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
