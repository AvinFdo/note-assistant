"""Memory: SQLite-backed CRUD for conversations, notes, actions, and rolling context.

Abstracts all persistence behind a clean, storage-agnostic interface so that a future
migration to Firestore (Stage 2) is an implementation swap, not a consumer rewrite.
All identifiers are UUID strings; all timestamps are ISO 8601.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from assistant.config import config as _default_config
from assistant.context_assembly import assemble_context_string

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class MemoryError(Exception):
    """Base exception for all Memory storage errors."""


class RecordNotFoundError(MemoryError):
    """Raised when a requested record does not exist in the database."""


# ---------------------------------------------------------------------------
# Return-type dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Conversation:
    id: str
    started_at: str
    ended_at: str | None
    transcript: str
    audio_path: str | None
    created_at: str


@dataclass
class Note:
    id: str
    conversation_id: str
    summary: str
    is_noteworthy: bool
    created_at: str
    # Scored-retrieval fields (2.3.2). Both are nullable: legacy notes written
    # before this migration have None until backfilled / re-rated.
    importance: float | None = None  # 0..1 (LLM 1-10 rating normalised)
    embedding: list[float] | None = None  # semantic vector of the summary


@dataclass
class Action:
    id: str
    conversation_id: str
    intent: str
    details: dict
    status: str
    execution_mode: str
    executed_at: str | None
    created_at: str


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    transcript  TEXT NOT NULL,
    audio_path  TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS notes (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    summary         TEXT NOT NULL,
    is_noteworthy   INTEGER DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    importance      REAL,            -- 0..1 (LLM rating); NULL = unrated/legacy
    embedding       TEXT,            -- JSON float array; NULL = not yet embedded
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS actions (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    intent          TEXT NOT NULL,
    details         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    execution_mode  TEXT NOT NULL,
    executed_at     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS context_window (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now().isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Memory class
# ---------------------------------------------------------------------------


class Memory:
    """SQLite-backed persistence layer for the Avin assistant.

    Pass ``db_path=":memory:"`` for in-process testing — no files are created.
    Otherwise, defaults to ``config.memory.db_path`` and auto-creates the
    parent directory on first use.
    """

    def __init__(self, db_path: str | None = None) -> None:
        resolved = db_path if db_path is not None else _default_config.memory.db_path
        if resolved != ":memory:":
            Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(resolved, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate_notes_columns()
        self._conn.commit()

    def _migrate_notes_columns(self) -> None:
        """Add the scored-retrieval columns to a pre-existing notes table.

        ``CREATE TABLE IF NOT EXISTS`` never alters an existing table, so a DB
        created before the 2.3.2 migration lacks ``importance`` / ``embedding``.
        SQLite has no ``ADD COLUMN IF NOT EXISTS``, so we inspect the schema and
        add only what's missing — a no-op on freshly created tables.
        """
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(notes)")}
        if "importance" not in existing:
            self._conn.execute("ALTER TABLE notes ADD COLUMN importance REAL")
        if "embedding" not in existing:
            self._conn.execute("ALTER TABLE notes ADD COLUMN embedding TEXT")

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    def save_conversation(
        self,
        transcript: str,
        audio_path: str | None = None,
    ) -> str:
        """Persist a new conversation and return its UUID string."""
        conversation_id = _new_id()
        now = _now()
        self._conn.execute(
            """
            INSERT INTO conversations (id, started_at, transcript, audio_path, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, now, transcript, audio_path, now),
        )
        self._conn.commit()
        return conversation_id

    def get_recent_conversations(self, limit: int = 5) -> list[Conversation]:
        """Return up to *limit* conversations, newest first."""
        rows = self._conn.execute(
            """
            SELECT id, started_at, ended_at, transcript, audio_path, created_at
            FROM conversations
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            Conversation(
                id=row["id"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                transcript=row["transcript"],
                audio_path=row["audio_path"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

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
        """Persist a note linked to *conversation_id* and return its UUID string.

        ``importance`` (0..1) and ``embedding`` are optional scored-retrieval
        fields; when omitted they are stored as NULL and the note is treated as
        unrated / not-yet-embedded by the retrieval layer.
        """
        note_id = _new_id()
        embedding_json = json.dumps(embedding) if embedding is not None else None
        self._conn.execute(
            """
            INSERT INTO notes
                (id, conversation_id, summary, is_noteworthy, created_at, importance, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note_id,
                conversation_id,
                summary,
                1 if is_noteworthy else 0,
                _now(),
                importance,
                embedding_json,
            ),
        )
        self._conn.commit()
        return note_id

    def get_recent_notes(self, limit: int = 20) -> list[Note]:
        """Return up to *limit* notes, newest first."""
        rows = self._conn.execute(
            """
            SELECT id, conversation_id, summary, is_noteworthy, created_at, importance, embedding
            FROM notes
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_note(row) for row in rows]

    def count_notes(self) -> int:
        """Return the total number of notes stored."""
        row = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()
        return row[0]

    def get_notes_without_embedding(self, limit: int = 500) -> list[Note]:
        """Return up to *limit* notes that have no embedding yet (oldest first).

        Used by the one-off backfill to embed legacy notes.  Oldest-first so a
        resumable backfill makes steady forward progress.
        """
        rows = self._conn.execute(
            """
            SELECT id, conversation_id, summary, is_noteworthy, created_at, importance, embedding
            FROM notes
            WHERE embedding IS NULL
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_note(row) for row in rows]

    def update_note_embedding(self, note_id: str, embedding: list[float]) -> None:
        """Set the embedding vector (JSON-serialised) for an existing note."""
        self._conn.execute(
            "UPDATE notes SET embedding = ? WHERE id = ?",
            (json.dumps(embedding), note_id),
        )
        self._conn.commit()

    def delete_note(self, note_id: str) -> bool:
        """Delete the note with *note_id*. Returns True if a row was removed."""
        cur = self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def search_notes(self, query: str) -> list[Note]:
        """Return notes whose summary contains *query* (case-insensitive LIKE search).

        Uses parameterized SQL — safe against injection.
        """
        pattern = f"%{query}%"
        rows = self._conn.execute(
            """
            SELECT id, conversation_id, summary, is_noteworthy, created_at, importance, embedding
            FROM notes
            WHERE summary LIKE ?
            ORDER BY created_at DESC
            """,
            (pattern,),
        ).fetchall()
        return [self._row_to_note(row) for row in rows]

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
        """Persist an action linked to *conversation_id* and return its UUID string.

        *details* is serialised to a JSON string if a dict is passed.
        """
        action_id = _new_id()
        details_str = json.dumps(details) if isinstance(details, dict) else details
        self._conn.execute(
            """
            INSERT INTO actions (id, conversation_id, intent, details, execution_mode, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (action_id, conversation_id, intent, details_str, execution_mode, _now()),
        )
        self._conn.commit()
        return action_id

    def update_action_status(self, action_id: str, status: str) -> None:
        """Update the status of an action.  Sets *executed_at* when status is 'executed'."""
        if status == "executed":
            self._conn.execute(
                """
                UPDATE actions SET status = ?, executed_at = ? WHERE id = ?
                """,
                (status, _now(), action_id),
            )
        else:
            self._conn.execute(
                """
                UPDATE actions SET status = ? WHERE id = ?
                """,
                (status, action_id),
            )
        self._conn.commit()

    def get_pending_actions(self) -> list[Action]:
        """Return all actions with status='pending'."""
        rows = self._conn.execute(
            """
            SELECT id, conversation_id, intent, details, status,
                   execution_mode, executed_at, created_at
            FROM actions
            WHERE status = 'pending'
            ORDER BY created_at DESC
            """,
        ).fetchall()
        return [self._row_to_action(row) for row in rows]

    def get_recent_actions(self, limit: int = 10) -> list[Action]:
        """Return up to *limit* actions (any status), newest first."""
        rows = self._conn.execute(
            """
            SELECT id, conversation_id, intent, details, status,
                   execution_mode, executed_at, created_at
            FROM actions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_action(row) for row in rows]

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> Note:
        keys = row.keys()
        importance = row["importance"] if "importance" in keys else None
        embedding = None
        if "embedding" in keys and row["embedding"] is not None:
            try:
                embedding = json.loads(row["embedding"])
            except (json.JSONDecodeError, TypeError):
                embedding = None
        return Note(
            id=row["id"],
            conversation_id=row["conversation_id"],
            summary=row["summary"],
            is_noteworthy=bool(row["is_noteworthy"]),
            created_at=row["created_at"],
            importance=importance,
            embedding=embedding,
        )

    @staticmethod
    def _row_to_action(row: sqlite3.Row) -> Action:
        try:
            details = json.loads(row["details"])
        except (json.JSONDecodeError, TypeError):
            details = {"raw": row["details"]}
        return Action(
            id=row["id"],
            conversation_id=row["conversation_id"],
            intent=row["intent"],
            details=details,
            status=row["status"],
            execution_mode=row["execution_mode"],
            executed_at=row["executed_at"],
            created_at=row["created_at"],
        )

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Return a single conversation by *conversation_id*, or ``None`` if not found."""
        row = self._conn.execute(
            """
            SELECT id, started_at, ended_at, transcript, audio_path, created_at
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        return Conversation(
            id=row["id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            transcript=row["transcript"],
            audio_path=row["audio_path"],
            created_at=row["created_at"],
        )

    def get_notes_for_conversation(self, conversation_id: str) -> list[Note]:
        """Return all notes linked to *conversation_id*, newest first."""
        rows = self._conn.execute(
            """
            SELECT id, conversation_id, summary, is_noteworthy, created_at, importance, embedding
            FROM notes
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            """,
            (conversation_id,),
        ).fetchall()
        return [self._row_to_note(row) for row in rows]

    def get_actions_for_conversation(self, conversation_id: str) -> list[Action]:
        """Return all actions linked to *conversation_id*, newest first."""
        rows = self._conn.execute(
            """
            SELECT id, conversation_id, intent, details, status,
                   execution_mode, executed_at, created_at
            FROM actions
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            """,
            (conversation_id,),
        ).fetchall()
        return [self._row_to_action(row) for row in rows]

    def get_action(self, action_id: str) -> Action | None:
        """Return a single action by *action_id*, or ``None`` if not found."""
        row = self._conn.execute(
            """
            SELECT id, conversation_id, intent, details, status,
                   execution_mode, executed_at, created_at
            FROM actions
            WHERE id = ?
            """,
            (action_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_action(row)

    # ------------------------------------------------------------------
    # Context window (key-value store)
    # ------------------------------------------------------------------

    def set_context(self, key: str, value: str) -> None:
        """Insert or update a context key, refreshing *updated_at*."""
        self._conn.execute(
            """
            INSERT INTO context_window (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_at = excluded.updated_at
            """,
            (key, value, _now()),
        )
        self._conn.commit()

    def get_context(self, key: str) -> str | None:
        """Return the value for *key*, or None if not set."""
        row = self._conn.execute(
            "SELECT value FROM context_window WHERE key = ?",
            (key,),
        ).fetchone()
        return row["value"] if row else None

    def get_all_context(self) -> dict:
        """Return all context key-value pairs as a plain dict."""
        rows = self._conn.execute(
            "SELECT key, value FROM context_window",
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    # ------------------------------------------------------------------
    # Context assembly for LLM prompt enrichment
    # ------------------------------------------------------------------

    def get_recent_noteworthy_notes(self, limit: int = 200) -> list[Note]:
        """Return up to *limit* noteworthy notes, newest-first (with embeddings).

        The candidate pool for both recency and scored context assembly.
        """
        rows = self._conn.execute(
            """
            SELECT id, conversation_id, summary, is_noteworthy, created_at, importance, embedding
            FROM notes
            WHERE is_noteworthy = 1
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_note(row) for row in rows]

    def assemble_context(self, query_embedding: list[float] | None = None) -> str:
        """Assemble the "memory" section of the LLM prompt from three sources.

        Delegates formatting and truncation to
        :func:`~assistant.context_assembly.assemble_context_string` so that
        SQLite Memory and FirestoreMemory produce identical output.

        Sources
        -------
        1. **Recent History** — chosen by ``config.memory.retrieval``: either the
           last ``context_window_size`` noteworthy notes (``recency`` mode) or the
           top-``top_k`` by recency+importance+relevance (``scored`` mode, using
           *query_embedding*).
        2. **Pending / recent actions** — last 10 actions (any status), newest-first.
        3. **Known Information** — all key-value pairs from ``context_window``.

        Truncation
        ----------
        If the assembled string exceeds ``config.memory.max_context_tokens * 4``
        characters, the **oldest history entries** are dropped until it fits.
        """
        from assistant.retrieval import select_context_summaries

        cfg = _default_config.memory
        rcfg = cfg.retrieval

        # 1. Recent History — recency or scored selection over the candidate pool.
        pool_limit = rcfg.candidate_pool if rcfg.mode == "scored" else cfg.context_window_size
        notes = self.get_recent_noteworthy_notes(limit=pool_limit)
        history: list[str] = select_context_summaries(
            notes, query_embedding, rcfg, cfg.context_window_size, datetime.now()
        )

        # 2. Recent actions (last 10)
        actions = self.get_recent_actions(limit=10)
        action_lines: list[str] = []
        for action in actions:
            # Provide a short detail string: first value in details dict, or raw JSON
            if isinstance(action.details, dict) and action.details:
                short_detail = next(iter(action.details.values()))
                if not isinstance(short_detail, str):
                    short_detail = str(short_detail)
            else:
                short_detail = str(action.details)
            action_lines.append(f"{action.intent}: {short_detail}")

        # 3. Known information (key-value context window)
        ctx = self.get_all_context()

        return assemble_context_string(history, action_lines, ctx, cfg.max_context_tokens)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
