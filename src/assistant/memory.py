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
        self._conn = sqlite3.connect(resolved)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

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
    ) -> str:
        """Persist a note linked to *conversation_id* and return its UUID string."""
        note_id = _new_id()
        self._conn.execute(
            """
            INSERT INTO notes (id, conversation_id, summary, is_noteworthy, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (note_id, conversation_id, summary, 1 if is_noteworthy else 0, _now()),
        )
        self._conn.commit()
        return note_id

    def get_recent_notes(self, limit: int = 20) -> list[Note]:
        """Return up to *limit* notes, newest first."""
        rows = self._conn.execute(
            """
            SELECT id, conversation_id, summary, is_noteworthy, created_at
            FROM notes
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            Note(
                id=row["id"],
                conversation_id=row["conversation_id"],
                summary=row["summary"],
                is_noteworthy=bool(row["is_noteworthy"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def search_notes(self, query: str) -> list[Note]:
        """Return notes whose summary contains *query* (case-insensitive LIKE search).

        Uses parameterized SQL — safe against injection.
        """
        pattern = f"%{query}%"
        rows = self._conn.execute(
            """
            SELECT id, conversation_id, summary, is_noteworthy, created_at
            FROM notes
            WHERE summary LIKE ?
            ORDER BY created_at DESC
            """,
            (pattern,),
        ).fetchall()
        return [
            Note(
                id=row["id"],
                conversation_id=row["conversation_id"],
                summary=row["summary"],
                is_noteworthy=bool(row["is_noteworthy"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

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
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
