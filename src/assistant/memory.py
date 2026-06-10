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
        self._conn = sqlite3.connect(resolved, check_same_thread=False)
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

    def count_notes(self) -> int:
        """Return the total number of notes stored."""
        row = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()
        return row[0]

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
            SELECT id, conversation_id, summary, is_noteworthy, created_at
            FROM notes
            WHERE conversation_id = ?
            ORDER BY created_at DESC
            """,
            (conversation_id,),
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

    def assemble_context(self) -> str:
        """Assemble the "memory" section of the LLM prompt from three sources.

        Format
        ------
        The returned string follows the prompt template defined in PROJECT_BRIEF §6::

            CONTEXT (Recent History):
            - <summary 1>
            - <summary 2>

            CONTEXT (Known Information):
            - key: value

            CONTEXT (Pending Actions):
            - <intent>: <short detail>

        Sources
        -------
        1. **Recent History** — last ``config.memory.context_window_size`` noteworthy
           notes (``is_noteworthy=1``), ordered newest-first.
        2. **Pending / recent actions** — last 10 actions (any status), newest-first.
           Each line shows ``intent: <first detail value or raw JSON>``.
        3. **Known Information** — all key-value pairs from ``context_window`` via
           :meth:`get_all_context`.

        Truncation
        ----------
        If the assembled string exceeds ``config.memory.max_context_tokens * 4``
        characters (approximate: 1 token ≈ 4 chars), the **oldest history entries**
        are dropped one-by-one until the string fits within the budget.  Known
        Information and section headers are always preserved.  Truncation is
        deterministic: entries are removed from the oldest end of the history list
        first (the list is ordered newest-first, so we pop from the tail).
        """
        cfg = _default_config.memory
        char_budget = cfg.max_context_tokens * 4

        # 1. Recent noteworthy history
        rows = self._conn.execute(
            """
            SELECT summary
            FROM notes
            WHERE is_noteworthy = 1
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (cfg.context_window_size,),
        ).fetchall()
        # history is newest-first; index 0 = newest, last = oldest
        history: list[str] = [row["summary"] for row in rows]

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

        def _build(hist: list[str]) -> str:
            history_section = "\n".join(f"- {s}" for s in hist) if hist else "- (none)"
            info_section = "\n".join(f"- {k}: {v}" for k, v in ctx.items()) if ctx else "- (none)"
            actions_section = (
                "\n".join(f"- {line}" for line in action_lines) if action_lines else "- (none)"
            )

            return (
                f"CONTEXT (Recent History):\n{history_section}\n\n"
                f"CONTEXT (Known Information):\n{info_section}\n\n"
                f"CONTEXT (Pending Actions):\n{actions_section}"
            )

        result = _build(history)

        # Truncate by dropping oldest history entries until within budget
        # history[0] = newest, history[-1] = oldest  →  pop from the tail
        while len(result) > char_budget and history:
            history.pop()  # remove oldest entry
            result = _build(history)

        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
