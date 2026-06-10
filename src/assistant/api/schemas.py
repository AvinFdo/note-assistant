"""Pydantic v2 response/request schemas for the Avin REST API.

Each schema maps cleanly from the Memory dataclasses (Note, Action, Conversation)
without leaking SQLite implementation details into the API layer.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from assistant.memory import Action, Conversation, Note

# ---------------------------------------------------------------------------
# Output schemas — map from Memory dataclasses
# ---------------------------------------------------------------------------


class NoteOut(BaseModel):
    """Serialised representation of a single Note."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    summary: str
    is_noteworthy: bool
    created_at: str

    @classmethod
    def from_dataclass(cls, note: Note) -> NoteOut:
        """Construct from a :class:`~assistant.memory.Note` dataclass."""
        return cls(
            id=note.id,
            conversation_id=note.conversation_id,
            summary=note.summary,
            is_noteworthy=note.is_noteworthy,
            created_at=note.created_at,
        )


class ConversationOut(BaseModel):
    """Serialised representation of a single Conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    started_at: str
    ended_at: str | None
    transcript: str
    audio_path: str | None
    created_at: str

    @classmethod
    def from_dataclass(cls, conv: Conversation) -> ConversationOut:
        """Construct from a :class:`~assistant.memory.Conversation` dataclass."""
        return cls(
            id=conv.id,
            started_at=conv.started_at,
            ended_at=conv.ended_at,
            transcript=conv.transcript,
            audio_path=conv.audio_path,
            created_at=conv.created_at,
        )


class ActionOut(BaseModel):
    """Serialised representation of a single Action."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    intent: str
    details: dict[str, Any]
    status: str
    execution_mode: str
    executed_at: str | None
    created_at: str

    @classmethod
    def from_dataclass(cls, action: Action) -> ActionOut:
        """Construct from a :class:`~assistant.memory.Action` dataclass."""
        return cls(
            id=action.id,
            conversation_id=action.conversation_id,
            intent=action.intent,
            details=action.details,
            status=action.status,
            execution_mode=action.execution_mode,
            executed_at=action.executed_at,
            created_at=action.created_at,
        )


# ---------------------------------------------------------------------------
# List/detail response schemas
# ---------------------------------------------------------------------------


class NotesListResponse(BaseModel):
    """Paginated list of notes."""

    notes: list[NoteOut]
    total: int


class NoteDetailResponse(BaseModel):
    """Full detail for a single note including its conversation and actions."""

    note: NoteOut
    conversation: ConversationOut | None
    actions: list[ActionOut]


class ActionsListResponse(BaseModel):
    """List of actions, optionally filtered."""

    actions: list[ActionOut]


class ActionResponse(BaseModel):
    """Single action response (e.g. after a PATCH)."""

    action: ActionOut


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Body for the POST /search endpoint."""

    query: str


class SearchResponse(BaseModel):
    """Search results — list of matching notes."""

    results: list[NoteOut]


class ContextResponse(BaseModel):
    """Full context window as a key→value dict."""

    context: dict[str, str]


class ActionStatusUpdate(BaseModel):
    """Body for PATCH /actions/{id} — update the action status."""

    status: str
