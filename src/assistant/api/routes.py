"""FastAPI router implementing all /api/v1 endpoints.

Endpoints
---------
GET   /api/v1/notes               — list notes, paginated
GET   /api/v1/notes/{id}          — note detail + conversation + actions
DELETE /api/v1/notes/{id}         — delete a note
GET   /api/v1/actions             — list actions with optional status/intent filters
PATCH /api/v1/actions/{id}        — update action status
POST  /api/v1/search              — keyword search over note summaries
GET   /api/v1/context             — full context window key→value dict

The Memory instance is injected via the ``get_memory`` FastAPI dependency so
that tests can override it with an in-memory fixture without patching globals.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from assistant.config import config
from assistant.memory import Memory, Note
from assistant.memory_factory import create_memory

from .schemas import (
    ActionOut,
    ActionResponse,
    ActionsListResponse,
    ActionStatusUpdate,
    ContextResponse,
    ConversationOut,
    NoteDetailResponse,
    NoteOut,
    NotesListResponse,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

_LARGE_LIMIT = 10_000  # practical upper bound for "fetch all" slicing


def get_memory() -> Generator[Memory, None, None]:
    """Yield the configured storage backend (SQLite or Firestore) and close it.

    The backend is chosen by ``config.memory.backend`` via
    :func:`~assistant.memory_factory.create_memory`.  Tests override this via
    ``app.dependency_overrides[get_memory]``.
    """
    mem = create_memory()
    try:
        yield mem
    finally:
        mem.close()


MemoryDep = Annotated[Memory, Depends(get_memory)]

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1")


# --- Notes ------------------------------------------------------------------


@router.get("/notes", response_model=NotesListResponse)
def list_notes(
    memory: MemoryDep,
    limit: int = Query(default=20, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> NotesListResponse:
    """Return a paginated slice of all notes, newest first."""
    total = memory.count_notes()
    # Fetch enough rows to cover offset+limit, then slice in Python.
    all_notes = memory.get_recent_notes(limit=offset + limit)
    page = all_notes[offset : offset + limit]
    return NotesListResponse(
        notes=[NoteOut.from_dataclass(n) for n in page],
        total=total,
    )


@router.get("/notes/{note_id}", response_model=NoteDetailResponse)
def get_note(note_id: str, memory: MemoryDep) -> NoteDetailResponse:
    """Return a note by ID together with its parent conversation and actions."""
    all_notes = memory.get_recent_notes(limit=_LARGE_LIMIT)
    note = next((n for n in all_notes if n.id == note_id), None)
    if note is None:
        raise HTTPException(status_code=404, detail=f"Note '{note_id}' not found")

    conversation = memory.get_conversation(note.conversation_id)
    actions = memory.get_actions_for_conversation(note.conversation_id)

    return NoteDetailResponse(
        note=NoteOut.from_dataclass(note),
        conversation=ConversationOut.from_dataclass(conversation) if conversation else None,
        actions=[ActionOut.from_dataclass(a) for a in actions],
    )


@router.delete("/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_note(note_id: str, memory: MemoryDep) -> Response:
    """Delete a note by ID. 204 on success, 404 if it does not exist."""
    if not memory.delete_note(note_id):
        raise HTTPException(status_code=404, detail=f"Note '{note_id}' not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- Actions ----------------------------------------------------------------


@router.get("/actions", response_model=ActionsListResponse)
def list_actions(
    memory: MemoryDep,
    status: str | None = Query(default=None),
    intent: str | None = Query(default=None),
) -> ActionsListResponse:
    """Return actions, optionally filtered by *status* and/or *intent*."""
    actions = memory.get_recent_actions(limit=_LARGE_LIMIT)
    if status is not None:
        actions = [a for a in actions if a.status == status]
    if intent is not None:
        actions = [a for a in actions if a.intent == intent]
    return ActionsListResponse(actions=[ActionOut.from_dataclass(a) for a in actions])


@router.patch("/actions/{action_id}", response_model=ActionResponse)
def update_action(
    action_id: str,
    body: ActionStatusUpdate,
    memory: MemoryDep,
) -> ActionResponse:
    """Update the status of an action and return the updated record."""
    existing = memory.get_action(action_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Action '{action_id}' not found")
    memory.update_action_status(action_id, body.status)
    updated = memory.get_action(action_id)
    return ActionResponse(action=ActionOut.from_dataclass(updated))  # type: ignore[arg-type]


# --- Search -----------------------------------------------------------------


def make_embedder():
    """Return an Embedder for query embedding (monkeypatched in tests).

    Lazy import keeps the router importable without google-genai installed.
    """
    from assistant.embeddings import Embedder

    return Embedder()


def _semantic_search(memory: Memory, query: str, limit: int) -> list[Note] | None:
    """Rank notes by embedding similarity to *query*.

    Returns ``None`` (so the caller falls back to keyword search) when note
    embedding is disabled, the query can't be embedded, or no note has a vector.
    """
    if not config.memory.embed_notes:
        return None
    try:
        query_embedding = make_embedder().embed_text(query)
    except Exception:  # noqa: BLE001
        logger.warning("Semantic search embedding failed; falling back to keyword", exc_info=True)
        return None

    from assistant.retrieval import cosine_similarity

    candidates = [n for n in memory.get_recent_notes(limit=_LARGE_LIMIT) if n.embedding]
    if not candidates:
        return None
    candidates.sort(key=lambda n: cosine_similarity(query_embedding, n.embedding), reverse=True)
    return candidates[:limit]


@router.post("/search", response_model=SearchResponse)
def search_notes(body: SearchRequest, memory: MemoryDep) -> SearchResponse:
    """Search note summaries — semantic (by meaning) when embeddings are enabled,
    otherwise a keyword substring match."""
    results = _semantic_search(memory, body.query, limit=20)
    if results is None:
        results = memory.search_notes(body.query)
    return SearchResponse(results=[NoteOut.from_dataclass(n) for n in results])


# --- Context ----------------------------------------------------------------


@router.get("/context", response_model=ContextResponse)
def get_context(memory: MemoryDep) -> ContextResponse:
    """Return the entire context window as a key→value dict."""
    return ContextResponse(context=memory.get_all_context())
