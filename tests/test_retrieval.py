"""Tests for assistant.retrieval — pure recency/importance/relevance scorer (2.3.2)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from assistant.memory import Note
from assistant.retrieval import (
    RetrievalWeights,
    cosine_similarity,
    rank_memories,
)

NOW = datetime(2026, 6, 14, 12, 0, 0)


def _note(
    note_id: str,
    *,
    summary: str = "s",
    created_at: datetime | None = None,
    importance: float | None = None,
    embedding: list[float] | None = None,
) -> Note:
    return Note(
        id=note_id,
        conversation_id="c",
        summary=summary,
        is_noteworthy=True,
        created_at=(created_at or NOW).isoformat(),
        importance=importance,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_returns_zero(self):
        assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_empty_returns_zero(self):
        assert cosine_similarity([], [1.0]) == 0.0


# ---------------------------------------------------------------------------
# rank_memories — ordering & top-k
# ---------------------------------------------------------------------------


class TestRanking:
    def test_empty_notes_returns_empty(self):
        assert rank_memories([1.0], [], NOW) == []

    def test_top_k_limit(self):
        notes = [_note(str(i), importance=i / 10) for i in range(10)]
        ranked = rank_memories(None, notes, NOW, top_k=3)
        assert len(ranked) == 3

    def test_recency_orders_when_only_signal(self):
        """With no query and equal importance, newer notes rank higher."""
        old = _note("old", created_at=NOW - timedelta(days=30), importance=0.5)
        mid = _note("mid", created_at=NOW - timedelta(days=3), importance=0.5)
        new = _note("new", created_at=NOW - timedelta(hours=1), importance=0.5)
        ranked = rank_memories(None, [old, mid, new], NOW, top_k=3)
        assert [s.note.id for s in ranked] == ["new", "mid", "old"]

    def test_relevance_pulls_old_but_on_topic_note(self):
        """A semantically matching old note can beat a recent off-topic one."""
        query = [1.0, 0.0]
        on_topic_old = _note(
            "relevant",
            created_at=NOW - timedelta(days=20),
            importance=0.5,
            embedding=[1.0, 0.0],  # cosine 1.0 with query
        )
        off_topic_new = _note(
            "irrelevant",
            created_at=NOW - timedelta(hours=1),
            importance=0.5,
            embedding=[0.0, 1.0],  # cosine 0.0 with query
        )
        weights = RetrievalWeights(recency=1.0, importance=1.0, relevance=3.0)
        ranked = rank_memories(query, [on_topic_old, off_topic_new], NOW, weights=weights)
        assert ranked[0].note.id == "relevant"

    def test_importance_breaks_ties(self):
        """Same age, no query → higher importance wins."""
        a = _note("low", importance=0.1)
        b = _note("high", importance=0.9)
        ranked = rank_memories(None, [a, b], NOW)
        assert ranked[0].note.id == "high"

    def test_missing_importance_treated_as_zero(self):
        rated = _note("rated", importance=0.8)
        unrated = _note("unrated", importance=None)
        ranked = rank_memories(None, [rated, unrated], NOW)
        assert ranked[0].note.id == "rated"

    def test_legacy_note_without_embedding_has_zero_relevance(self):
        """A note with no embedding contributes 0 relevance, never crashes."""
        query = [1.0, 0.0]
        legacy = _note("legacy", importance=0.5, embedding=None)
        embedded = _note("embedded", importance=0.5, embedding=[1.0, 0.0])
        ranked = rank_memories(query, [legacy, embedded], NOW, top_k=2)
        ids = [s.note.id for s in ranked]
        assert ids[0] == "embedded"
        assert set(ids) == {"legacy", "embedded"}


# ---------------------------------------------------------------------------
# Component values
# ---------------------------------------------------------------------------


class TestComponents:
    def test_components_normalised_to_unit_range(self):
        notes = [
            _note("a", created_at=NOW - timedelta(days=10), importance=0.0),
            _note("b", created_at=NOW, importance=1.0),
        ]
        ranked = rank_memories(None, notes, NOW, top_k=2)
        for s in ranked:
            assert 0.0 <= s.recency <= 1.0
            assert 0.0 <= s.importance <= 1.0
            assert 0.0 <= s.relevance <= 1.0

    def test_uniform_signal_normalises_to_zero(self):
        """When all notes share a value, that signal carries no information."""
        notes = [_note("a", importance=0.5), _note("b", importance=0.5)]
        ranked = rank_memories(None, notes, NOW, top_k=2)
        # identical age + identical importance → all components zero → score zero
        assert all(s.score == 0.0 for s in ranked)
