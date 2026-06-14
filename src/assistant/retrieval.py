"""Scored memory retrieval (2.3.2 phase 3).

A pure, dependency-free scorer that ranks notes for inclusion in the LLM
context by combining three signals, following the "memory stream" of Park et
al. 2023 (*Generative Agents*):

    score = w_recency·recency + w_importance·importance + w_relevance·relevance

Each raw signal is in ``[0, 1]`` and is **min-max normalised across the
candidate set** before weighting, so no single signal's spread dominates.  This
module performs no I/O — it takes already-fetched ``Note`` objects and a query
embedding and returns the ranked subset.  The wiring into ``assemble_context``
(and the config that supplies the weights) lands in phase 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from assistant.memory import Note


@dataclass
class RetrievalWeights:
    """Relative weights for the three retrieval signals (default: equal)."""

    recency: float = 1.0
    importance: float = 1.0
    relevance: float = 1.0


@dataclass
class ScoredNote:
    """A note paired with its composite score and the (normalised) components."""

    note: Note
    score: float
    recency: float
    importance: float
    relevance: float


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity of two equal-length vectors, in ``[-1, 1]``.

    Returns ``0.0`` defensively when either vector is empty, their lengths
    differ, or either has zero magnitude (so a degenerate vector never raises).
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _recency_raw(created_at: str, now: datetime, half_life_hours: float) -> float:
    """Exponential-decay recency in ``[0, 1]``: 1.0 now, 0.5 one half-life ago.

    Unparseable timestamps or non-positive half-lives yield ``0.0`` (treated as
    maximally stale) rather than raising.
    """
    if half_life_hours <= 0:
        return 0.0
    try:
        created = datetime.fromisoformat(created_at)
    except (ValueError, TypeError):
        return 0.0
    hours = (now - created).total_seconds() / 3600.0
    if hours < 0:
        hours = 0.0  # clock skew / future timestamp → treat as "now"
    return 0.5 ** (hours / half_life_hours)


def _minmax(values: list[float]) -> list[float]:
    """Min-max normalise *values* to ``[0, 1]``.

    When every value is equal (zero spread) the signal carries no discriminating
    information for this candidate set, so all entries normalise to ``0.0``.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def rank_memories(
    query_embedding: list[float] | None,
    notes: list[Note],
    now: datetime,
    weights: RetrievalWeights | None = None,
    half_life_hours: float = 72.0,
    top_k: int = 6,
) -> list[ScoredNote]:
    """Rank *notes* by composite recency/importance/relevance and return top-K.

    Args:
        query_embedding: Embedding of the current transcript.  When ``None`` the
                         relevance signal is ignored (all-zero) and ranking falls
                         back to recency + importance.
        notes:           Candidate notes (each may carry ``importance`` /
                         ``embedding``; missing values are treated as neutral 0).
        now:             Reference time for recency decay.
        weights:         Signal weights (defaults to equal weighting).
        half_life_hours: Recency half-life in hours.
        top_k:           Maximum number of notes to return.

    Returns:
        Up to *top_k* :class:`ScoredNote` objects, highest score first.
    """
    if not notes:
        return []
    weights = weights or RetrievalWeights()

    recency_raw = [_recency_raw(n.created_at, now, half_life_hours) for n in notes]
    importance_raw = [n.importance if n.importance is not None else 0.0 for n in notes]
    if query_embedding:
        relevance_raw = [
            max(0.0, cosine_similarity(query_embedding, n.embedding)) if n.embedding else 0.0
            for n in notes
        ]
    else:
        relevance_raw = [0.0 for _ in notes]

    recency_n = _minmax(recency_raw)
    importance_n = _minmax(importance_raw)
    relevance_n = _minmax(relevance_raw)

    scored: list[ScoredNote] = []
    for note, rec, imp, rel in zip(notes, recency_n, importance_n, relevance_n, strict=True):
        score = weights.recency * rec + weights.importance * imp + weights.relevance * rel
        scored.append(
            ScoredNote(note=note, score=score, recency=rec, importance=imp, relevance=rel)
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:top_k]
