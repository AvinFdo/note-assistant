"""Tests for assistant.backfill — embedding backfill for legacy notes (2.3.2)."""

from __future__ import annotations

from assistant.backfill import backfill_note_embeddings
from assistant.memory import Memory


class _FakeEmbedder:
    """Returns a deterministic vector per text and records the calls."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(t))] for t in texts]


def _mem_with_notes(n: int, *, embedded: int = 0) -> Memory:
    m = Memory(db_path=":memory:")
    cid = m.save_conversation("transcript")
    for i in range(n):
        emb = [0.0] if i < embedded else None
        m.save_note(cid, f"note {i}", embedding=emb)
    return m


def test_backfill_embeds_all_missing():
    mem = _mem_with_notes(3)
    embedder = _FakeEmbedder()
    count = backfill_note_embeddings(mem, embedder, batch_size=10)
    assert count == 3
    assert mem.get_notes_without_embedding() == []
    mem.close()


def test_backfill_skips_already_embedded():
    mem = _mem_with_notes(5, embedded=2)
    embedder = _FakeEmbedder()
    count = backfill_note_embeddings(mem, embedder, batch_size=10)
    assert count == 3  # only the 3 unembedded
    mem.close()


def test_backfill_respects_max_notes():
    mem = _mem_with_notes(5)
    embedder = _FakeEmbedder()
    count = backfill_note_embeddings(mem, embedder, batch_size=10, max_notes=2)
    assert count == 2
    assert len(mem.get_notes_without_embedding()) == 3
    mem.close()


def test_backfill_batches():
    mem = _mem_with_notes(5)
    embedder = _FakeEmbedder()
    backfill_note_embeddings(mem, embedder, batch_size=2)
    # 5 notes in batches of 2 → batch sizes 2, 2, 1
    assert [len(b) for b in embedder.batches] == [2, 2, 1]
    mem.close()


def test_backfill_noop_when_all_embedded():
    mem = _mem_with_notes(3, embedded=3)
    embedder = _FakeEmbedder()
    assert backfill_note_embeddings(mem, embedder) == 0
    assert embedder.batches == []
    mem.close()
