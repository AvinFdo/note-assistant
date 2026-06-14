"""One-off backfill: embed existing notes that predate the 2.3.2 migration.

Legacy notes have ``embedding = NULL``.  This walks them in batches, embeds each
summary, and writes the vector back via ``memory.update_note_embedding``.  It is
idempotent and resumable — re-running only processes notes that still lack a
vector — so it is safe to run repeatedly.

Run against the configured backend::

    python -m assistant.backfill            # uses config.memory.backend
    python -m assistant.backfill --limit 50 # cap total notes this run

The core :func:`backfill_note_embeddings` takes injected ``memory`` and
``embedder`` objects so it is fully unit-testable offline.
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)


def backfill_note_embeddings(
    memory,
    embedder,
    batch_size: int = 100,
    max_notes: int | None = None,
) -> int:
    """Embed all notes lacking a vector and return the count updated.

    Args:
        memory:     A Memory / FirestoreMemory exposing
                    ``get_notes_without_embedding`` and ``update_note_embedding``.
        embedder:   An :class:`~assistant.embeddings.Embedder` (or compatible).
        batch_size: How many notes to embed per API call / loop iteration.
        max_notes:  Optional cap on the total processed this run (``None`` = all).

    Returns:
        The number of notes that were embedded and updated.
    """
    total = 0
    while True:
        remaining = batch_size if max_notes is None else min(batch_size, max_notes - total)
        if remaining <= 0:
            break

        notes = memory.get_notes_without_embedding(limit=remaining)
        if not notes:
            break

        vectors = embedder.embed_batch([n.summary for n in notes])
        for note, vector in zip(notes, vectors, strict=True):
            memory.update_note_embedding(note.id, vector)
            total += 1

        logger.info("Backfilled %d notes so far", total)

        # A short final batch means we've drained the queue.
        if len(notes) < remaining:
            break

    return total


def main() -> None:
    """CLI entry point: backfill embeddings for the configured memory backend."""
    parser = argparse.ArgumentParser(description="Backfill note embeddings (2.3.2).")
    parser.add_argument("--limit", type=int, default=None, help="Max notes to embed this run.")
    parser.add_argument("--batch-size", type=int, default=100, help="Notes per embedding call.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from assistant.embeddings import Embedder
    from assistant.memory_factory import create_memory

    memory = create_memory()
    embedder = Embedder()

    count = backfill_note_embeddings(
        memory, embedder, batch_size=args.batch_size, max_notes=args.limit
    )
    logger.info("Done. Embedded %d notes.", count)


if __name__ == "__main__":
    main()
