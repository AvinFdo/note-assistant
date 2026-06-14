"""Purge expired notes per the retention policy (config.memory.retention).

Notes are given an ``expires_at`` at save time when ``note_days > 0`` (and their
importance is below ``keep_importance_above``).  This command deletes the ones
whose time is up.  Safe to run repeatedly (idempotent) and cron-friendly::

    python -m assistant.purge            # uses config.memory.backend

For the Firestore backend at scale, prefer enabling a native TTL policy on the
notes ``expires_at`` field; this command then serves as a manual top-up.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from assistant.memory_factory import create_memory

    memory = create_memory()
    removed = memory.purge_expired_notes()
    logger.info("Purged %d expired note(s).", removed)


if __name__ == "__main__":
    main()
