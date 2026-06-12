"""Factory that returns the configured storage backend.

Consumers call :func:`create_memory` instead of constructing ``Memory`` directly,
so flipping ``config.memory.backend`` between ``"sqlite"`` and ``"firestore"``
swaps the persistence layer with no other code changes.  Both backends expose the
same public interface and return the same dataclasses.
"""

from __future__ import annotations

from assistant.config import config as _default_config
from assistant.memory import Memory


def create_memory(backend: str | None = None) -> Memory:
    """Return a storage backend instance based on *backend* (default: config).

    Parameters
    ----------
    backend:
        ``"sqlite"`` or ``"firestore"``.  When *None*, read
        ``config.memory.backend``.

    Returns
    -------
    A ``Memory`` (SQLite) or ``FirestoreMemory`` instance.  The return type is
    annotated as ``Memory`` because the two are interchangeable through the same
    public interface; ``FirestoreMemory`` is not a subclass but is a structural
    drop-in.

    Raises
    ------
    ValueError
        If *backend* is not a recognised value.
    """
    resolved = (backend or _default_config.memory.backend or "sqlite").lower()

    if resolved == "sqlite":
        return Memory()
    if resolved == "firestore":
        # Imported lazily so the google-cloud-firestore dependency is only
        # touched when the Firestore backend is actually selected.
        from assistant.firestore_memory import FirestoreMemory

        return FirestoreMemory()  # type: ignore[return-value]

    raise ValueError(f"Unknown memory backend: {resolved!r}. Expected 'sqlite' or 'firestore'.")
