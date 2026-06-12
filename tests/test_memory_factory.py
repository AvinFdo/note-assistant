"""Tests for the storage-backend factory and the memory.backend config switch."""

from __future__ import annotations

import pytest

from assistant.config import config
from assistant.memory import Memory
from assistant.memory_factory import create_memory


def test_create_memory_sqlite_explicit():
    mem = create_memory("sqlite")
    assert isinstance(mem, Memory)
    mem.close()


def test_create_memory_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown memory backend"):
        create_memory("redis")


def test_create_memory_reads_config_default(monkeypatch):
    monkeypatch.setattr(config.memory, "backend", "sqlite")
    mem = create_memory()
    assert isinstance(mem, Memory)
    mem.close()


def test_create_memory_firestore_uses_firestore_backend(monkeypatch):
    """backend='firestore' constructs FirestoreMemory (without touching real GCP)."""
    monkeypatch.setattr(config.memory, "backend", "firestore")

    captured = {}

    class _StubFirestoreMemory:
        def __init__(self):
            captured["built"] = True

    # Patch the symbol imported lazily inside create_memory.
    import assistant.firestore_memory as fm

    monkeypatch.setattr(fm, "FirestoreMemory", _StubFirestoreMemory)

    mem = create_memory()
    assert captured.get("built") is True
    assert isinstance(mem, _StubFirestoreMemory)


def test_config_backend_default_is_sqlite(tmp_path):
    """A YAML without a backend key defaults to 'sqlite'."""
    from assistant.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("memory:\n  db_path: ':memory:'\n")
    cfg = load_config(p)
    assert cfg.memory.backend == "sqlite"


def test_config_backend_env_override(tmp_path, monkeypatch):
    """AVIN_MEMORY_BACKEND overrides the YAML value."""
    from assistant.config import load_config

    p = tmp_path / "c.yaml"
    p.write_text("memory:\n  backend: 'sqlite'\n")
    monkeypatch.setenv("AVIN_MEMORY_BACKEND", "firestore")
    cfg = load_config(p)
    assert cfg.memory.backend == "firestore"
