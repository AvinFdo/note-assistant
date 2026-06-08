"""Shared pytest fixtures for the Avin test suite.

Provides the three reusable fixtures called for in task 1.8.2:

- ``memory``           — a :class:`~assistant.memory.Memory` backed by an
                         in-memory SQLite database, closed automatically after
                         the test (no files, no leaked connections).
- ``test_config``      — a :class:`~assistant.config.Config` loaded from a
                         throwaway YAML file, so tests can exercise the real
                         loader without touching the project default.
- ``mock_genai_client``— a factory that builds a stand-in for ``genai.Client``
                         whose ``models.generate_content`` returns a response
                         object with a caller-supplied ``.text``.  Lets
                         Brain/Transcriber tests run with zero live API calls.

All fixtures are import-safe and offline.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
import yaml

from assistant.config import Config, load_config
from assistant.memory import Memory


@pytest.fixture
def memory():
    """Yield an in-memory Memory instance, closing it after the test."""
    mem = Memory(db_path=":memory:")
    try:
        yield mem
    finally:
        mem.close()


_TEST_CONFIG_DATA = {
    "gcp": {"project_id": "test-project", "region": "us-test1"},
    "models": {"transcription": "gemini-test", "reasoning": "gemini-test"},
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "format": "int16",
        "recordings_dir": "recordings",
        "default_duration": 10,
    },
    "vad": {
        "enabled": True,
        "model": "silero",
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "silence_duration_ms": 1500,
        "buffer_duration_s": 3.0,
    },
    "memory": {
        "db_path": ":memory:",
        "context_window_size": 5,
        "max_context_tokens": 4000,
        "min_transcript_words": 10,
    },
    "actions": {
        "confidence_threshold": 0.7,
        "create_todo": {"mode": "auto_execute"},
        "send_email": {"mode": "confirm_first"},
        "add_calendar": {"mode": "confirm_first"},
        "research_topic": {"mode": "auto_execute"},
    },
    "integrations": {
        "obsidian": {"vault_path": "", "notes_folder": "assistant"},
        "google": {"oauth_credentials_path": ""},
    },
}


@pytest.fixture
def test_config(tmp_path) -> Config:
    """Load a Config from a throwaway YAML file written to *tmp_path*."""
    path = tmp_path / "test_config.yaml"
    path.write_text(yaml.dump(_TEST_CONFIG_DATA))
    return load_config(path)


@pytest.fixture
def mock_genai_client():
    """Return a factory that builds a mock genai client returning given JSON/text.

    Usage::

        client = mock_genai_client({"is_noteworthy": False, "summary_note": "", "actions": []})
        brain = Brain(memory, client=client)
    """

    def _make(response_payload) -> MagicMock:
        text = (
            response_payload if isinstance(response_payload, str) else json.dumps(response_payload)
        )
        client = MagicMock()
        client.models.generate_content.return_value = MagicMock(text=text)
        return client

    return _make
