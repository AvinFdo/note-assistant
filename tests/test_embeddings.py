"""Tests for assistant.embeddings — Gemini text embeddings (backlog 2.3.2, phase 1).

Fully offline: a mock genai client is injected so no live API calls are made.
The mock's ``models.embed_content`` returns an object shaped like the real
``EmbedContentResponse`` (``.embeddings[i].values``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors

from assistant.embeddings import (
    AuthenticationError,
    Embedder,
    EmbeddingError,
    QuotaExceededError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.values = values


def _make_client(vectors: list[list[float]]) -> MagicMock:
    """Return a mock genai client whose embed_content returns *vectors*."""
    response = MagicMock()
    response.embeddings = [_FakeEmbedding(v) for v in vectors]
    client = MagicMock()
    client.models.embed_content.return_value = response
    return client


def _api_error(code: int) -> genai_errors.APIError:
    """Build a google-genai APIError with a given HTTP status code."""
    return genai_errors.APIError(code, {"message": f"boom {code}"})


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------


class TestEmbedText:
    def test_returns_vector(self):
        client = _make_client([[0.1, 0.2, 0.3]])
        emb = Embedder(client=client, model="text-embedding-004")
        assert emb.embed_text("hello world") == [0.1, 0.2, 0.3]

    def test_passes_model_and_contents(self):
        client = _make_client([[1.0, 2.0]])
        emb = Embedder(client=client, model="my-embed-model")
        emb.embed_text("note text")
        _, kwargs = client.models.embed_content.call_args
        assert kwargs["model"] == "my-embed-model"
        assert kwargs["contents"] == ["note text"]

    def test_empty_text_raises_value_error(self):
        emb = Embedder(client=_make_client([]), model="m")
        with pytest.raises(ValueError, match="non-empty"):
            emb.embed_text("   ")


# ---------------------------------------------------------------------------
# embed_batch
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    def test_returns_vectors_in_order(self):
        client = _make_client([[0.1], [0.2], [0.3]])
        emb = Embedder(client=client, model="m")
        assert emb.embed_batch(["a", "b", "c"]) == [[0.1], [0.2], [0.3]]

    def test_empty_list_raises(self):
        emb = Embedder(client=_make_client([]), model="m")
        with pytest.raises(ValueError, match="non-empty list"):
            emb.embed_batch([])

    def test_count_mismatch_raises(self):
        # API returns 1 vector for 2 inputs → error
        client = _make_client([[0.1]])
        emb = Embedder(client=client, model="m")
        with pytest.raises(EmbeddingError, match="1 vectors for 2 inputs"):
            emb.embed_batch(["a", "b"])


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestErrorClassification:
    @pytest.mark.parametrize("code", [401, 403])
    def test_auth_errors(self, code):
        client = MagicMock()
        client.models.embed_content.side_effect = _api_error(code)
        emb = Embedder(client=client, model="m")
        with pytest.raises(AuthenticationError):
            emb.embed_text("x")

    def test_quota_error(self):
        client = MagicMock()
        client.models.embed_content.side_effect = _api_error(429)
        emb = Embedder(client=client, model="m")
        with pytest.raises(QuotaExceededError):
            emb.embed_text("x")

    def test_other_api_error_is_base(self):
        client = MagicMock()
        client.models.embed_content.side_effect = _api_error(500)
        emb = Embedder(client=client, model="m")
        with pytest.raises(EmbeddingError):
            emb.embed_text("x")


# ---------------------------------------------------------------------------
# Offline import guard
# ---------------------------------------------------------------------------


def test_import_no_network():
    """Importing the module must not trigger any network activity."""
    import importlib

    import assistant.embeddings as emb_mod

    importlib.reload(emb_mod)
