"""Embedder: text-to-vector embeddings via Gemini (google-genai).

Phase 1 of the scored-retrieval upgrade (backlog 2.3.2).  This module only
*produces* embedding vectors; the storage of those vectors on notes and the
recency/importance/relevance scoring that consumes them land in later phases.

The model is read from ``config.models.embedding`` (default ``text-embedding-004``)
and the client is built from the same Vertex AI / ADC config as the Transcriber
and Brain.  Pass a pre-constructed client in tests to stay fully offline.

Exception hierarchy
-------------------
EmbeddingError              — base for all embedding failures
  AuthenticationError       — ADC credentials missing/invalid (HTTP 401/403)
  QuotaExceededError        — API rate-limit or quota exhausted (HTTP 429)
  NetworkError              — connection/timeout reaching the API
"""

from __future__ import annotations

from google import genai
from google.genai import errors as genai_errors

from assistant.config import config

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class EmbeddingError(Exception):
    """Base exception for all Embedder failures."""


class AuthenticationError(EmbeddingError):
    """ADC credentials are missing, invalid, or expired (HTTP 401/403)."""


class QuotaExceededError(EmbeddingError):
    """The API rate-limit or quota was exhausted (HTTP 429)."""


class NetworkError(EmbeddingError):
    """The API could not be reached due to a connection or timeout error."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_api_error(exc: genai_errors.APIError) -> EmbeddingError:
    """Map a google-genai APIError to our custom exception hierarchy."""
    code = exc.code or 0
    message = exc.message or str(exc)

    if code in (401, 403):
        err: EmbeddingError = AuthenticationError(f"Embedding auth failed: {message}")
    elif code == 429:
        err = QuotaExceededError(f"Embedding quota exceeded: {message}")
    else:
        err = EmbeddingError(f"Embedding API error ({code}): {message}")
    err.__cause__ = exc
    return err


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class Embedder:
    """Produces dense embedding vectors for text via the Gemini embedding model.

    Args:
        client: An optional pre-constructed ``genai.Client``.  When *None* the
                real Vertex AI client is built from
                :data:`assistant.config.config`.  Pass a mock client in tests to
                avoid live API calls.
        model:  Optional model-name override.  Defaults to
                ``config.models.embedding``.
    """

    def __init__(self, client: genai.Client | None = None, model: str | None = None) -> None:
        self._model = model or config.models.embedding
        if client is not None:
            self._client = client
        else:
            self._client = genai.Client(
                vertexai=True,
                project=config.gcp.project_id,
                location=config.gcp.region,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        """Return the embedding vector for a single non-empty *text*.

        Args:
            text: The text to embed.  Must contain non-whitespace content.

        Returns:
            A list of floats — the embedding vector.

        Raises:
            ValueError:      If *text* is empty or whitespace-only.
            EmbeddingError:  (or a subclass) on API failure.
        """
        if not text or not text.strip():
            raise ValueError("Embedder.embed_text requires non-empty text.")
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for a batch of *texts* (order preserved).

        Batching is used by the one-off backfill of existing notes and keeps the
        per-call overhead amortised.

        Args:
            texts: A non-empty list of texts to embed.

        Returns:
            A list of embedding vectors, one per input text, in the same order.

        Raises:
            ValueError:      If *texts* is empty.
            EmbeddingError:  (or a subclass) on API failure, or if the API
                             returns a different number of embeddings than inputs.
        """
        if not texts:
            raise ValueError("Embedder.embed_batch requires a non-empty list.")

        try:
            response = self._client.models.embed_content(model=self._model, contents=texts)
        except genai_errors.APIError as exc:
            raise _classify_api_error(exc) from exc
        except (ConnectionError, TimeoutError) as exc:
            raise NetworkError(f"Could not reach the embedding API: {exc}") from exc

        embeddings = getattr(response, "embeddings", None)
        if not embeddings or len(embeddings) != len(texts):
            got = 0 if not embeddings else len(embeddings)
            raise EmbeddingError(f"Embedding API returned {got} vectors for {len(texts)} inputs.")

        return [list(item.values) for item in embeddings]
