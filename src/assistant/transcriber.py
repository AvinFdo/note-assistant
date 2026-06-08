"""Transcriber: audio-to-text via Gemini multimodal (google-genai).

Sends WAV audio bytes to a Gemini model using the inline/Part bytes API and
returns a structured TranscriptionResult.

Exception hierarchy
-------------------
TranscriptionError          — base for all transcription failures
  AuthenticationError       — ADC credentials missing, invalid, or expired (HTTP 401/403)
  QuotaExceededError        — API rate-limit or quota exhausted (HTTP 429)
  NetworkError              — connection/timeout reaching the API (retried up to 3 times)
  SilenceError              — model returned an empty / whitespace-only transcription

Retry policy
------------
NetworkError is retried up to 3 attempts with exponential backoff.
The backoff base (RETRY_BACKOFF_BASE) is a module-level constant so tests can
monkeypatch it to 0 and avoid slow sleeps.
"""

from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors

from assistant.config import config

# ---------------------------------------------------------------------------
# Retry configuration (patchable in tests)
# ---------------------------------------------------------------------------

#: Base sleep time in seconds for exponential backoff on NetworkError.
RETRY_BACKOFF_BASE: float = 1.0

#: Maximum number of attempts (including the first) for NetworkError.
MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TranscriptionError(Exception):
    """Base exception for all transcription failures."""


class AuthenticationError(TranscriptionError):
    """ADC credentials are missing, invalid, or expired (HTTP 401/403)."""


class QuotaExceededError(TranscriptionError):
    """API rate-limit or quota was exhausted (HTTP 429 / RESOURCE_EXHAUSTED)."""


class NetworkError(TranscriptionError):
    """The API could not be reached due to a connection or timeout error."""


class SilenceError(TranscriptionError):
    """The model returned an empty transcription — audio contained no detectable speech."""


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass
class TranscriptionResult:
    """Structured result from a single transcription call.

    Attributes:
        text:         The transcribed speech as plain text.
        confidence:   A float in [0.0, 1.0].  Gemini does not return a numeric
                      confidence for transcription, so this is always 1.0 as a
                      sensible default indicating the model produced a result.
        duration_ms:  Audio duration in milliseconds, computed from the WAV
                      header (frames / framerate * 1000).
    """

    text: str
    confidence: float
    duration_ms: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRANSCRIPTION_PROMPT = (
    "Transcribe the following audio exactly as spoken. "
    "Return only the transcription, no commentary."
)


def _wav_duration_ms(audio_path: Path) -> int:
    """Return the duration of a WAV file in milliseconds using the stdlib wave module."""
    with wave.open(str(audio_path), "rb") as wf:
        frames = wf.getnframes()
        framerate = wf.getframerate()
    return int(frames / framerate * 1000)


def _classify_api_error(exc: genai_errors.APIError) -> TranscriptionError:
    """Map a google-genai APIError to our custom exception hierarchy.

    Returns the appropriate :class:`TranscriptionError` subclass instance
    with ``__cause__`` set to *exc*.
    """
    code = exc.code or 0
    status = (exc.status or "").upper()
    message = (exc.message or "").lower()

    # Authentication / authorisation errors
    if (
        code in (401, 403)
        or status in ("UNAUTHENTICATED", "PERMISSION_DENIED")
        or "credential" in message
        or "auth" in message
    ):
        mapped: TranscriptionError = AuthenticationError(str(exc))
        mapped.__cause__ = exc
        return mapped

    # Quota / rate-limit errors
    if code == 429 or status == "RESOURCE_EXHAUSTED" or "quota" in message or "rate" in message:
        mapped = QuotaExceededError(str(exc))
        mapped.__cause__ = exc
        return mapped

    # Default: treat other API errors as NetworkError (retryable)
    mapped = NetworkError(str(exc))
    mapped.__cause__ = exc
    return mapped


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class Transcriber:
    """Transcribes WAV audio files using Gemini multimodal via google-genai.

    Args:
        client: An optional pre-constructed ``genai.Client``.  When *None* the
                real Vertex AI client is built from :data:`assistant.config.config`.
                Pass a mock client in tests to avoid live API calls.
    """

    def __init__(self, client: genai.Client | None = None) -> None:
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

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe a WAV file and return a :class:`TranscriptionResult`.

        Args:
            audio_path: Path to a valid WAV file.

        Returns:
            :class:`TranscriptionResult` with ``text``, ``confidence``, and
            ``duration_ms`` fields populated.

        Raises:
            AuthenticationError: If ADC credentials are missing or invalid.
            QuotaExceededError:   If the API rate-limit / quota is exceeded.
            NetworkError:         If the API cannot be reached after
                                  :data:`MAX_RETRIES` attempts.
            SilenceError:         If the model returns an empty transcription.
            TranscriptionError:   For any other transcription failure.
        """
        audio_path = Path(audio_path)
        duration_ms = _wav_duration_ms(audio_path)
        audio_bytes = audio_path.read_bytes()

        text = self._call_with_retry(audio_bytes)

        if not text.strip():
            raise SilenceError(
                f"No speech detected in {audio_path.name}: model returned an empty transcription."
            )

        return TranscriptionResult(
            text=text.strip(),
            # Gemini does not return a numeric confidence for transcription;
            # 1.0 is used as a sensible default indicating a successful result.
            confidence=1.0,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_api(self, audio_bytes: bytes) -> str:
        """Send audio bytes to Gemini and return the raw text response."""
        audio_part = genai.types.Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/wav",
        )
        response = self._client.models.generate_content(
            model=config.models.transcription,
            contents=[audio_part, _TRANSCRIPTION_PROMPT],
        )
        return response.text or ""

    def _call_with_retry(self, audio_bytes: bytes) -> str:
        """Call the API with exponential backoff on :class:`NetworkError`."""
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._call_api(audio_bytes)
            except genai_errors.APIError as exc:
                mapped = _classify_api_error(exc)
                if isinstance(mapped, NetworkError):
                    last_exc = mapped
                    if attempt < MAX_RETRIES - 1:
                        sleep_secs = RETRY_BACKOFF_BASE * (2**attempt)
                        time.sleep(sleep_secs)
                    continue
                # Non-retryable API error — raise immediately
                raise mapped from exc
            except (OSError, ConnectionError, TimeoutError) as exc:
                net_err = NetworkError(str(exc))
                net_err.__cause__ = exc
                last_exc = net_err
                if attempt < MAX_RETRIES - 1:
                    sleep_secs = RETRY_BACKOFF_BASE * (2**attempt)
                    time.sleep(sleep_secs)
            except Exception as exc:
                raise TranscriptionError(f"Unexpected error during transcription: {exc}") from exc

        # All retries exhausted
        raise last_exc or NetworkError("Max retries exceeded with no response.")
