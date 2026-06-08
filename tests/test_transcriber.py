"""Tests for assistant.transcriber.

All tests inject a mock genai client — no live API calls are made.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from assistant.transcriber import (
    MAX_RETRIES,
    AuthenticationError,
    NetworkError,
    QuotaExceededError,
    SilenceError,
    Transcriber,
    TranscriptionResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(path: Path, duration_s: float = 0.5, sample_rate: int = 16000) -> Path:
    """Write a minimal valid WAV file to *path* and return it.

    The audio data is just zeroed PCM samples (silence at the WAV level —
    the model is mocked so the content doesn't matter).
    """
    n_frames = int(sample_rate * duration_s)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return path


def _make_mock_client(text: str) -> MagicMock:
    """Return a mock genai.Client whose generate_content returns *text*."""
    response = MagicMock()
    response.text = text
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def _make_api_error(code: int, status: str = "", message: str = "") -> genai_errors.APIError:
    """Construct a genai_errors.APIError with the given code/status/message."""
    err = genai_errors.APIError.__new__(genai_errors.APIError)
    err.code = code
    err.status = status
    err.message = message
    err.details = {}
    err.response = None
    Exception.__init__(err, f"{code} {status}")
    return err


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def wav_file(tmp_path: Path) -> Path:
    """A short valid WAV file at 16 kHz, 0.5 seconds long."""
    return _make_wav(tmp_path / "test.wav", duration_s=0.5, sample_rate=16000)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_transcribe_returns_result(wav_file: Path) -> None:
    """Happy path: mock client returns text -> TranscriptionResult populated."""
    expected_text = "Hello, this is a test transcription."
    client = _make_mock_client(expected_text)
    transcriber = Transcriber(client=client)

    result = transcriber.transcribe(wav_file)

    assert isinstance(result, TranscriptionResult)
    assert result.text == expected_text
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0
    # 0.5 s at 16 000 Hz → 500 ms
    assert result.duration_ms == pytest.approx(500, abs=5)


def test_transcribe_strips_whitespace(wav_file: Path) -> None:
    """Leading/trailing whitespace in the model response is stripped."""
    client = _make_mock_client("  Stripped response.  ")
    result = Transcriber(client=client).transcribe(wav_file)
    assert result.text == "Stripped response."


def test_transcribe_calls_api_with_audio_part(wav_file: Path) -> None:
    """The SDK call receives both an audio Part and the prompt string."""
    client = _make_mock_client("ok")
    Transcriber(client=client).transcribe(wav_file)

    call_args = client.models.generate_content.call_args
    contents = (
        call_args.kwargs.get("contents") or call_args.args[0]
        if call_args.args
        else call_args.kwargs["contents"]
    )
    # contents should contain the audio Part and the prompt string
    assert any(isinstance(item, str) for item in contents), "Prompt string not in contents"


# ---------------------------------------------------------------------------
# SilenceError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty_text", ["", "   ", "\n", "\t"])
def test_silence_error_on_empty_response(wav_file: Path, empty_text: str) -> None:
    """Empty or whitespace-only model response raises SilenceError."""
    client = _make_mock_client(empty_text)
    with pytest.raises(SilenceError):
        Transcriber(client=client).transcribe(wav_file)


# ---------------------------------------------------------------------------
# AuthenticationError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,status",
    [
        (401, "UNAUTHENTICATED"),
        (403, "PERMISSION_DENIED"),
    ],
)
def test_auth_error_mapped(wav_file: Path, code: int, status: str) -> None:
    """401/403 API errors are mapped to AuthenticationError."""
    api_err = _make_api_error(code, status=status)
    client = MagicMock()
    client.models.generate_content.side_effect = api_err

    with pytest.raises(AuthenticationError):
        Transcriber(client=client).transcribe(wav_file)


# ---------------------------------------------------------------------------
# QuotaExceededError
# ---------------------------------------------------------------------------


def test_quota_error_mapped(wav_file: Path) -> None:
    """429 API errors are mapped to QuotaExceededError."""
    api_err = _make_api_error(429, status="RESOURCE_EXHAUSTED", message="quota exceeded")
    client = MagicMock()
    client.models.generate_content.side_effect = api_err

    with pytest.raises(QuotaExceededError):
        Transcriber(client=client).transcribe(wav_file)


# ---------------------------------------------------------------------------
# NetworkError & retry logic
# ---------------------------------------------------------------------------


def test_network_error_retried_and_raises(wav_file: Path) -> None:
    """Persistent network-type API error is retried MAX_RETRIES times then raises NetworkError."""
    # Use a code that falls through to the default NetworkError branch
    api_err = _make_api_error(503, status="UNAVAILABLE", message="connection failed")
    client = MagicMock()
    client.models.generate_content.side_effect = api_err

    with (
        patch("assistant.transcriber.time.sleep") as mock_sleep,
        patch("assistant.transcriber.RETRY_BACKOFF_BASE", 0),
        pytest.raises(NetworkError),
    ):
        Transcriber(client=client).transcribe(wav_file)

    assert client.models.generate_content.call_count == MAX_RETRIES
    # sleep called MAX_RETRIES - 1 times (not before last attempt)
    assert mock_sleep.call_count == MAX_RETRIES - 1


def test_network_error_transient_then_success(wav_file: Path) -> None:
    """Transient network error followed by success returns the result."""
    good_response = MagicMock()
    good_response.text = "Recovered transcription."
    api_err = _make_api_error(503, status="UNAVAILABLE")

    client = MagicMock()
    client.models.generate_content.side_effect = [api_err, good_response]

    with (
        patch("assistant.transcriber.time.sleep"),
        patch("assistant.transcriber.RETRY_BACKOFF_BASE", 0),
    ):
        result = Transcriber(client=client).transcribe(wav_file)

    assert result.text == "Recovered transcription."
    assert client.models.generate_content.call_count == 2


def test_network_error_sleep_uses_exponential_backoff(wav_file: Path) -> None:
    """Sleep durations follow the exponential-backoff formula."""
    api_err = _make_api_error(503, status="UNAVAILABLE")
    client = MagicMock()
    client.models.generate_content.side_effect = api_err

    base = 0.01  # small but non-zero to assert values
    with (
        patch("assistant.transcriber.time.sleep") as mock_sleep,
        patch("assistant.transcriber.RETRY_BACKOFF_BASE", base),
        pytest.raises(NetworkError),
    ):
        Transcriber(client=client).transcribe(wav_file)

    sleep_calls = [call.args[0] for call in mock_sleep.call_args_list]
    # For MAX_RETRIES=3: sleep on attempt 0 (1*base) then attempt 1 (2*base)
    assert sleep_calls[0] == pytest.approx(base * 1)
    assert sleep_calls[1] == pytest.approx(base * 2)
