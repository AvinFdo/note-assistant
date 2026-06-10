"""Tests for the WebSocket /api/v1/stream endpoint (task 2.1.2).

All tests are fully offline — no real audio hardware, no live API calls, no
Silero model download.  Dependencies are replaced with lightweight fakes:

- ``FakeVAD``         — scripted booleans; returns True for the first N calls.
- ``FakeTranscriber`` — returns a fixed transcript without hitting any API.
- ``FakeBrain``       — marks segments noteworthy, saves a note + send_email
                        action, and returns a fixed ``ProcessingResult``.

Receive strategy
----------------
``TestClient.websocket_connect()`` is synchronous.  After sending
``control/stop``, the server breaks its receive loop and calls ``ws.close()``,
which completes the handshake.  Tests therefore:

1. Send audio bytes.
2. Collect the expected JSON messages with ``ws.receive_text()`` (blocks until
   the server sends each one — there is no race because processing is
   synchronous in the test event loop).
3. Send ``control/stop``.
4. Exit the ``with`` block (no ``receive_text()`` after stop — the server is
   closing at that point and calling ``receive_text()`` would deadlock in
   Starlette's TestClient under anyio).
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import assistant.api.stream as stream_mod
from assistant.api.app import app
from assistant.api.routes import get_memory
from assistant.brain import ActionItem, ProcessingResult
from assistant.memory import Memory
from assistant.vad import FRAME_SAMPLES, SAMPLE_RATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default config: silence_duration_ms=1500, min_speech_duration_ms=250
# At 16kHz / 480-sample frames: ~33.3 fps
# silence_frames ≈ 50, min_speech_frames ≈ 8
_SPEECH_FRAMES = 15  # > min_speech_frames threshold
_SILENCE_FRAMES = 55  # > silence_frames threshold


# ---------------------------------------------------------------------------
# Fake components
# ---------------------------------------------------------------------------


class FakeVAD:
    """Returns True for the first *n_speech* calls, False thereafter."""

    def __init__(self, n_speech: int = _SPEECH_FRAMES) -> None:
        self._n_speech = n_speech
        self._count = 0

    def process_frame(self, frame: np.ndarray) -> bool:  # noqa: ARG002
        result = self._count < self._n_speech
        self._count += 1
        return result


class _FakeTranscriptionResult:
    def __init__(self, text: str) -> None:
        self.text = text
        self.confidence = 1.0
        self.duration_ms = 1000


class FakeTranscriber:
    """Returns a fixed transcript without hitting any API."""

    def __init__(self, text: str = "hello world this is a test sentence") -> None:
        self._text = text
        self.call_count = 0

    def transcribe(self, audio_path: Path) -> _FakeTranscriptionResult:  # noqa: ARG002
        self.call_count += 1
        return _FakeTranscriptionResult(self._text)


class FakeBrain:
    """Saves note + send_email (confirm_first) action and marks as noteworthy."""

    def __init__(self, memory: Memory, noteworthy: bool = True) -> None:
        self._memory = memory
        self._noteworthy = noteworthy

    def process(self, transcript: str) -> ProcessingResult:
        cid = self._memory.save_conversation(transcript)
        if self._noteworthy:
            self._memory.save_note(cid, f"Summary: {transcript[:40]}", is_noteworthy=True)
            self._memory.save_action(
                cid,
                "send_email",
                {"recipient": "boss@example.com", "subject": "Test"},
                "confirm_first",
            )
            return ProcessingResult(
                is_noteworthy=True,
                summary_note=f"Summary: {transcript[:40]}",
                actions=[
                    ActionItem(
                        intent="send_email",
                        confidence=0.9,
                        details={"recipient": "boss@example.com"},
                    )
                ],
            )
        return ProcessingResult(is_noteworthy=False, summary_note="", actions=[])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio_bytes(
    speech_frames: int = _SPEECH_FRAMES,
    silence_frames: int = _SILENCE_FRAMES,
) -> bytes:
    """Build raw int16 PCM: speech_frames of noise followed by silence_frames of zeros.

    The FakeVAD returns True for the first *speech_frames* calls, then False,
    so feeding this payload drives the _SpeechSegmenter to complete a segment.
    """
    rng = np.random.default_rng(42)
    speech = rng.integers(-1000, 1000, size=speech_frames * FRAME_SAMPLES, dtype=np.int16)
    silence = np.zeros(silence_frames * FRAME_SAMPLES, dtype=np.int16)
    return np.concatenate([speech, silence]).tobytes()


def _make_wav(path: Path) -> None:
    """Write a minimal valid WAV to *path* so the FakeTranscriber can open it."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem():
    """In-memory SQLite Memory, closed after the test."""
    m = Memory(db_path=":memory:")
    yield m
    m.close()


@pytest.fixture()
def fake_transcriber():
    return FakeTranscriber()


@pytest.fixture()
def stream_client(mem, fake_transcriber, tmp_path):
    """TestClient with all external dependencies replaced by fakes.

    - Memory: in-memory SQLite via FastAPI dependency_overrides.
    - VAD: FakeVAD (scripted True/False — no Silero download).
    - Transcriber: FakeTranscriber (fixed text — no Gemini calls).
    - Brain: FakeBrain (saves note + send_email — no Gemini calls).
    - AudioRecorder: recordings dir patched to tmp_path; _wav_path returns
      a pre-written WAV in tmp_path so _write_wav_mono writes correctly.
    """

    def override_get_memory():
        yield mem

    app.dependency_overrides[get_memory] = override_get_memory

    fake_vad = FakeVAD(n_speech=_SPEECH_FRAMES)
    wav_path = tmp_path / "test_segment.wav"
    _make_wav(wav_path)  # pre-write so the file exists if transcriber reads it

    with (  # noqa: SIM117
        patch.object(stream_mod, "make_transcriber", return_value=fake_transcriber),
        patch.object(stream_mod, "make_brain", side_effect=lambda m: FakeBrain(m)),
        patch.object(stream_mod, "make_vad", return_value=fake_vad),
        patch(
            "assistant.audio.AudioRecorder._ensure_recordings_dir",
            lambda self: None,
        ),
        patch(
            "assistant.audio.AudioRecorder._wav_path",
            lambda self: wav_path,
        ),
    ):
        with TestClient(app) as c:
            yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWebSocketStream:
    """Core protocol and guardrail tests."""

    def test_connect_and_stop_no_audio(self, stream_client):
        """Control/stop with no buffered audio closes the connection cleanly."""
        with stream_client.websocket_connect("/api/v1/stream") as ws:
            ws.send_text(json.dumps({"type": "control", "action": "stop"}))
            # Server closes; exiting with-block sends the close handshake.

    def test_audio_produces_transcript_note_action(self, stream_client):
        """Streaming audio triggers the full pipeline; server emits 3 message types."""
        audio = _make_audio_bytes()

        with stream_client.websocket_connect("/api/v1/stream") as ws:
            ws.send_bytes(audio)

            # Receive in order: transcript → note → action.
            transcript_msg = json.loads(ws.receive_text())
            note_msg = json.loads(ws.receive_text())
            action_msg = json.loads(ws.receive_text())

            ws.send_text(json.dumps({"type": "control", "action": "stop"}))

        # --- transcript ---
        assert transcript_msg["type"] == "transcript"
        assert transcript_msg["text"] == "hello world this is a test sentence"
        assert transcript_msg["conversation_id"] != ""

        # --- note ---
        assert note_msg["type"] == "note"
        assert "summary" in note_msg
        assert "note_id" in note_msg

        # --- action ---
        assert action_msg["type"] == "action"
        assert action_msg["intent"] == "send_email"
        assert "action_id" in action_msg
        assert "details" in action_msg
        assert "needs_confirmation" in action_msg

    def test_send_email_needs_confirmation_true(self, stream_client):
        """send_email action must always be reported with needs_confirmation=True."""
        audio = _make_audio_bytes()

        with stream_client.websocket_connect("/api/v1/stream") as ws:
            ws.send_bytes(audio)
            _transcript = json.loads(ws.receive_text())
            _note = json.loads(ws.receive_text())
            action_msg = json.loads(ws.receive_text())
            ws.send_text(json.dumps({"type": "control", "action": "stop"}))

        assert action_msg["intent"] == "send_email"
        assert action_msg["needs_confirmation"] is True, (
            f"send_email must have needs_confirmation=True, got: {action_msg}"
        )

    def test_send_email_not_executed(self, stream_client, mem):
        """GUARDRAIL: send_email action must never be auto-executed by the endpoint.

        After streaming, all send_email actions in memory must have status
        'pending' (or 'confirmed', if manually updated) — never 'executed'.
        The execution must go through the REST PATCH /api/v1/actions/{id} flow.
        """
        audio = _make_audio_bytes()

        with stream_client.websocket_connect("/api/v1/stream") as ws:
            ws.send_bytes(audio)
            for _ in range(3):  # drain transcript, note, action
                ws.receive_text()
            ws.send_text(json.dumps({"type": "control", "action": "stop"}))

        all_actions = mem.get_recent_actions(limit=100)
        email_actions = [a for a in all_actions if a.intent == "send_email"]
        assert email_actions, "Expected at least one send_email action persisted"
        for act in email_actions:
            assert act.status != "executed", (
                f"send_email action {act.id} was auto-executed — guardrail violated!"
            )

    def test_malformed_json_emits_error_stays_alive(self, stream_client):
        """Malformed text frame → error message; connection stays alive for more messages."""
        with stream_client.websocket_connect("/api/v1/stream") as ws:
            ws.send_text("not valid json }{")
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"
            assert "message" in err

            # Still alive — can send stop cleanly.
            ws.send_text(json.dumps({"type": "control", "action": "stop"}))

    def test_unknown_control_message_emits_error_stays_alive(self, stream_client):
        """Unknown control type → error message; connection stays alive."""
        with stream_client.websocket_connect("/api/v1/stream") as ws:
            ws.send_text(json.dumps({"type": "bogus_type", "action": "do_nothing"}))
            err = json.loads(ws.receive_text())
            assert err["type"] == "error"

            ws.send_text(json.dumps({"type": "control", "action": "stop"}))

    def test_partial_frame_bytes_buffered_no_error(self, stream_client):
        """Sub-frame PCM bytes (<960 bytes) are buffered silently — no error emitted.

        The buffer accumulates incomplete frame data and waits for more bytes
        before processing.  This is normal behaviour when audio arrives in
        small UDP-like chunks.
        """
        with stream_client.websocket_connect("/api/v1/stream") as ws:
            # 3 bytes is far below one frame (960 bytes for int16).
            ws.send_bytes(b"\x01\x02\x03")
            # No error message expected; stop cleanly.
            ws.send_text(json.dumps({"type": "control", "action": "stop"}))


class TestStreamOffline:
    """Import-level checks — confirm no live network activity on import."""

    def test_import_no_live_api(self):
        """Importing stream module must not trigger any network activity."""
        import importlib

        importlib.reload(stream_mod)

    def test_app_imports_cleanly(self):
        """``import assistant.api.app`` succeeds without live API calls."""
        import importlib

        import assistant.api.app  # noqa: F401

        importlib.reload(assistant.api.app)
