"""WebSocket endpoint for real-time audio streaming.

Handles ``WS /api/v1/stream``: a phone (or any browser-based) client streams raw
PCM audio captured at 16kHz/mono/int16 over binary WebSocket frames.  The server
buffers the incoming bytes, slices them into 512-sample frames (FRAME_SAMPLES),
and feeds them through the existing ``_SpeechSegmenter`` / ``VADProcessor`` state
machine for segmentation.  When a speech segment completes, the audio is
transcribed and processed through the ``Brain`` pipeline; results are streamed
back as JSON text frames.

WebSocket protocol
------------------
Client → Server:
  - Binary frames: raw PCM chunks (16kHz, mono, int16).  Any chunk size is
    accepted; the server splits into FRAME_SAMPLES (512-sample) windows.
  - Text frame: ``{"type": "control", "action": "stop"}`` — flush any
    in-progress segment and close.

Server → Client (JSON text frames):
  - ``{"type": "transcript", "text": "...", "conversation_id": "..."}``
  - ``{"type": "note", "summary": "...", "note_id": "..."}``
  - ``{"type": "action", "intent": "...", "details": {...},
       "action_id": "...", "needs_confirmation": true|false}``
  - ``{"type": "error", "message": "..."}``

Guardrail: ``confirm_first`` actions (including ``send_email``) are **never**
auto-executed here — they are reported with ``needs_confirmation: true`` so
that the human-in-the-loop REST PATCH ``/api/v1/actions/{id}`` flow handles
them.  The WebSocket endpoint deliberately has no execution path; all action
routing is deferred to the REST layer.

Dependency injection
--------------------
``make_transcriber``, ``make_brain``, and ``make_vad`` are module-level factory
functions.  Tests monkeypatch them to return fake objects, keeping the test
suite fully offline (no real Silero download, no live Gemini calls).  Memory
is injected via ``get_memory`` (same FastAPI dependency used by the REST
router) so that ``app.dependency_overrides[get_memory]`` works from tests.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from assistant.audio import AudioRecorder, _SpeechSegmenter
from assistant.config import config
from assistant.memory import Memory
from assistant.vad import FRAME_SAMPLES

from .routes import get_memory

if TYPE_CHECKING:
    from assistant.brain import Brain
    from assistant.transcriber import Transcriber
    from assistant.vad import VADProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

stream_router = APIRouter(prefix="/api/v1")

# ---------------------------------------------------------------------------
# Factory functions — monkeypatched in tests to stay fully offline
# ---------------------------------------------------------------------------


def make_transcriber() -> Transcriber:
    """Return a real :class:`~assistant.transcriber.Transcriber` instance.

    Monkeypatch this function in tests to return a fake transcriber so that
    no live Gemini API calls are made::

        import assistant.api.stream as stream_mod
        stream_mod.make_transcriber = lambda: FakeTranscriber()
    """
    from assistant.transcriber import Transcriber

    return Transcriber()


def make_brain(memory: Memory) -> Brain:
    """Return a real :class:`~assistant.brain.Brain` instance wired to *memory*.

    Monkeypatch this function in tests to return a fake brain::

        import assistant.api.stream as stream_mod
        stream_mod.make_brain = lambda mem: FakeBrain(mem)
    """
    from assistant.brain import Brain

    return Brain(memory=memory)


def make_vad() -> VADProcessor:
    """Return a real :class:`~assistant.vad.VADProcessor` instance.

    Monkeypatch this function in tests to return a fake VAD so that the
    Silero model is never downloaded::

        import assistant.api.stream as stream_mod
        stream_mod.make_vad = lambda: FakeVAD()
    """
    from assistant.vad import VADProcessor

    return VADProcessor()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _send_json(ws: WebSocket, payload: dict) -> None:
    """Serialise *payload* to JSON and send it as a text frame."""
    await ws.send_text(json.dumps(payload))


async def _send_error(ws: WebSocket, message: str) -> None:
    """Send an ``error`` message frame to the client."""
    logger.warning("WebSocket error sent to client: %s", message)
    await _send_json(ws, {"type": "error", "message": message})


async def _run_pipeline(
    ws: WebSocket,
    wav_path: Path,
    memory: Memory,
    transcriber: Transcriber,
    brain: Brain,
) -> None:
    """Run transcription + brain processing for one completed WAV segment.

    1. Transcribe the WAV → send ``transcript`` message.
    2. Run ``brain.process(text)`` → persist conversation/note/actions.
    3. Look up the just-created conversation (newest in memory) to get IDs.
    4. Send ``note`` message if noteworthy.
    5. Send ``action`` messages for all pending actions — never execute them.

    Errors are caught per step and sent as ``error`` messages so the socket
    remains alive for subsequent segments.
    """
    # --- Transcription ---
    try:
        result = transcriber.transcribe(wav_path)
        text = result.text
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed for %s", wav_path)
        await _send_error(ws, f"Transcription failed: {exc}")
        return

    # --- Brain processing ---
    try:
        processing_result = brain.process(text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Brain processing failed for transcript: %r", text)
        await _send_error(ws, f"Processing failed: {exc}")
        return

    # Resolve the conversation_id: brain.process always saves a conversation;
    # the most recent one in memory is the one we just created.
    conversations = memory.get_recent_conversations(limit=1)
    conversation_id = conversations[0].id if conversations else ""

    # --- Send transcript message ---
    await _send_json(
        ws,
        {
            "type": "transcript",
            "text": text,
            "conversation_id": conversation_id,
        },
    )

    # --- Send note message if noteworthy ---
    if processing_result.is_noteworthy and conversation_id:
        notes = memory.get_notes_for_conversation(conversation_id)
        if notes:
            note = notes[0]
            await _send_json(
                ws,
                {
                    "type": "note",
                    "summary": note.summary,
                    "note_id": note.id,
                },
            )

    # --- Send action messages — report only, never execute ---
    # We use get_pending_actions() to find actions that survived the
    # confidence guardrail (high-confidence = status 'pending').
    # confirm_first actions (e.g. send_email) are reported with
    # needs_confirmation=True.  auto_execute actions are also reported
    # with needs_confirmation=False.  Neither is executed here; the
    # REST PATCH /actions/{id} flow handles execution so that the
    # no-auto-send-email guardrail is never bypassed.
    pending = memory.get_pending_actions()
    for action in pending:
        await _send_json(
            ws,
            {
                "type": "action",
                "intent": action.intent,
                "details": action.details,
                "action_id": action.id,
                "needs_confirmation": action.execution_mode == "confirm_first",
            },
        )


async def _flush_segmenter(
    segmenter: _SpeechSegmenter,
    ws: WebSocket,
    memory: Memory,
    transcriber: Transcriber,
    brain: Brain,
) -> None:
    """Force-complete any in-progress speech segment in *segmenter*.

    ``_SpeechSegmenter`` completes a segment only when trailing silence exceeds
    the threshold.  On ``control/stop`` (or disconnect) we synthesise that
    condition by repeatedly feeding silent frames until the segmenter fires.
    A maximum of ``_silence_frames + 1`` silent frames are fed to avoid an
    infinite loop if the segmenter is already in SILENCE state.
    """
    if not segmenter._in_speech:
        return  # nothing to flush

    # Feed silent frames until the segmenter closes the segment.
    silent_frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
    max_flush_frames = segmenter._silence_frames + 1
    for _ in range(max_flush_frames):
        wav_path = segmenter.process_frame(silent_frame)
        if wav_path is not None:
            await _run_pipeline(ws, wav_path, memory, transcriber, brain)
            return
    # If we ran out of frames and the segmenter never fired, call _finalise
    # directly (segment may be below min_speech_frames and thus discarded).
    segmenter._finalise_segment()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


def _check_ws_api_key(api_key: str | None) -> bool:
    """Return True if the WebSocket connection should be allowed.

    Mirrors the REST ``require_api_key`` dependency logic but for WebSocket:
    - If no keys are configured, auth is disabled and all connections are allowed.
    - Otherwise the provided *api_key* (from ``?api_key=`` query param or the
      ``X-API-Key`` header via ``Sec-WebSocket-Protocol`` — standard browsers
      cannot set custom headers on WebSocket, so the query param is the primary
      mechanism) must match one of the configured keys.

    Comparison is constant-time (``hmac.compare_digest``) to prevent timing
    attacks.
    """
    configured_keys: list[str] = config.api.api_keys
    if not configured_keys:
        return True  # Auth disabled.
    if api_key is None:
        return False
    return any(hmac.compare_digest(api_key, candidate) for candidate in configured_keys)


@stream_router.websocket("/stream")
async def websocket_stream(
    ws: WebSocket,
    api_key: str | None = None,
    memory: Memory = Depends(get_memory),  # noqa: B008
) -> None:
    """``WS /api/v1/stream`` — real-time audio streaming endpoint.

    Authentication
    --------------
    Browsers cannot set custom HTTP headers on WebSocket upgrades, so the API
    key is accepted as the ``?api_key=`` query parameter (e.g.
    ``ws://host/api/v1/stream?api_key=secret``).  The server also accepts the
    ``X-API-Key`` header when the client supports it (non-browser clients).
    When keys are configured and the key is absent or wrong, the WebSocket is
    accepted and then immediately closed with code 1008 (Policy Violation).
    When no keys are configured (default / local dev), all connections are
    allowed regardless of the ``api_key`` parameter.

    See module docstring for the full protocol specification.

    The endpoint:
    1. Accepts the connection.
    2. Checks API-key auth; closes with 1008 on failure.
    3. Constructs per-connection instances of VADProcessor, AudioRecorder,
       and _SpeechSegmenter.
    4. Loops, receiving frames:
       - Binary → accumulate into a byte buffer, slice into FRAME_SAMPLES
         windows, feed each window to the segmenter.
       - Text JSON ``control/stop`` → flush the in-progress segment, break.
    5. Handles ``WebSocketDisconnect`` gracefully (flush then close Memory).
    6. Wraps per-pipeline errors in ``error`` messages; the socket stays open
       for subsequent segments unless a disconnect is signalled.
    """
    await ws.accept()

    # --- API-key auth for WebSocket ---
    # The query-param key takes precedence; fall back to the X-API-Key header
    # when the client is not a browser and can set custom headers.
    effective_key = api_key or ws.headers.get("x-api-key")
    if not _check_ws_api_key(effective_key):
        logger.warning("WebSocket connection rejected: invalid or missing API key")
        await ws.close(code=1008, reason="Invalid or missing API key")
        return
    logger.info("WebSocket /api/v1/stream connected")

    transcriber = make_transcriber()
    brain = make_brain(memory)
    vad = make_vad()

    # AudioRecorder is needed by _SpeechSegmenter for _wav_path / _write_wav.
    recorder = AudioRecorder()
    segmenter = _SpeechSegmenter(recorder=recorder, vad=vad)

    # Byte buffer for accumulating incoming PCM chunks before slicing into
    # FRAME_SAMPLES windows (each window = FRAME_SAMPLES * 2 bytes for int16).
    pcm_buffer = bytearray()
    frame_bytes = FRAME_SAMPLES * 2  # int16 = 2 bytes per sample

    try:
        while True:
            message = await ws.receive()

            # --- Binary frame: PCM audio ---
            if "bytes" in message and message["bytes"] is not None:
                pcm_buffer.extend(message["bytes"])

                # Process all complete frames from the buffer.
                while len(pcm_buffer) >= frame_bytes:
                    frame_raw = bytes(pcm_buffer[:frame_bytes])
                    del pcm_buffer[:frame_bytes]

                    frame = np.frombuffer(frame_raw, dtype=np.int16)
                    try:
                        wav_path = segmenter.process_frame(frame)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Segmenter error on frame")
                        await _send_error(ws, f"Segmenter error: {exc}")
                        continue

                    if wav_path is not None:
                        await _run_pipeline(ws, wav_path, memory, transcriber, brain)

            # --- Text frame: control message ---
            elif "text" in message and message["text"] is not None:
                raw_text = message["text"]
                try:
                    ctrl = json.loads(raw_text)
                except (json.JSONDecodeError, ValueError) as exc:
                    await _send_error(ws, f"Malformed control message: {exc}")
                    continue

                if not isinstance(ctrl, dict):
                    await _send_error(ws, "Control message must be a JSON object")
                    continue

                msg_type = ctrl.get("type")
                action = ctrl.get("action")

                if msg_type == "control" and action == "stop":
                    logger.info("WebSocket received control/stop — flushing segment")
                    await _flush_segmenter(segmenter, ws, memory, transcriber, brain)
                    break
                else:
                    # Unknown control message — report error and continue.
                    await _send_error(
                        ws,
                        f"Unknown control message: type={msg_type!r} action={action!r}",
                    )

            # --- WebSocket close frame (receive() returns {"type": "websocket.disconnect"}) ---
            elif message.get("type") == "websocket.disconnect":
                logger.info("WebSocket client disconnected (disconnect frame)")
                await _flush_segmenter(segmenter, ws, memory, transcriber, brain)
                return  # Connection already closed — do not call ws.close() again.

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected — flushing in-progress segment")
        try:
            await _flush_segmenter(segmenter, ws, memory, transcriber, brain)
        except Exception:  # noqa: BLE001
            logger.exception("Error flushing segment after disconnect")
        return  # Connection already closed — do not call ws.close() again.
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in WebSocket handler")
        with contextlib.suppress(Exception):
            await _send_error(ws, f"Internal error: {exc}")
        return  # Skip the close call — state is uncertain.
    finally:
        logger.info("WebSocket /api/v1/stream closing")

    # Normal exit path: control/stop was received.  Explicitly close the
    # WebSocket so the client receives the close handshake and its
    # receive_*() calls unblock immediately.
    with contextlib.suppress(Exception):
        await ws.close()
    logger.info("WebSocket /api/v1/stream closed")
    # Memory is closed by the get_memory dependency's finally block.
