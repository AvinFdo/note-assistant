"""AudioRecorder: mic capture with configurable sample rate and device selection.

Provides the ``AudioRecorder`` class for fixed-duration recording via sounddevice
and ``list_devices()`` to enumerate available audio input devices.  All tunables
(sample_rate, channels, format, recordings_dir) are read from ``config.audio``
— nothing is hardcoded.

Also provides ``_SpeechSegmenter``, a pure (no I/O) state machine that consumes
audio frames one at a time and emits completed WAV paths whenever a speech segment
finishes.  ``AudioRecorder.listen_continuous`` is a thin loop that opens a
sounddevice ``InputStream`` and feeds each callback frame into a ``_SpeechSegmenter``
instance, invoking the caller's callback on each completed segment.

State machine (inside ``_SpeechSegmenter``):
  SILENCE → SPEECH  when VAD returns True
  SPEECH  → SILENCE when trailing silence exceeds ``config.vad.silence_duration_ms``
  Segment discarded if total speech frames < ``config.vad.min_speech_duration_ms``

Exception hierarchy:
    AudioError         — base for all audio-related errors in this module.
    NoMicrophoneError  — raised when no input device is available.
    DeviceBusyError    — raised when the requested device is locked by another process.

For microphone permission denial on macOS, the builtin ``PermissionError`` is raised
directly (the OS surfaces this as a PortAudio error with message containing
"Permission denied").
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from assistant.config import config
from assistant.vad import FRAME_SAMPLES, SAMPLE_RATE

if TYPE_CHECKING:
    from assistant.vad import VADProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class AudioError(Exception):
    """Base exception for all audio-related errors raised by this module."""


class NoMicrophoneError(AudioError):
    """Raised when no audio input device is available on this system.

    This can happen when:
    - Running in a headless/CI environment with no physical mic.
    - All microphones are disabled in the OS.
    - The sounddevice/PortAudio layer finds no input devices.

    Check ``list_devices()`` to see what devices PortAudio can see, and
    ensure your microphone is plugged in and not disabled in system settings.
    """


class DeviceBusyError(AudioError):
    """Raised when the audio input device is locked by another application.

    Close any other app that may be using the microphone (e.g. video
    conferencing, another recording program) and retry.
    """


# ---------------------------------------------------------------------------
# Dataclass for device info
# ---------------------------------------------------------------------------


@dataclass
class DeviceInfo:
    """Information about a single audio input device."""

    id: int
    name: str
    sample_rate: float
    max_input_channels: int = field(repr=False)


# ---------------------------------------------------------------------------
# _SpeechSegmenter — pure state machine, fully testable without hardware
# ---------------------------------------------------------------------------


class _SpeechSegmenter:
    """Frame-by-frame speech segmentation state machine.

    Consumes one audio frame at a time and writes a WAV file whenever a complete
    speech segment is detected.  There is **no I/O** other than writing the
    finished WAV — all microphone / sounddevice logic lives in
    ``AudioRecorder.listen_continuous``.

    State transitions
    -----------------
    SILENCE → SPEECH
        Triggered when ``vad.process_frame`` returns ``True``.
        The rolling pre-speech buffer is prepended to the new segment so that
        the beginnings of utterances are not lost.
    SPEECH → SILENCE
        Triggered when trailing silence (consecutive VAD-False frames) exceeds
        ``config.vad.silence_duration_ms``.  The segment is written to a WAV
        and the path is returned from ``process_frame``.  If the total speech
        (non-trailing-silence) frames are shorter than
        ``config.vad.min_speech_duration_ms`` the segment is silently discarded
        (returns ``None``) to avoid persisting blips.

    Parameters
    ----------
    recorder:
        The owning ``AudioRecorder`` instance.  Used to call ``_wav_path`` and
        ``_write_wav`` so that file naming and formatting are consistent with the
        rest of the recording pipeline.
    vad:
        An already-constructed ``VADProcessor`` (or a fake for tests).
    """

    def __init__(self, recorder: AudioRecorder, vad: VADProcessor) -> None:
        self._recorder = recorder
        self._vad = vad

        # Thresholds derived from config (convert ms → frame counts).
        frames_per_second = SAMPLE_RATE / FRAME_SAMPLES
        self._silence_frames: int = int(config.vad.silence_duration_ms / 1000.0 * frames_per_second)
        self._min_speech_frames: int = int(
            config.vad.min_speech_duration_ms / 1000.0 * frames_per_second
        )
        buffer_capacity: int = int(config.vad.buffer_duration_s * frames_per_second)

        # Rolling pre-speech buffer — deque with a fixed maximum length.
        self._pre_buffer: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, buffer_capacity)
        )

        # Mutable state.
        self._in_speech: bool = False
        self._segment_frames: list[np.ndarray] = []
        self._trailing_silence: int = 0  # consecutive silent frames at end of segment

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Path | None:
        """Process one audio frame and return a WAV path if a segment completed.

        Parameters
        ----------
        frame:
            A 1-D NumPy array of exactly ``FRAME_SAMPLES`` (512) samples.

        Returns
        -------
        Path | None
            The path to the written WAV file if this frame completed a speech
            segment, otherwise ``None``.
        """
        is_speech: bool = self._vad.process_frame(frame)

        if not self._in_speech:
            # ---- SILENCE state ----
            self._pre_buffer.append(frame)
            if is_speech:
                # Transition to SPEECH: prepend the rolling buffer then start
                # accumulating.
                self._in_speech = True
                self._segment_frames = list(self._pre_buffer)  # includes current frame
                self._trailing_silence = 0
                logger.debug(
                    "Speech segment started (pre-buffer: %d frames)", len(self._pre_buffer)
                )
        else:
            # ---- SPEECH state ----
            self._segment_frames.append(frame)
            if is_speech:
                self._trailing_silence = 0
            else:
                self._trailing_silence += 1
                if self._trailing_silence >= self._silence_frames:
                    # End of segment — evaluate and possibly write.
                    return self._finalise_segment()

        return None

    def reset(self) -> None:
        """Reset the segmenter to the initial SILENCE state."""
        self._in_speech = False
        self._segment_frames = []
        self._trailing_silence = 0
        self._pre_buffer.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _finalise_segment(self) -> Path | None:
        """Evaluate the accumulated segment and write a WAV if it's long enough.

        The segment is considered complete regardless; state is reset to SILENCE.

        Returns
        -------
        Path | None
            Written WAV path, or ``None`` if the segment was discarded.
        """
        # Count only the speech frames (exclude trailing silence frames).
        speech_only_frames = len(self._segment_frames) - self._trailing_silence
        result: Path | None = None

        if speech_only_frames >= self._min_speech_frames:
            # Combine all accumulated frames (including trailing silence — it's
            # part of the natural cadence of speech).
            audio = np.concatenate(self._segment_frames, axis=0)
            wav_path = self._recorder._wav_path()
            self._recorder._ensure_recordings_dir()
            # listen_continuous always captures mono (channels=1); write mono
            # regardless of recorder._channels to keep the WAV valid.
            self._recorder._write_wav_mono(wav_path, audio)
            logger.info(
                "Segment complete — %d frames (%dms), saved to %s",
                len(self._segment_frames),
                len(self._segment_frames) * FRAME_SAMPLES / SAMPLE_RATE * 1000,
                wav_path,
            )
            result = wav_path
        else:
            logger.debug(
                "Segment discarded — only %d speech frames (min %d required)",
                speech_only_frames,
                self._min_speech_frames,
            )

        # Reset to SILENCE state.
        self._in_speech = False
        self._segment_frames = []
        self._trailing_silence = 0

        return result


# ---------------------------------------------------------------------------
# AudioRecorder
# ---------------------------------------------------------------------------


class AudioRecorder:
    """Records audio from the default microphone using settings from config.audio.

    Reads ``sample_rate``, ``channels``, ``format``, and ``recordings_dir``
    from the global config singleton — never from hardcoded constants.

    Usage::

        recorder = AudioRecorder()
        wav_path = recorder.record(duration_seconds=5)
        devices  = recorder.list_devices()
    """

    def __init__(self) -> None:
        self._sample_rate: int = config.audio.sample_rate
        self._channels: int = config.audio.channels
        self._format: str = config.audio.format
        self._recordings_dir: Path = Path(config.audio.recordings_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, duration_seconds: int) -> Path:
        """Record audio for *duration_seconds* and save to a WAV file.

        Args:
            duration_seconds: How long to record in seconds.

        Returns:
            Path to the saved WAV file inside ``config.audio.recordings_dir``.

        Raises:
            NoMicrophoneError: No input device found.
            PermissionError:   Microphone access denied (macOS permission gate).
            DeviceBusyError:   Device is in use by another application.
            AudioError:        Any other PortAudio / sounddevice failure.
        """
        self._ensure_recordings_dir()
        self._assert_input_device_available()

        num_samples = int(self._sample_rate * duration_seconds)
        dtype = self._numpy_dtype()

        logger.info(
            "Recording %ds at %dHz, %dch, dtype=%s",
            duration_seconds,
            self._sample_rate,
            self._channels,
            dtype,
        )

        try:
            audio_data: np.ndarray = sd.rec(
                frames=num_samples,
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype=dtype,
            )
            sd.wait()
        except sd.PortAudioError as exc:
            self._map_portaudio_error(exc)

        rms = self._compute_rms(audio_data)
        logger.info("Recording complete — RMS level: %.4f", rms)
        if rms < 1e-4:
            logger.warning(
                "Very low RMS (%.6f) — the recording may be silence or the mic gain is too low.",
                rms,
            )

        out_path = self._wav_path()
        self._write_wav(out_path, audio_data)
        logger.info("Saved recording to %s", out_path)
        return out_path

    def listen_continuous(
        self,
        callback: Callable[[Path], None],
        vad: VADProcessor | None = None,
    ) -> None:
        """Listen for speech continuously and invoke *callback* for each segment.

        Opens a sounddevice ``InputStream`` in callback (non-blocking) mode.
        Each incoming frame is passed to a ``_SpeechSegmenter`` which drives the
        VAD-based state machine.  Whenever the segmenter completes a segment it
        writes a WAV file and ``callback`` is invoked with the path.

        The method blocks until the user presses Ctrl+C (``KeyboardInterrupt``),
        at which point the stream is stopped and closed gracefully before
        returning.

        Parameters
        ----------
        callback:
            Called once per completed speech segment with the path to the
            written WAV file.
        vad:
            A ``VADProcessor`` instance to use.  If ``None`` a default one is
            constructed.  Provide a fake/mock here in tests so that the test
            suite remains fully offline.

        Notes
        -----
        The ``sounddevice.InputStream`` callback portion of this method is not
        unit-tested (it requires real audio hardware).  The segmentation logic
        is exercised independently via ``_SpeechSegmenter`` tests.
        """
        if vad is None:
            from assistant.vad import VADProcessor

            vad = VADProcessor()

        self._ensure_recordings_dir()
        segmenter = _SpeechSegmenter(recorder=self, vad=vad)

        # Use a queue to pass frames from the sounddevice callback thread
        # to the main thread so that WAV writing and the user callback run
        # outside the PortAudio callback (PortAudio callbacks must be fast
        # and must not block).
        frame_queue: queue.SimpleQueue[np.ndarray | None] = queue.SimpleQueue()

        def _sd_callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: sd.CallbackFlags,
        ) -> None:
            if status:
                logger.warning("sounddevice status: %s", status)
            # Copy to detach from the PortAudio-managed buffer.
            frame_queue.put(indata[:, 0].copy())

        logger.info(
            "Continuous capture started — sample_rate=%d, frame_samples=%d. Press Ctrl+C to stop.",
            self._sample_rate,
            FRAME_SAMPLES,
        )

        stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=_sd_callback,
        )

        # Signal the drain loop to exit after the stream closes.
        _sentinel: np.ndarray | None = None

        def _drain() -> None:
            while True:
                frame = frame_queue.get()
                if frame is None:
                    break
                wav_path = segmenter.process_frame(frame)
                if wav_path is not None:
                    try:
                        callback(wav_path)
                    except Exception:
                        logger.exception("Exception in listen_continuous callback")

        drain_thread = threading.Thread(target=_drain, daemon=True)

        try:
            with stream:
                drain_thread.start()
                stream.start()
                # Block until Ctrl+C.
                threading.Event().wait()
        except KeyboardInterrupt:
            logger.info("Continuous capture stopped by user (Ctrl+C).")
        finally:
            # Signal drain thread to finish processing then exit.
            frame_queue.put(_sentinel)
            drain_thread.join(timeout=5)

    def list_devices(self) -> list[dict]:
        """Return a list of available audio input devices.

        Each dict contains the keys: ``id``, ``name``, ``sample_rate``,
        ``max_input_channels``.

        Raises:
            AudioError: If the device query itself fails.
        """
        try:
            raw: list = sd.query_devices()  # type: ignore[assignment]
        except sd.PortAudioError as exc:
            raise AudioError(f"Failed to query audio devices: {exc}") from exc

        result: list[dict] = []
        for idx, dev in enumerate(raw):
            if dev.get("max_input_channels", 0) > 0:
                result.append(
                    {
                        "id": idx,
                        "name": dev.get("name", ""),
                        "sample_rate": dev.get("default_samplerate", 0.0),
                        "max_input_channels": dev.get("max_input_channels", 0),
                    }
                )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_recordings_dir(self) -> None:
        """Create the recordings directory if it doesn't exist yet."""
        self._recordings_dir.mkdir(parents=True, exist_ok=True)

    def _assert_input_device_available(self) -> None:
        """Raise ``NoMicrophoneError`` if no input device is available."""
        try:
            devices = sd.query_devices()
        except sd.PortAudioError as exc:
            raise NoMicrophoneError(
                "PortAudio failed to query devices — no audio system found on this machine. "
                "Install PortAudio or attach a microphone and retry."
            ) from exc

        has_input = any(d.get("max_input_channels", 0) > 0 for d in devices)
        if not has_input:
            raise NoMicrophoneError(
                "No audio input device was found. "
                "Plug in a microphone, enable it in your OS sound settings, and retry. "
                "In CI/headless environments set AVIN_AUDIO_RECORDINGS_DIR to a temp path "
                "and mock sounddevice."
            )

    def _numpy_dtype(self) -> str:
        """Map the config format string to a numpy dtype string."""
        _map = {
            "int16": "int16",
            "int32": "int32",
            "float32": "float32",
        }
        dtype = _map.get(self._format)
        if dtype is None:
            raise AudioError(
                f"Unsupported audio format '{self._format}'. Valid options: {sorted(_map)}"
            )
        return dtype

    @staticmethod
    def _compute_rms(audio: np.ndarray) -> float:
        """Compute root-mean-square level of *audio* as a float."""
        return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    def _wav_path(self) -> Path:
        """Return a timestamped WAV path inside the recordings directory.

        Includes microseconds so that rapid consecutive calls (e.g. multiple
        speech segments in the same second) produce distinct filenames.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self._recordings_dir / f"recording_{ts}.wav"

    def _write_wav_mono(self, path: Path, audio: np.ndarray) -> None:
        """Write a mono (1-channel) *audio* array to *path* as a WAV.

        Used by ``_SpeechSegmenter`` because ``listen_continuous`` always
        captures a single-channel stream regardless of ``config.audio.channels``.
        """
        self._write_wav_with_channels(path, audio, nchannels=1)

    def _write_wav(self, path: Path, audio: np.ndarray) -> None:
        """Write *audio* to *path* as a 16-bit PCM WAV (or matching bit-depth)."""
        self._write_wav_with_channels(path, audio, nchannels=self._channels)

    def _write_wav_with_channels(self, path: Path, audio: np.ndarray, nchannels: int) -> None:
        """Write *audio* to *path*, explicitly setting *nchannels* in the WAV header."""
        # Determine sample width from dtype
        dtype = audio.dtype
        if dtype == np.int16:
            sampwidth = 2
        elif dtype == np.int32:
            sampwidth = 4
        elif dtype == np.float32:
            # scipy.io.wavfile supports float32 natively; use wave for consistency
            # Convert to int16 for broadest WAV compatibility
            audio = (audio * 32767).astype(np.int16)
            sampwidth = 2
        else:
            sampwidth = 2

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio.tobytes())

    @staticmethod
    def _map_portaudio_error(exc: sd.PortAudioError) -> None:
        """Map a PortAudioError to a domain-specific exception and raise it.

        Always raises; never returns normally.
        """
        msg = str(exc).lower()
        if "permission" in msg or "access denied" in msg:
            raise PermissionError(
                "Microphone access was denied. "
                "On macOS go to System Settings → Privacy & Security → Microphone "
                "and allow this application."
            ) from exc
        if "device unavailable" in msg or "busy" in msg or "in use" in msg:
            raise DeviceBusyError(
                "The audio input device is busy or unavailable. "
                "Close other applications that may be using the microphone and retry."
            ) from exc
        if "no default input" in msg or "invalid device" in msg:
            raise NoMicrophoneError(
                "No default input device found. "
                "Attach a microphone and ensure it is set as the default input in your OS."
            ) from exc
        raise AudioError(f"Unexpected PortAudio error: {exc}") from exc
