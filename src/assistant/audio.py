"""AudioRecorder: mic capture with configurable sample rate and device selection.

Provides the ``AudioRecorder`` class for fixed-duration recording via sounddevice
and ``list_devices()`` to enumerate available audio input devices.  All tunables
(sample_rate, channels, format, recordings_dir) are read from ``config.audio``
— nothing is hardcoded.

Exception hierarchy:
    AudioError         — base for all audio-related errors in this module.
    NoMicrophoneError  — raised when no input device is available.
    DeviceBusyError    — raised when the requested device is locked by another process.

For microphone permission denial on macOS, the builtin ``PermissionError`` is raised
directly (the OS surfaces this as a PortAudio error with message containing
"Permission denied").
"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from assistant.config import config

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
        """Return a timestamped WAV path inside the recordings directory."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._recordings_dir / f"recording_{ts}.wav"

    def _write_wav(self, path: Path, audio: np.ndarray) -> None:
        """Write *audio* to *path* as a 16-bit PCM WAV (or matching bit-depth)."""
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
            wf.setnchannels(self._channels)
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
