"""Tests for assistant.audio — AudioRecorder class.

All tests use mocks for sounddevice so that no real hardware is required.
The CI environment has no microphone; these tests must pass in that context.
"""

from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np
import pytest

from assistant.audio import (
    AudioError,
    AudioRecorder,
    DeviceBusyError,
    NoMicrophoneError,
)
from assistant.config import config

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# A tiny fake recording: 100 samples, 1 channel, int16
FAKE_AUDIO = np.zeros((100, 1), dtype=np.int16)

# One fake device dict as sounddevice.query_devices() would return
FAKE_DEVICES = [
    {
        "name": "Built-in Microphone",
        "max_input_channels": 1,
        "max_output_channels": 0,
        "default_samplerate": 44100.0,
    },
    {
        "name": "MacBook Pro Speakers",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
]


@pytest.fixture()
def tmp_recordings(tmp_path: Path) -> Path:
    """Return a temporary directory to use as recordings_dir."""
    return tmp_path / "recordings"


@pytest.fixture()
def recorder(tmp_recordings: Path, monkeypatch: pytest.MonkeyPatch) -> AudioRecorder:
    """Return an AudioRecorder that writes into a temp directory."""
    monkeypatch.setattr(config.audio, "recordings_dir", str(tmp_recordings))
    return AudioRecorder()


# ---------------------------------------------------------------------------
# record() — happy path
# ---------------------------------------------------------------------------


class TestRecord:
    def test_writes_wav_file(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """record() must create a WAV file in the configured recordings dir."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "rec", lambda *a, **kw: FAKE_AUDIO)
        monkeypatch.setattr(sd, "wait", lambda: None)
        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        out = recorder.record(1)

        assert out.exists(), "WAV file should exist on disk"
        assert out.suffix == ".wav"

    def test_filename_matches_pattern(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Filename must be recording_YYYYMMDD_HHMMSS.wav."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "rec", lambda *a, **kw: FAKE_AUDIO)
        monkeypatch.setattr(sd, "wait", lambda: None)
        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        out = recorder.record(1)

        pattern = re.compile(r"recording_\d{8}_\d{6}\.wav")
        assert pattern.fullmatch(out.name), f"Unexpected filename: {out.name}"

    def test_wav_is_readable(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Written WAV must be readable by the stdlib wave module."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "rec", lambda *a, **kw: FAKE_AUDIO)
        monkeypatch.setattr(sd, "wait", lambda: None)
        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        out = recorder.record(1)

        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() >= 1
            assert wf.getsampwidth() > 0
            assert wf.getframerate() > 0

    def test_wav_uses_config_sample_rate(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The WAV frame rate must match config.audio.sample_rate, not a hardcoded value."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "rec", lambda *a, **kw: FAKE_AUDIO)
        monkeypatch.setattr(sd, "wait", lambda: None)
        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        out = recorder.record(1)

        with wave.open(str(out), "rb") as wf:
            assert wf.getframerate() == config.audio.sample_rate

    def test_wav_uses_config_channels(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The WAV channel count must match config.audio.channels."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "rec", lambda *a, **kw: FAKE_AUDIO)
        monkeypatch.setattr(sd, "wait", lambda: None)
        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        out = recorder.record(1)

        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() == config.audio.channels

    def test_saved_to_configured_dir(
        self,
        recorder: AudioRecorder,
        tmp_recordings: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Record must save into the directory specified by config.audio.recordings_dir."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "rec", lambda *a, **kw: FAKE_AUDIO)
        monkeypatch.setattr(sd, "wait", lambda: None)
        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        out = recorder.record(1)

        assert out.parent == tmp_recordings

    def test_creates_recordings_dir_if_missing(
        self,
        recorder: AudioRecorder,
        tmp_recordings: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The recordings directory should be created automatically if it does not exist."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "rec", lambda *a, **kw: FAKE_AUDIO)
        monkeypatch.setattr(sd, "wait", lambda: None)
        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        assert not tmp_recordings.exists()
        recorder.record(1)
        assert tmp_recordings.exists()

    def test_rec_called_with_config_params(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """sd.rec must be called with samplerate and channels from config, not hardcoded."""
        import sounddevice as sd

        captured: dict = {}

        def fake_rec(frames, samplerate, channels, dtype):
            captured["samplerate"] = samplerate
            captured["channels"] = channels
            return FAKE_AUDIO

        monkeypatch.setattr(sd, "rec", fake_rec)
        monkeypatch.setattr(sd, "wait", lambda: None)
        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        recorder.record(1)

        assert captured["samplerate"] == config.audio.sample_rate
        assert captured["channels"] == config.audio.channels


# ---------------------------------------------------------------------------
# list_devices()
# ---------------------------------------------------------------------------


class TestListDevices:
    def test_returns_list(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """list_devices() must return a list."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        result = recorder.list_devices()
        assert isinstance(result, list)

    def test_only_input_devices_returned(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Output-only devices must be excluded from the result."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        result = recorder.list_devices()
        # FAKE_DEVICES has 1 input device and 1 output-only device
        assert len(result) == 1
        assert result[0]["name"] == "Built-in Microphone"

    def test_device_dict_has_required_keys(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each device dict must contain id, name, sample_rate, max_input_channels."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        result = recorder.list_devices()
        for dev in result:
            assert "id" in dev
            assert "name" in dev
            assert "sample_rate" in dev
            assert "max_input_channels" in dev

    def test_id_is_an_integer(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Device id must be an integer index."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        result = recorder.list_devices()
        for dev in result:
            assert isinstance(dev["id"], int)

    def test_empty_when_no_input_devices(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """list_devices() returns an empty list when there are no input devices."""
        import sounddevice as sd

        output_only = [
            {
                "name": "Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            }
        ]
        monkeypatch.setattr(sd, "query_devices", lambda: output_only)

        result = recorder.list_devices()
        assert result == []


# ---------------------------------------------------------------------------
# NoMicrophoneError — raised when no input device is available
# ---------------------------------------------------------------------------


class TestNoMicrophoneError:
    def test_raised_when_no_input_devices(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """record() must raise NoMicrophoneError when no input device exists."""
        import sounddevice as sd

        output_only = [
            {
                "name": "Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            }
        ]
        monkeypatch.setattr(sd, "query_devices", lambda: output_only)

        with pytest.raises(NoMicrophoneError):
            recorder.record(1)

    def test_error_message_is_helpful(
        self,
        recorder: AudioRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """NoMicrophoneError message should explain how to resolve the issue."""
        import sounddevice as sd

        monkeypatch.setattr(sd, "query_devices", lambda: [])

        with pytest.raises(NoMicrophoneError, match=r"(?i)microphone|device|input"):
            recorder.record(1)

    def test_is_subclass_of_audio_error(self) -> None:
        """NoMicrophoneError must inherit from AudioError."""
        assert issubclass(NoMicrophoneError, AudioError)

    def test_device_busy_error_is_subclass_of_audio_error(self) -> None:
        """DeviceBusyError must inherit from AudioError."""
        assert issubclass(DeviceBusyError, AudioError)


# ---------------------------------------------------------------------------
# Config values are read from config, not hardcoded
# ---------------------------------------------------------------------------


class TestConfigDriven:
    def test_custom_sample_rate_used(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AudioRecorder must respect a non-default sample_rate from config."""
        import sounddevice as sd

        custom_rate = 8000
        monkeypatch.setattr(config.audio, "sample_rate", custom_rate)
        monkeypatch.setattr(config.audio, "recordings_dir", str(tmp_path / "recs"))

        recorder = AudioRecorder()

        captured: dict = {}

        def fake_rec(frames, samplerate, channels, dtype):
            captured["samplerate"] = samplerate
            return np.zeros((frames, config.audio.channels), dtype=np.int16)

        monkeypatch.setattr(sd, "rec", fake_rec)
        monkeypatch.setattr(sd, "wait", lambda: None)
        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        recorder.record(1)

        assert captured["samplerate"] == custom_rate

    def test_recordings_dir_from_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AudioRecorder must write into the directory from config.audio.recordings_dir."""
        import sounddevice as sd

        custom_dir = tmp_path / "my_custom_recordings"
        monkeypatch.setattr(config.audio, "recordings_dir", str(custom_dir))

        recorder = AudioRecorder()

        monkeypatch.setattr(sd, "rec", lambda *a, **kw: FAKE_AUDIO)
        monkeypatch.setattr(sd, "wait", lambda: None)
        monkeypatch.setattr(sd, "query_devices", lambda: FAKE_DEVICES)

        out = recorder.record(1)

        assert out.parent == custom_dir
