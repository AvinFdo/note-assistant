"""Tests for _SpeechSegmenter — the pure speech-segmentation state machine.

All tests are fully offline: they use a fake VAD that returns a scripted
bool sequence and inject a temporary recordings directory via monkeypatch.
No sounddevice stream is opened; that thin wrapper in listen_continuous is
intentionally left for live-mic end-to-end validation.

Each test constructs a ``_SpeechSegmenter`` via a real ``AudioRecorder``
(pointed at ``tmp_path``) so that the same ``_wav_path`` / ``_write_wav``
helpers used in production are exercised.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from assistant.audio import AudioRecorder, _SpeechSegmenter
from assistant.config import config
from assistant.vad import FRAME_SAMPLES, SAMPLE_RATE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _silence_frame() -> np.ndarray:
    """Return a frame of zeros (silence)."""
    return np.zeros(FRAME_SAMPLES, dtype=np.int16)


def _speech_frame(value: int = 1000) -> np.ndarray:
    """Return a frame filled with *value* (detectable as non-silence)."""
    return np.full(FRAME_SAMPLES, value, dtype=np.int16)


def _ms_to_frames(ms: int) -> int:
    """Convert milliseconds to a frame count (ceiling)."""
    frames_per_second = SAMPLE_RATE / FRAME_SAMPLES
    return int(ms / 1000.0 * frames_per_second) + 1


class FakeVAD:
    """Scripted VAD that replays a pre-set sequence of booleans.

    After the sequence is exhausted every call returns False.
    Raises ``ValueError`` (via real VADProcessor) if frame length is wrong,
    so we keep the same contract — but here we just check shape ourselves.
    """

    def __init__(self, responses: list[bool]) -> None:
        self._responses = list(responses)
        self._index = 0

    def process_frame(self, frame: np.ndarray) -> bool:
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"FakeVAD: expected {FRAME_SAMPLES} samples, got {frame.shape[0]}")
        if self._index < len(self._responses):
            result = self._responses[self._index]
            self._index += 1
            return result
        return False


@pytest.fixture()
def recorder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AudioRecorder:
    """AudioRecorder writing into tmp_path, with config patched to match."""
    monkeypatch.setattr(config.audio, "recordings_dir", str(tmp_path / "recordings"))
    return AudioRecorder()


def _make_segmenter(recorder: AudioRecorder, vad: FakeVAD) -> _SpeechSegmenter:
    return _SpeechSegmenter(recorder=recorder, vad=vad)


def _collect_segments(segmenter: _SpeechSegmenter, frames: list[np.ndarray]) -> list[Path]:
    """Feed all frames into the segmenter and return every non-None result."""
    results: list[Path] = []
    for frame in frames:
        path = segmenter.process_frame(frame)
        if path is not None:
            results.append(path)
    return results


# ---------------------------------------------------------------------------
# Helper: build a frame sequence that will definitely produce segments
# ---------------------------------------------------------------------------


def _silence_burst(n: int) -> list[np.ndarray]:
    return [_silence_frame() for _ in range(n)]


def _speech_burst(n: int, value: int = 1000) -> list[np.ndarray]:
    return [_speech_frame(value) for _ in range(n)]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestSingleSegment:
    """[silence*k, speech*N, silence*M] → exactly ONE WAV."""

    def test_produces_exactly_one_wav(
        self, recorder: AudioRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single speech burst followed by sufficient silence yields one WAV."""
        silence_lead = 5
        speech_n = _ms_to_frames(config.vad.min_speech_duration_ms) + 5
        silence_trail = _ms_to_frames(config.vad.silence_duration_ms) + 5

        responses = [False] * silence_lead + [True] * speech_n + [False] * silence_trail
        frames = (
            _silence_burst(silence_lead) + _speech_burst(speech_n) + _silence_burst(silence_trail)
        )

        vad = FakeVAD(responses)
        segmenter = _make_segmenter(recorder, vad)
        wavs = _collect_segments(segmenter, frames)

        assert len(wavs) == 1, f"Expected 1 WAV, got {len(wavs)}"
        assert wavs[0].exists()
        assert wavs[0].suffix == ".wav"

    def test_wav_is_readable(
        self, recorder: AudioRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The produced WAV must be readable via the stdlib wave module."""
        silence_lead = 3
        speech_n = _ms_to_frames(config.vad.min_speech_duration_ms) + 3
        silence_trail = _ms_to_frames(config.vad.silence_duration_ms) + 3

        responses = [False] * silence_lead + [True] * speech_n + [False] * silence_trail
        frames = (
            _silence_burst(silence_lead) + _speech_burst(speech_n) + _silence_burst(silence_trail)
        )

        vad = FakeVAD(responses)
        segmenter = _make_segmenter(recorder, vad)
        wavs = _collect_segments(segmenter, frames)

        assert len(wavs) == 1
        with wave.open(str(wavs[0]), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getsampwidth() > 0

    def test_frame_count_includes_pre_buffer_and_speech(
        self, recorder: AudioRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WAV frame count must include rolling-buffer frames + speech + trailing silence.

        The state machine works as follows when a speech onset is detected:
        - ``pre_buffer.append(frame)`` is called first (so the triggering speech
          frame is the last item in the buffer snapshot).
        - ``segment_frames = list(pre_buffer)`` — this holds ``buffer_cap`` frames
          where the final frame is the first speech frame.
        - Remaining ``speech_n - 1`` speech frames are appended in SPEECH state.
        - The segment finalises exactly when ``trailing_silence >= silence_frames``
          (the threshold), so exactly ``silence_frames`` trailing-silence frames
          are included (not the full ``silence_trail`` burst).

        Total frames in WAV = buffer_cap + (speech_n - 1) + silence_frames_threshold
        """
        # Force a specific short buffer so we can compute the exact expectation.
        buffer_s = 0.1  # 100ms → 3 frames at 33.3 fps
        monkeypatch.setattr(config.vad, "buffer_duration_s", buffer_s)

        # Re-create recorder AFTER patching buffer config.
        monkeypatch.setattr(config.audio, "recordings_dir", str(recorder._recordings_dir))
        fresh_recorder = AudioRecorder()

        frames_per_second = SAMPLE_RATE / FRAME_SAMPLES
        buffer_cap = int(buffer_s * frames_per_second)  # deque maxlen

        # silence_frames is the exact threshold used in _SpeechSegmenter
        silence_frames_threshold = int(config.vad.silence_duration_ms / 1000.0 * frames_per_second)

        silence_lead = buffer_cap + 2  # more silence than buffer; only buffer_cap kept
        speech_n = _ms_to_frames(config.vad.min_speech_duration_ms) + 2
        # Send more silence than the threshold to ensure the segment closes.
        silence_trail = silence_frames_threshold + 3

        responses = [False] * silence_lead + [True] * speech_n + [False] * silence_trail
        frames = (
            _silence_burst(silence_lead) + _speech_burst(speech_n) + _silence_burst(silence_trail)
        )

        vad = FakeVAD(responses)
        segmenter = _make_segmenter(fresh_recorder, vad)
        wavs = _collect_segments(segmenter, frames)

        assert len(wavs) == 1
        with wave.open(str(wavs[0]), "rb") as wf:
            total_written_samples = wf.getnframes()

        # buffer_cap frames (pre-buffer snapshot, last one is first speech frame)
        # + (speech_n - 1) frames appended in SPEECH state
        # + silence_frames_threshold trailing-silence frames (triggers finalisation)
        expected_total_frames = buffer_cap + (speech_n - 1) + silence_frames_threshold
        expected_total_samples = expected_total_frames * FRAME_SAMPLES
        assert total_written_samples == expected_total_samples, (
            f"Expected {expected_total_samples} samples ({expected_total_frames} frames) "
            f"in WAV, got {total_written_samples}"
        )


class TestTwoSegments:
    """Two speech bursts separated by long silence → TWO WAVs."""

    def test_produces_two_wavs(
        self, recorder: AudioRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        silence_gap = _ms_to_frames(config.vad.silence_duration_ms) + 5
        speech_n = _ms_to_frames(config.vad.min_speech_duration_ms) + 5

        responses = (
            [True] * speech_n + [False] * silence_gap + [True] * speech_n + [False] * silence_gap
        )
        frames = (
            _speech_burst(speech_n)
            + _silence_burst(silence_gap)
            + _speech_burst(speech_n)
            + _silence_burst(silence_gap)
        )

        vad = FakeVAD(responses)
        segmenter = _make_segmenter(recorder, vad)
        wavs = _collect_segments(segmenter, frames)

        assert len(wavs) == 2, f"Expected 2 WAVs, got {len(wavs)}"
        assert wavs[0].exists()
        assert wavs[1].exists()
        assert wavs[0] != wavs[1], "Two segments should produce two distinct paths"

    def test_both_wavs_are_readable(
        self, recorder: AudioRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        silence_gap = _ms_to_frames(config.vad.silence_duration_ms) + 3
        speech_n = _ms_to_frames(config.vad.min_speech_duration_ms) + 3

        responses = (
            [True] * speech_n + [False] * silence_gap + [True] * speech_n + [False] * silence_gap
        )
        frames = (
            _speech_burst(speech_n)
            + _silence_burst(silence_gap)
            + _speech_burst(speech_n)
            + _silence_burst(silence_gap)
        )

        vad = FakeVAD(responses)
        segmenter = _make_segmenter(recorder, vad)
        wavs = _collect_segments(segmenter, frames)

        assert len(wavs) == 2
        for wav in wavs:
            with wave.open(str(wav), "rb") as wf:
                assert wf.getnframes() > 0


class TestShortSegmentDiscarded:
    """Speech burst shorter than min_speech_duration_ms → NO WAV."""

    def test_short_burst_discarded(
        self, recorder: AudioRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blip shorter than the minimum speech duration must be discarded."""
        # Ensure the burst is shorter than the threshold.
        frames_per_second = SAMPLE_RATE / FRAME_SAMPLES
        min_frames = int(config.vad.min_speech_duration_ms / 1000.0 * frames_per_second)
        # Use fewer frames than the minimum — at least 1, but strictly less than min.
        short_burst = max(1, min_frames - 2)

        silence_trail = _ms_to_frames(config.vad.silence_duration_ms) + 3
        responses = [True] * short_burst + [False] * silence_trail
        frames = _speech_burst(short_burst) + _silence_burst(silence_trail)

        vad = FakeVAD(responses)
        segmenter = _make_segmenter(recorder, vad)
        wavs = _collect_segments(segmenter, frames)

        assert wavs == [], f"Expected no WAV for short burst, got {wavs}"

    def test_short_then_long_burst(
        self, recorder: AudioRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short blip is discarded; a subsequent full burst still produces a WAV."""
        frames_per_second = SAMPLE_RATE / FRAME_SAMPLES
        min_frames = int(config.vad.min_speech_duration_ms / 1000.0 * frames_per_second)
        short_burst = max(1, min_frames - 2)
        long_burst = min_frames + 5

        silence_gap = _ms_to_frames(config.vad.silence_duration_ms) + 3

        responses = (
            [True] * short_burst
            + [False] * silence_gap
            + [True] * long_burst
            + [False] * silence_gap
        )
        frames = (
            _speech_burst(short_burst)
            + _silence_burst(silence_gap)
            + _speech_burst(long_burst)
            + _silence_burst(silence_gap)
        )

        vad = FakeVAD(responses)
        segmenter = _make_segmenter(recorder, vad)
        wavs = _collect_segments(segmenter, frames)

        assert len(wavs) == 1, f"Expected 1 WAV (only the long burst), got {len(wavs)}"


class TestRollingBuffer:
    """Pre-speech frames are included in the output WAV."""

    def test_pre_speech_frames_present_in_wav(
        self, recorder: AudioRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Frames sent just before speech onset (in the rolling buffer) appear in the WAV.

        When speech is first detected, ``pre_buffer.append(frame)`` is called
        before the buffer snapshot is taken, so the buffer's last entry IS the
        first speech frame.  The ``(buffer_frames - 1)`` preceding silence frames
        therefore appear at the very start of the WAV with their distinctive value.
        """
        # Use a short, exact buffer: 5 frames.
        buffer_frames = 5
        buffer_s = buffer_frames * FRAME_SAMPLES / SAMPLE_RATE
        monkeypatch.setattr(config.vad, "buffer_duration_s", buffer_s)
        monkeypatch.setattr(config.audio, "recordings_dir", str(recorder._recordings_dir))
        fresh_recorder = AudioRecorder()

        # Send SILENCE frames with a distinctive sample value (42) before speech.
        pre_val = np.int16(42)
        silence_before = [
            np.full(FRAME_SAMPLES, pre_val, dtype=np.int16) for _ in range(buffer_frames)
        ]

        speech_n = _ms_to_frames(config.vad.min_speech_duration_ms) + 2
        silence_trail = _ms_to_frames(config.vad.silence_duration_ms) + 2

        responses = [False] * buffer_frames + [True] * speech_n + [False] * silence_trail
        frames = silence_before + _speech_burst(speech_n) + _silence_burst(silence_trail)

        vad = FakeVAD(responses)
        segmenter = _make_segmenter(fresh_recorder, vad)
        wavs = _collect_segments(segmenter, frames)

        assert len(wavs) == 1
        with wave.open(str(wavs[0]), "rb") as wf:
            raw = wf.readframes(wf.getnframes())

        pcm = np.frombuffer(raw, dtype=np.int16)

        # The buffer snapshot contains (buffer_frames - 1) pre-speech silence
        # frames followed by the first speech frame (which was appended to the
        # buffer before the snapshot was taken).  Verify the silence frames are
        # at the very start of the WAV.
        pre_silence_samples = (buffer_frames - 1) * FRAME_SAMPLES
        assert (pcm[:pre_silence_samples] == pre_val).all(), (
            f"Expected first {pre_silence_samples} samples to be {pre_val} "
            f"(pre-speech buffer), but got: {pcm[:pre_silence_samples]}"
        )


class TestTrailingSilenceWithinSegment:
    """Short trailing silence does NOT prematurely cut the segment."""

    def test_partial_silence_does_not_cut_segment(
        self, recorder: AudioRecorder, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Silence below the threshold doesn't end the segment; speech can resume."""
        frames_per_second = SAMPLE_RATE / FRAME_SAMPLES
        silence_thresh = int(config.vad.silence_duration_ms / 1000.0 * frames_per_second)
        # Use half the silence threshold — not enough to terminate.
        short_silence = max(1, silence_thresh // 2)

        speech_before = _ms_to_frames(config.vad.min_speech_duration_ms) + 2
        speech_after = 5
        long_silence = silence_thresh + 3

        # Pattern: speech, short pause, more speech, then long silence → ONE segment.
        responses = (
            [True] * speech_before
            + [False] * short_silence
            + [True] * speech_after
            + [False] * long_silence
        )
        frames = (
            _speech_burst(speech_before)
            + _silence_burst(short_silence)
            + _speech_burst(speech_after)
            + _silence_burst(long_silence)
        )

        vad = FakeVAD(responses)
        segmenter = _make_segmenter(recorder, vad)
        wavs = _collect_segments(segmenter, frames)

        assert len(wavs) == 1, (
            f"Short internal pause should not split segment — got {len(wavs)} WAVs"
        )
