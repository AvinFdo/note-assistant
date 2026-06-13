"""Tests for VADProcessor (vad.py — task 1.7.1).

All tests are fully offline: the Silero model is NEVER downloaded.
A fake model is injected via the ``model`` parameter of ``VADProcessor``.

Fake model call signature matches Silero's:
    probability_tensor = fake_model(tensor, sample_rate)
    probability_float  = probability_tensor.item()

A small helper class ``_FakeModel`` captures call arguments so tests can
assert normalisation behaviour without touching the network.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from assistant.vad import FRAME_SAMPLES, SAMPLE_RATE, VADProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeModel:
    """Minimal stand-in for the Silero VAD model.

    Parameters
    ----------
    fixed_prob:
        The float probability that will be returned for *every* call.
    """

    def __init__(self, fixed_prob: float) -> None:
        self.fixed_prob = fixed_prob
        self.last_tensor: torch.Tensor | None = None
        self.last_sample_rate: int | None = None

    def __call__(self, tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self.last_tensor = tensor
        self.last_sample_rate = sample_rate
        return torch.tensor(self.fixed_prob)


def _make_frame(dtype=np.float32, value: float = 0.0) -> np.ndarray:
    """Return a silent frame of the correct size and dtype."""
    frame = np.full(FRAME_SAMPLES, value, dtype=np.float64)
    return frame.astype(dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpeechDetection:
    """VADProcessor.process_frame returns correct bool based on model output."""

    def test_above_threshold_returns_true(self, monkeypatch):
        """Model returns probability above threshold → True."""
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        fake = _FakeModel(fixed_prob=0.9)
        vad = VADProcessor(model=fake)
        frame = _make_frame(np.float32)
        assert vad.process_frame(frame) is True

    def test_below_threshold_returns_false(self, monkeypatch):
        """Model returns probability below threshold → False."""
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        fake = _FakeModel(fixed_prob=0.1)
        vad = VADProcessor(model=fake)
        frame = _make_frame(np.float32)
        assert vad.process_frame(frame) is False

    def test_exactly_at_threshold_returns_true(self, monkeypatch):
        """Probability == threshold is on-or-above → True."""
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        fake = _FakeModel(fixed_prob=0.5)
        vad = VADProcessor(model=fake)
        frame = _make_frame(np.float32)
        assert vad.process_frame(frame) is True

    def test_silence_frame_returns_false(self, monkeypatch):
        """A silence frame (all zeros) with near-zero model score → False."""
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        fake = _FakeModel(fixed_prob=0.0)
        vad = VADProcessor(model=fake)
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        assert vad.process_frame(silence) is False


class TestThresholdConfig:
    """Threshold is read from config and respected."""

    def test_high_threshold_rejects_moderate_probability(self, monkeypatch):
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.9)
        fake = _FakeModel(fixed_prob=0.6)
        vad = VADProcessor(model=fake)
        frame = _make_frame(np.float32)
        assert vad.process_frame(frame) is False

    def test_low_threshold_accepts_moderate_probability(self, monkeypatch):
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.2)
        fake = _FakeModel(fixed_prob=0.6)
        vad = VADProcessor(model=fake)
        frame = _make_frame(np.float32)
        assert vad.process_frame(frame) is True

    def test_threshold_stored_from_config(self, monkeypatch):
        """VADProcessor stores the threshold value from config at init time."""
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.75)
        fake = _FakeModel(fixed_prob=0.0)
        vad = VADProcessor(model=fake)
        assert vad.threshold == pytest.approx(0.75)


class TestInt16Normalisation:
    """int16 frames are normalised to [-1, 1] before being passed to the model."""

    def test_int16_max_value_normalised_to_approx_one(self, monkeypatch):
        """int16 max (32767) → float32 ≈ 1.0 after /32768."""
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        fake = _FakeModel(fixed_prob=0.9)
        vad = VADProcessor(model=fake)

        frame = np.full(FRAME_SAMPLES, 32767, dtype=np.int16)
        vad.process_frame(frame)

        assert fake.last_tensor is not None
        max_abs = fake.last_tensor.abs().max().item()
        # 32767 / 32768 < 1.0 but very close
        assert max_abs <= 1.0
        assert max_abs > 0.99

    def test_int16_frame_max_abs_le_one(self, monkeypatch):
        """For any int16 frame, the tensor passed to the model has max abs ≤ 1."""
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        rng = np.random.default_rng(seed=42)
        raw_int16 = rng.integers(-32768, 32768, size=FRAME_SAMPLES, dtype=np.int16)

        fake = _FakeModel(fixed_prob=0.0)
        vad = VADProcessor(model=fake)
        vad.process_frame(raw_int16)

        assert fake.last_tensor is not None
        assert fake.last_tensor.abs().max().item() <= 1.0

    def test_int16_zero_frame_passes_zero_tensor(self, monkeypatch):
        """A silent int16 frame (all zeros) → all-zero tensor."""
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        fake = _FakeModel(fixed_prob=0.0)
        vad = VADProcessor(model=fake)

        silence = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        vad.process_frame(silence)

        assert fake.last_tensor is not None
        assert fake.last_tensor.abs().max().item() == pytest.approx(0.0)


class TestFloat32PassThrough:
    """float32 frames are used as-is (no re-normalisation)."""

    def test_float32_frame_passes_unchanged(self, monkeypatch):
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        fake = _FakeModel(fixed_prob=0.0)
        vad = VADProcessor(model=fake)

        frame = np.linspace(-0.5, 0.5, FRAME_SAMPLES, dtype=np.float32)
        vad.process_frame(frame)

        assert fake.last_tensor is not None
        np.testing.assert_allclose(fake.last_tensor.numpy(), frame, rtol=1e-5, atol=1e-7)


class TestModelCallSignature:
    """The model is called with (tensor, SAMPLE_RATE)."""

    def test_sample_rate_passed_correctly(self, monkeypatch):
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        fake = _FakeModel(fixed_prob=0.5)
        vad = VADProcessor(model=fake)
        vad.process_frame(_make_frame(np.float32))
        assert fake.last_sample_rate == SAMPLE_RATE

    def test_tensor_shape_is_1d_frame_samples(self, monkeypatch):
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        fake = _FakeModel(fixed_prob=0.5)
        vad = VADProcessor(model=fake)
        vad.process_frame(_make_frame(np.float32))
        assert fake.last_tensor is not None
        assert fake.last_tensor.shape == (FRAME_SAMPLES,)
        assert fake.last_tensor.dtype == torch.float32


class TestFrameSizeValidation:
    """process_frame rejects frames with wrong number of samples."""

    def test_too_short_raises_value_error(self, monkeypatch):
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        fake = _FakeModel(fixed_prob=0.5)
        vad = VADProcessor(model=fake)
        short = np.zeros(240, dtype=np.float32)
        with pytest.raises(ValueError, match="512"):
            vad.process_frame(short)

    def test_too_long_raises_value_error(self, monkeypatch):
        monkeypatch.setattr("assistant.vad.config.vad.threshold", 0.5)
        fake = _FakeModel(fixed_prob=0.5)
        vad = VADProcessor(model=fake)
        long_frame = np.zeros(960, dtype=np.float32)
        with pytest.raises(ValueError, match="512"):
            vad.process_frame(long_frame)
