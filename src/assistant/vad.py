"""VADProcessor: per-frame voice activity detection using the Silero VAD model.

This module provides a single-frame classifier that returns True when speech is
detected in a 30ms audio frame (480 samples at 16kHz).  The continuous
speech-segmentation state machine (listen_continuous) is implemented in task
1.7.2 and lives in audio.py.

Model loading is deferred to ``__init__`` so the first import of this module
does not trigger a network download.  Pass a fake *model* object in tests to
keep the test suite fully offline.

Constants
---------
SAMPLE_RATE : int
    Expected sample rate in Hz (16 000).  Silero VAD requires exactly 16kHz.
FRAME_SAMPLES : int
    Number of samples per frame (480 = 30ms at 16kHz).  Silero's window size.
    Frames that are not exactly this length are *rejected* – the caller is
    responsible for chunking the audio stream into fixed-size frames.
"""

from __future__ import annotations

import numpy as np

from assistant.config import config

# ---------------------------------------------------------------------------
# Named constants (not tunables — these are Silero model requirements)
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 16_000  # Hz — Silero VAD requires 16kHz input
FRAME_SAMPLES: int = 480  # samples — 30ms window at 16kHz


class VADProcessor:
    """Per-frame voice activity classifier backed by the Silero VAD model.

    Parameters
    ----------
    model:
        A callable that matches Silero's call signature::

            probability: float = model(tensor, sample_rate).item()

        where *tensor* is a 1-D float32 torch.Tensor of shape ``(FRAME_SAMPLES,)``
        normalised to ``[-1, 1]`` and *sample_rate* is an integer.  If ``None``
        the real Silero model is loaded via ``torch.hub.load`` on first
        construction (requires an internet connection on that first call only;
        the ~2MB model is cached locally afterwards).
    utils:
        Optional utilities tuple returned by ``torch.hub.load`` alongside the
        model (currently unused — kept for future segmentation helpers).
    """

    def __init__(self, model=None, utils=None) -> None:
        # Read all tunables from config — never hardcode these.
        self.threshold: float = config.vad.threshold
        self.min_speech_duration_ms: int = config.vad.min_speech_duration_ms
        self.silence_duration_ms: int = config.vad.silence_duration_ms

        if model is None:
            import torch  # local import keeps the module importable without torch installed

            loaded_model, loaded_utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,  # non-interactive (Cloud Run has no TTY for the trust prompt)
            )
            self._model = loaded_model
            self._utils = loaded_utils
        else:
            self._model = model
            self._utils = utils

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """Classify a single audio frame as speech or silence.

        Parameters
        ----------
        audio_frame:
            A 1-D NumPy array of exactly ``FRAME_SAMPLES`` (480) samples.
            Accepted dtypes:
            - ``float32`` — assumed already in ``[-1, 1]``; used as-is.
            - ``int16``   — normalised to ``[-1, 1]`` by dividing by 32 768.
            Any other dtype is cast to float32 first.

        Returns
        -------
        bool
            ``True`` when the model's speech probability is >= the configured
            threshold (``config.vad.threshold``).

        Raises
        ------
        ValueError
            If *audio_frame* does not contain exactly ``FRAME_SAMPLES`` samples.
        """
        if audio_frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(
                f"VADProcessor.process_frame expects exactly {FRAME_SAMPLES} samples "
                f"({FRAME_SAMPLES / SAMPLE_RATE * 1000:.0f}ms at {SAMPLE_RATE}Hz), "
                f"got {audio_frame.shape[0]}."
            )

        # Normalise to float32 in [-1, 1]
        if audio_frame.dtype == np.int16:
            tensor_data = audio_frame.astype(np.float32) / 32768.0
        elif audio_frame.dtype == np.float32:
            tensor_data = audio_frame
        else:
            tensor_data = audio_frame.astype(np.float32)

        import torch  # local import — avoids hard dependency at module level

        tensor = torch.from_numpy(tensor_data)
        probability: float = self._model(tensor, SAMPLE_RATE).item()
        return probability >= self.threshold
