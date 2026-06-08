"""CLI entry point for the Avin assistant.

Provides the ``avin`` command group and the ``listen`` sub-command which
supports fixed-duration recording (``--duration N``), default-duration
recording, and continuous VAD-based listening (``--continuous``).

All objects that touch hardware (AudioRecorder) or live APIs
(Transcriber, Brain, Memory) are constructed through thin factory
functions (``_make_recorder``, ``_make_transcriber``, ``_make_brain``,
``_make_memory``) so that tests can monkeypatch them without any I/O.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path

import click

from assistant.actions import route_action
from assistant.audio import AudioRecorder
from assistant.brain import ActionItem, Brain
from assistant.config import config
from assistant.memory import Memory
from assistant.transcriber import Transcriber

# ---------------------------------------------------------------------------
# Factory functions — monkeypatched in tests to inject mocks
# ---------------------------------------------------------------------------


def _make_recorder() -> AudioRecorder:
    """Construct and return a real AudioRecorder."""
    return AudioRecorder()


def _make_memory() -> Memory:
    """Construct and return a Memory instance backed by config.memory.db_path."""
    return Memory()


def _make_transcriber() -> Transcriber:
    """Construct and return a real Transcriber (uses live Gemini API)."""
    return Transcriber()


def _make_brain(memory: Memory) -> Brain:
    """Construct and return a real Brain backed by *memory* (uses live Gemini API)."""
    return Brain(memory=memory)


# ---------------------------------------------------------------------------
# Shared pipeline helper
# ---------------------------------------------------------------------------


def process_audio_file(
    wav_path: Path,
    *,
    transcriber: Transcriber,
    brain: Brain,
    memory: Memory,
    confirm: Callable[[str], bool] | None = None,
) -> None:
    """Transcribe *wav_path*, run the brain, and route any pending actions.

    Steps
    -----
    1. Transcribe the WAV file → plain text.
    2. Pass text to Brain.process() which persists the conversation, note,
       and all action rows with the correct confidence-filtered statuses:
       - confidence < threshold  → saved with status ``'low_confidence'``
       - confidence >= threshold → saved with status ``'pending'``
    3. Fetch ``memory.get_pending_actions()`` (only the high-confidence
       rows) and route each one through ``route_action``.  Low-confidence
       actions are intentionally skipped here — they never reach execute().
    4. Print the result of each routed action via click.echo.

    Args:
        wav_path:    Path to a valid WAV file produced by AudioRecorder.
        transcriber: Transcriber instance (injected for testability).
        brain:       Brain instance (injected for testability).
        memory:      Memory instance (injected for testability).
        confirm:     Optional callable ``(description: str) -> bool`` used for
                     ``confirm_first`` actions such as ``send_email``.  When
                     ``None``, confirm_first actions remain "pending" and are
                     never executed (GUARDRAIL intact).
    """
    # Step 1: transcribe
    result = transcriber.transcribe(wav_path)
    text = result.text
    click.echo(f"Transcript: {text}")

    # Step 2: brain processes and persists (applies confidence guardrail internally)
    brain.process(text)

    # Step 3: route only the high-confidence pending actions
    pending = memory.get_pending_actions()
    for action_row in pending:
        action_item = ActionItem(
            intent=action_row.intent,
            confidence=1.0,  # already filtered by brain; guaranteed high-confidence
            details=action_row.details,
        )
        outcome = route_action(
            action_item,
            memory,
            action_id=action_row.id,
            confirm=confirm,
        )
        click.echo(outcome)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Avin — context-aware voice assistant."""


# ---------------------------------------------------------------------------
# listen command
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--duration",
    "-d",
    type=int,
    default=None,
    help="Record for exactly N seconds (fixed mode).",
)
@click.option(
    "--continuous",
    "-c",
    is_flag=True,
    default=False,
    help="Listen continuously using VAD until Ctrl+C; process each speech segment.",
)
def listen(duration: int | None, continuous: bool) -> None:
    """Record audio and process it through the Avin pipeline.

    Without flags, records for config.audio.default_duration seconds.
    Use --duration N to record for exactly N seconds.
    Use --continuous to listen indefinitely with VAD speech segmentation.
    """
    if continuous and duration is not None:
        raise click.UsageError(
            "--continuous and --duration are mutually exclusive. "
            "Use --continuous to listen with VAD, or --duration N for fixed-length recording."
        )

    memory = _make_memory()
    transcriber = _make_transcriber()
    brain = _make_brain(memory)
    recorder = _make_recorder()

    def confirm(description: str) -> bool:
        return click.confirm(description)

    if continuous:
        click.echo("Listening continuously — press Ctrl+C to stop.")
        with contextlib.suppress(KeyboardInterrupt):
            recorder.listen_continuous(
                callback=lambda p: process_audio_file(
                    p,
                    transcriber=transcriber,
                    brain=brain,
                    memory=memory,
                    confirm=confirm,
                )
            )
        click.echo("Stopped.")
    else:
        duration_seconds = duration if duration is not None else config.audio.default_duration
        click.echo(f"Recording for {duration_seconds}s...")
        wav_path = recorder.record(duration_seconds)
        process_audio_file(
            wav_path,
            transcriber=transcriber,
            brain=brain,
            memory=memory,
            confirm=confirm,
        )
