"""CLI entry point for the Avin assistant.

Provides the ``avin`` command group and all sub-commands:

- ``listen``   — record audio and process through the pipeline.
- ``history``  — show recent notes in a rich Table.
- ``search``   — search notes by keyword.
- ``actions``  — list recent/pending actions; ``actions confirm <id>`` to execute.
- ``replay``   — display a full past conversation (transcript + notes + actions).
- ``config``   — print current configuration with sensitive values masked.
- ``devices``  — list available audio input devices.

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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from assistant.actions import route_action
from assistant.audio import AudioRecorder
from assistant.brain import ActionItem, Brain
from assistant.config import config
from assistant.integrations.obsidian import ObsidianWriter
from assistant.memory import Memory
from assistant.memory_factory import create_memory
from assistant.transcriber import Transcriber

_console = Console()

# ---------------------------------------------------------------------------
# Factory functions — monkeypatched in tests to inject mocks
# ---------------------------------------------------------------------------


def _make_recorder() -> AudioRecorder:
    """Construct and return a real AudioRecorder."""
    return AudioRecorder()


def _make_memory() -> Memory:
    """Construct the configured storage backend (SQLite or Firestore)."""
    return create_memory()


def _make_transcriber() -> Transcriber:
    """Construct and return a real Transcriber (uses live Gemini API)."""
    return Transcriber()


def _make_brain(memory: Memory) -> Brain:
    """Construct and return a real Brain backed by *memory* (uses live Gemini API)."""
    return Brain(memory=memory)


def _make_obsidian_writer() -> ObsidianWriter:
    """Construct and return an ObsidianWriter from config.

    Tests monkeypatch this to inject a writer pointed at a tmp vault.
    """
    return ObsidianWriter()


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
    obsidian_writer: ObsidianWriter | None = None,
) -> None:
    """Transcribe *wav_path*, run the brain, and route any pending actions.

    Steps
    -----
    1. Transcribe the WAV file → plain text.
    2. Pass text to Brain.process() which persists the conversation, note,
       and all action rows with the correct confidence-filtered statuses:
       - confidence < threshold  → saved with status ``'low_confidence'``
       - confidence >= threshold → saved with status ``'pending'``
    3. If the result is noteworthy and *obsidian_writer* is configured, write
       the summary + actions to the daily vault note.  The write is gated on
       ``writer.is_configured()`` so no vault configured → no write, no error.
    4. Fetch ``memory.get_pending_actions()`` (only the high-confidence
       rows) and route each one through ``route_action``.  Low-confidence
       actions are intentionally skipped here — they never reach execute().
    5. Print the result of each routed action via click.echo.

    Args:
        wav_path:         Path to a valid WAV file produced by AudioRecorder.
        transcriber:      Transcriber instance (injected for testability).
        brain:            Brain instance (injected for testability).
        memory:           Memory instance (injected for testability).
        confirm:          Optional callable ``(description: str) -> bool`` used for
                          ``confirm_first`` actions such as ``send_email``.  When
                          ``None``, confirm_first actions remain "pending" and are
                          never executed (GUARDRAIL intact).
        obsidian_writer:  Optional :class:`~assistant.integrations.obsidian.ObsidianWriter`
                          instance.  When ``None`` (the default), a writer is built
                          from config via :func:`_make_obsidian_writer`.  The write
                          is silently skipped when ``writer.is_configured()`` is False.
    """
    # Step 1: transcribe
    result = transcriber.transcribe(wav_path)
    text = result.text
    click.echo(f"Transcript: {text}")

    # Step 2: brain processes and persists (applies confidence guardrail internally)
    brain_result = brain.process(text)

    # Step 3: optionally write to Obsidian vault (gated on vault being configured)
    writer = obsidian_writer if obsidian_writer is not None else _make_obsidian_writer()
    if brain_result.is_noteworthy and writer.is_configured():
        writer.write_note(brain_result.summary_note, brain_result.actions)

    # Step 4: route only the high-confidence pending actions
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_value(value: str) -> str:
    """Return a masked representation of *value*.

    Shows the first 3 characters followed by ``***`` when the value is set and
    long enough; returns ``"(not set)"`` when the value is empty.  This prevents
    full secrets or paths from appearing in terminal output.
    """
    if not value:
        return "(not set)"
    if len(value) <= 3:
        return "***"
    return value[:3] + "***"


# ---------------------------------------------------------------------------
# history command
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--limit", "-n", default=20, show_default=True, help="Maximum number of notes to show."
)
def history(limit: int) -> None:
    """Show recent notes in a formatted table."""
    memory = _make_memory()
    notes = memory.get_recent_notes(limit=limit)

    if not notes:
        _console.print("[yellow]No notes found.[/yellow]")
        return

    table = Table(title="Recent Notes", show_lines=True)
    table.add_column("Created At", style="dim", no_wrap=True)
    table.add_column("Summary")
    table.add_column("Conv ID", style="dim", no_wrap=True)

    for note in notes:
        table.add_row(note.created_at, note.summary, note.conversation_id[:8] + "…")

    _console.print(table)


# ---------------------------------------------------------------------------
# search command
# ---------------------------------------------------------------------------


@main.command()
@click.argument("query")
def search(query: str) -> None:
    """Search notes by keyword and display matching results."""
    memory = _make_memory()
    results = memory.search_notes(query)

    if not results:
        _console.print(f"[yellow]No notes found matching '{query}'.[/yellow]")
        return

    table = Table(title=f"Search results for '{query}'", show_lines=True)
    table.add_column("Created At", style="dim", no_wrap=True)
    table.add_column("Summary")
    table.add_column("Conv ID", style="dim", no_wrap=True)

    for note in results:
        table.add_row(note.created_at, note.summary, note.conversation_id[:8] + "…")

    _console.print(table)


# ---------------------------------------------------------------------------
# actions group (list + confirm subcommands)
# ---------------------------------------------------------------------------


@main.group(invoke_without_command=True)
@click.pass_context
def actions(ctx: click.Context) -> None:
    """Show recent actions or confirm a pending action.

    Run without a subcommand to list recent actions.
    Use 'actions confirm <id>' to execute a pending action.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(actions_list)


@actions.command(name="list")
@click.option(
    "--limit", "-n", default=10, show_default=True, help="Maximum number of actions to show."
)
def actions_list(limit: int) -> None:
    """List recent actions with their status."""
    memory = _make_memory()
    recent = memory.get_recent_actions(limit=limit)

    if not recent:
        _console.print("[yellow]No actions found.[/yellow]")
        return

    table = Table(title="Recent Actions", show_lines=True)
    table.add_column("ID (short)", style="dim", no_wrap=True)
    table.add_column("Intent", style="bold")
    table.add_column("Status")
    table.add_column("Detail")

    for action in recent:
        # Build a short detail string from the details dict
        if isinstance(action.details, dict) and action.details:
            short_detail = str(next(iter(action.details.values())))
        else:
            short_detail = str(action.details)
        if len(short_detail) > 60:
            short_detail = short_detail[:57] + "…"

        status_style = {
            "pending": "yellow",
            "executed": "green",
            "dismissed": "red",
            "logged": "dim",
            "low_confidence": "dim",
        }.get(action.status, "white")

        table.add_row(
            action.id[:8] + "…",
            action.intent,
            f"[{status_style}]{action.status}[/{status_style}]",
            short_detail,
        )

    _console.print(table)


@actions.command(name="confirm")
@click.argument("action_id")
def actions_confirm(action_id: str) -> None:
    """Execute and confirm a pending action by its ACTION_ID."""
    memory = _make_memory()
    action_row = memory.get_action(action_id)

    if action_row is None:
        _console.print(f"[red]Error: no action found with id '{action_id}'.[/red]")
        raise SystemExit(1)

    action_item = ActionItem(
        intent=action_row.intent,
        confidence=1.0,
        details=action_row.details,
    )

    outcome = route_action(
        action_item,
        memory,
        action_id=action_row.id,
        confirm=lambda _description: True,
    )
    _console.print(Panel(outcome, title="Action Outcome", border_style="green"))


# ---------------------------------------------------------------------------
# replay command
# ---------------------------------------------------------------------------


@main.command()
@click.argument("conversation_id")
def replay(conversation_id: str) -> None:
    """Replay a past conversation: transcript, notes, and actions."""
    memory = _make_memory()
    conv = memory.get_conversation(conversation_id)

    if conv is None:
        _console.print(f"[red]Error: no conversation found with id '{conversation_id}'.[/red]")
        raise SystemExit(1)

    # Transcript panel
    _console.print(
        Panel(
            conv.transcript,
            title=f"Transcript — {conv.started_at}",
            border_style="blue",
        )
    )

    # Notes table
    notes = memory.get_notes_for_conversation(conversation_id)
    if notes:
        notes_table = Table(title="Extracted Notes", show_lines=True)
        notes_table.add_column("Created At", style="dim", no_wrap=True)
        notes_table.add_column("Summary")
        for note in notes:
            notes_table.add_row(note.created_at, note.summary)
        _console.print(notes_table)
    else:
        _console.print("[dim]No notes extracted for this conversation.[/dim]")

    # Actions table
    act_rows = memory.get_actions_for_conversation(conversation_id)
    if act_rows:
        actions_table = Table(title="Extracted Actions", show_lines=True)
        actions_table.add_column("Intent", style="bold")
        actions_table.add_column("Status")
        actions_table.add_column("Detail")
        for ar in act_rows:
            if isinstance(ar.details, dict) and ar.details:
                short_detail = str(next(iter(ar.details.values())))
            else:
                short_detail = str(ar.details)
            actions_table.add_row(ar.intent, ar.status, short_detail)
        _console.print(actions_table)
    else:
        _console.print("[dim]No actions extracted for this conversation.[/dim]")


# ---------------------------------------------------------------------------
# config command
# ---------------------------------------------------------------------------


@main.command(name="config")
def show_config() -> None:
    """Print current configuration with sensitive values masked."""
    table = Table(title="Avin Configuration", show_lines=True)
    table.add_column("Section", style="bold")
    table.add_column("Key")
    table.add_column("Value")

    def _add_section(section_name: str, rows: list[tuple[str, str]]) -> None:
        first = True
        for key, value in rows:
            table.add_row(section_name if first else "", key, value)
            first = False

    _add_section(
        "gcp",
        [
            ("project_id", _mask_value(config.gcp.project_id)),
            ("region", config.gcp.region),
        ],
    )
    _add_section(
        "models",
        [
            ("transcription", config.models.transcription),
            ("reasoning", config.models.reasoning),
        ],
    )
    _add_section(
        "audio",
        [
            ("sample_rate", str(config.audio.sample_rate)),
            ("channels", str(config.audio.channels)),
            ("format", config.audio.format),
            ("recordings_dir", config.audio.recordings_dir),
            ("default_duration", str(config.audio.default_duration)),
        ],
    )
    _add_section(
        "vad",
        [
            ("enabled", str(config.vad.enabled)),
            ("model", config.vad.model),
            ("threshold", str(config.vad.threshold)),
            ("min_speech_duration_ms", str(config.vad.min_speech_duration_ms)),
            ("silence_duration_ms", str(config.vad.silence_duration_ms)),
            ("buffer_duration_s", str(config.vad.buffer_duration_s)),
        ],
    )
    _add_section(
        "memory",
        [
            ("db_path", config.memory.db_path),
            ("context_window_size", str(config.memory.context_window_size)),
            ("max_context_tokens", str(config.memory.max_context_tokens)),
            ("min_transcript_words", str(config.memory.min_transcript_words)),
        ],
    )
    _add_section(
        "actions",
        [
            ("confidence_threshold", str(config.actions.confidence_threshold)),
            ("create_todo.mode", config.actions.create_todo.mode),
            ("send_email.mode", config.actions.send_email.mode),
            ("add_calendar.mode", config.actions.add_calendar.mode),
            ("research_topic.mode", config.actions.research_topic.mode),
        ],
    )
    _add_section(
        "integrations.obsidian",
        [
            ("vault_path", _mask_value(config.integrations.obsidian.vault_path)),
            ("notes_folder", config.integrations.obsidian.notes_folder),
        ],
    )
    _add_section(
        "integrations.google",
        [
            (
                "oauth_credentials_path",
                _mask_value(config.integrations.google.oauth_credentials_path),
            ),
        ],
    )

    _console.print(table)


# ---------------------------------------------------------------------------
# devices command
# ---------------------------------------------------------------------------


@main.command()
def devices() -> None:
    """List available audio input devices."""
    recorder = _make_recorder()
    device_list = recorder.list_devices()

    if not device_list:
        _console.print("[yellow]No audio input devices found.[/yellow]")
        return

    table = Table(title="Audio Input Devices", show_lines=True)
    table.add_column("ID", style="bold", no_wrap=True)
    table.add_column("Name")
    table.add_column("Sample Rate", no_wrap=True)

    for dev in device_list:
        table.add_row(
            str(dev.get("id", "")),
            str(dev.get("name", "")),
            str(dev.get("sample_rate", "")),
        )

    _console.print(table)
