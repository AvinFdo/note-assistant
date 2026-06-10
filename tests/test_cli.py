"""Tests for the avin CLI commands and shared processing pipeline.

All tests run fully offline: the hardware/API factory functions
(`_make_recorder`, `_make_transcriber`, `_make_brain`, `_make_memory`) are
monkeypatched to return fakes, so no microphone, sounddevice stream, or live
Gemini call is ever made.

Covers: listen, history, search, actions, actions confirm, replay, config, devices.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from assistant import cli
from assistant.brain import ActionItem, ProcessingResult
from assistant.config import config
from assistant.memory import Memory


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _patch_factories(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recorder,
    transcriber,
    memory,
    brain,
) -> None:
    """Point all CLI factory functions at the supplied fakes."""
    monkeypatch.setattr(cli, "_make_recorder", lambda: recorder)
    monkeypatch.setattr(cli, "_make_transcriber", lambda: transcriber)
    monkeypatch.setattr(cli, "_make_memory", lambda: memory)
    monkeypatch.setattr(cli, "_make_brain", lambda memory: brain)


def _transcriber_returning(text: str) -> MagicMock:
    t = MagicMock()
    t.transcribe.return_value = MagicMock(text=text)
    return t


class _SavingBrain:
    """A fake Brain whose process() persists actions to the real in-memory Memory.

    Mirrors how the real Brain persists high-confidence actions as 'pending'.
    """

    def __init__(self, memory: Memory, actions: list[tuple[str, dict]]) -> None:
        self._memory = memory
        self._actions = actions
        self.process_calls: list[str] = []

    def process(self, text: str) -> ProcessingResult:
        self.process_calls.append(text)
        cid = self._memory.save_conversation(text)
        action_items: list[ActionItem] = []
        for intent, details in self._actions:
            # execution_mode here is irrelevant — route_action reads mode from config
            self._memory.save_action(cid, intent, details, "pending")
            action_items.append(ActionItem(intent=intent, confidence=1.0, details=details))
        is_noteworthy = bool(self._actions) or bool(text.strip())
        return ProcessingResult(
            is_noteworthy=is_noteworthy,
            summary_note=text[:80] if is_noteworthy else "",
            actions=action_items,
        )


def test_listen_fixed_duration_calls_record_with_n(runner, monkeypatch):
    recorder = MagicMock()
    recorder.record.return_value = Path("/tmp/fake.wav")
    transcriber = _transcriber_returning("hello world this is a test transcript")
    memory = Memory(db_path=":memory:")
    brain = _SavingBrain(memory, actions=[])
    _patch_factories(
        monkeypatch, recorder=recorder, transcriber=transcriber, memory=memory, brain=brain
    )

    result = runner.invoke(cli.main, ["listen", "--duration", "5"])

    assert result.exit_code == 0, result.output
    recorder.record.assert_called_once_with(5)
    transcriber.transcribe.assert_called_once_with(Path("/tmp/fake.wav"))
    assert brain.process_calls == ["hello world this is a test transcript"]


def test_listen_default_duration_uses_config(runner, monkeypatch):
    recorder = MagicMock()
    recorder.record.return_value = Path("/tmp/fake.wav")
    transcriber = _transcriber_returning("text")
    memory = Memory(db_path=":memory:")
    brain = _SavingBrain(memory, actions=[])
    _patch_factories(
        monkeypatch, recorder=recorder, transcriber=transcriber, memory=memory, brain=brain
    )

    result = runner.invoke(cli.main, ["listen"])

    assert result.exit_code == 0, result.output
    recorder.record.assert_called_once_with(config.audio.default_duration)


def test_continuous_and_duration_are_mutually_exclusive(runner, monkeypatch):
    recorder = MagicMock()
    transcriber = _transcriber_returning("text")
    memory = Memory(db_path=":memory:")
    brain = _SavingBrain(memory, actions=[])
    _patch_factories(
        monkeypatch, recorder=recorder, transcriber=transcriber, memory=memory, brain=brain
    )

    result = runner.invoke(cli.main, ["listen", "--continuous", "--duration", "5"])

    assert result.exit_code != 0
    assert "mutually exclusive" in result.output
    recorder.record.assert_not_called()


def test_pipeline_executes_high_confidence_auto_action(runner, monkeypatch):
    recorder = MagicMock()
    recorder.record.return_value = Path("/tmp/fake.wav")
    transcriber = _transcriber_returning("remember to buy milk tomorrow morning please")
    memory = Memory(db_path=":memory:")
    brain = _SavingBrain(memory, actions=[("create_todo", {"task": "buy milk"})])
    _patch_factories(
        monkeypatch, recorder=recorder, transcriber=transcriber, memory=memory, brain=brain
    )

    result = runner.invoke(cli.main, ["listen", "--duration", "5"])

    assert result.exit_code == 0, result.output
    assert "[TODO]" in result.output
    # create_todo is auto_execute -> routed action marked executed
    actions = memory.get_recent_actions()
    assert len(actions) == 1
    assert actions[0].status == "executed"


def test_send_email_not_sent_when_user_declines(runner, monkeypatch):
    recorder = MagicMock()
    recorder.record.return_value = Path("/tmp/fake.wav")
    transcriber = _transcriber_returning("email john about the meeting agenda for friday")
    memory = Memory(db_path=":memory:")
    brain = _SavingBrain(
        memory, actions=[("send_email", {"recipient": "john@x.com", "subject": "Meeting"})]
    )
    _patch_factories(
        monkeypatch, recorder=recorder, transcriber=transcriber, memory=memory, brain=brain
    )

    # User answers 'n' to the confirm prompt -> email must NOT be sent.
    result = runner.invoke(cli.main, ["listen", "--duration", "5"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "[EMAIL]" not in result.output  # execute() never ran
    actions = memory.get_recent_actions()
    assert actions[0].status != "executed"


def test_send_email_sent_when_user_confirms(runner, monkeypatch):
    recorder = MagicMock()
    recorder.record.return_value = Path("/tmp/fake.wav")
    transcriber = _transcriber_returning("email john about the meeting agenda for friday")
    memory = Memory(db_path=":memory:")
    brain = _SavingBrain(
        memory, actions=[("send_email", {"recipient": "john@x.com", "subject": "Meeting"})]
    )
    _patch_factories(
        monkeypatch, recorder=recorder, transcriber=transcriber, memory=memory, brain=brain
    )

    result = runner.invoke(cli.main, ["listen", "--duration", "5"], input="y\n")

    assert result.exit_code == 0, result.output
    assert "[EMAIL]" in result.output
    actions = memory.get_recent_actions()
    assert actions[0].status == "executed"


def test_continuous_processes_each_segment(runner, monkeypatch):
    fake_wav = Path("/tmp/segment.wav")

    recorder = MagicMock()

    def fake_listen_continuous(callback):
        # Simulate one completed speech segment, then return (as on Ctrl+C).
        callback(fake_wav)

    recorder.listen_continuous.side_effect = fake_listen_continuous
    transcriber = _transcriber_returning("this is a spoken sentence captured by vad segmentation")
    memory = Memory(db_path=":memory:")
    brain = _SavingBrain(memory, actions=[])
    _patch_factories(
        monkeypatch, recorder=recorder, transcriber=transcriber, memory=memory, brain=brain
    )

    result = runner.invoke(cli.main, ["listen", "--continuous"])

    assert result.exit_code == 0, result.output
    transcriber.transcribe.assert_called_once_with(fake_wav)
    assert brain.process_calls == ["this is a spoken sentence captured by vad segmentation"]
    assert "Stopped." in result.output


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_help_lists_all_commands(runner):
    result = runner.invoke(cli.main, ["--help"])
    assert result.exit_code == 0, result.output
    for cmd in ("listen", "history", "search", "actions", "replay", "config", "devices"):
        assert cmd in result.output, f"'{cmd}' not found in --help output"


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


def test_history_shows_notes(runner, monkeypatch):
    memory = Memory(db_path=":memory:")
    cid = memory.save_conversation("test conv")
    memory.save_note(cid, "Project alpha review meeting")
    memory.save_note(cid, "Follow-up on budget proposal")
    monkeypatch.setattr(cli, "_make_memory", lambda: memory)

    result = runner.invoke(cli.main, ["history"])

    assert result.exit_code == 0, result.output
    assert "Project alpha review meeting" in result.output
    assert "Follow-up on budget proposal" in result.output


def test_history_empty_db(runner, monkeypatch):
    memory = Memory(db_path=":memory:")
    monkeypatch.setattr(cli, "_make_memory", lambda: memory)

    result = runner.invoke(cli.main, ["history"])

    assert result.exit_code == 0, result.output
    assert "No notes found" in result.output


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_shows_matching_note(runner, monkeypatch):
    memory = Memory(db_path=":memory:")
    cid = memory.save_conversation("conv")
    memory.save_note(cid, "Discuss the Q4 revenue targets")
    memory.save_note(cid, "Lunch with Bob")
    monkeypatch.setattr(cli, "_make_memory", lambda: memory)

    result = runner.invoke(cli.main, ["search", "revenue"])

    assert result.exit_code == 0, result.output
    assert "revenue" in result.output.lower()
    assert "Lunch with Bob" not in result.output


def test_search_no_results(runner, monkeypatch):
    memory = Memory(db_path=":memory:")
    cid = memory.save_conversation("conv")
    memory.save_note(cid, "unrelated content")
    monkeypatch.setattr(cli, "_make_memory", lambda: memory)

    result = runner.invoke(cli.main, ["search", "xyzzy999"])

    assert result.exit_code == 0, result.output
    assert "no notes found" in result.output.lower()


# ---------------------------------------------------------------------------
# actions list
# ---------------------------------------------------------------------------


def test_actions_list_shows_seeded_action(runner, monkeypatch):
    memory = Memory(db_path=":memory:")
    cid = memory.save_conversation("conv")
    memory.save_action(cid, "create_todo", {"task": "write tests"}, "auto_execute")
    monkeypatch.setattr(cli, "_make_memory", lambda: memory)

    result = runner.invoke(cli.main, ["actions"])

    assert result.exit_code == 0, result.output
    assert "create_todo" in result.output
    assert "pending" in result.output


def test_actions_list_empty(runner, monkeypatch):
    memory = Memory(db_path=":memory:")
    monkeypatch.setattr(cli, "_make_memory", lambda: memory)

    result = runner.invoke(cli.main, ["actions"])

    assert result.exit_code == 0, result.output
    assert "No actions found" in result.output


# ---------------------------------------------------------------------------
# actions confirm
# ---------------------------------------------------------------------------


def test_actions_confirm_executes_action(runner, monkeypatch):
    memory = Memory(db_path=":memory:")
    cid = memory.save_conversation("conv")
    aid = memory.save_action(cid, "create_todo", {"task": "buy milk"}, "auto_execute")
    monkeypatch.setattr(cli, "_make_memory", lambda: memory)

    result = runner.invoke(cli.main, ["actions", "confirm", aid])

    assert result.exit_code == 0, result.output
    updated = memory.get_action(aid)
    assert updated is not None
    assert updated.status == "executed"


def test_actions_confirm_unknown_id(runner, monkeypatch):
    memory = Memory(db_path=":memory:")
    monkeypatch.setattr(cli, "_make_memory", lambda: memory)

    result = runner.invoke(cli.main, ["actions", "confirm", "nonexistent-id"])

    assert result.exit_code != 0 or "Error" in result.output or "no action" in result.output.lower()


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def test_replay_shows_transcript_notes_actions(runner, monkeypatch):
    memory = Memory(db_path=":memory:")
    cid = memory.save_conversation("This is the transcript of a meeting")
    memory.save_note(cid, "Meeting summary note")
    memory.save_action(cid, "create_todo", {"task": "send follow-up"}, "auto_execute")
    monkeypatch.setattr(cli, "_make_memory", lambda: memory)

    result = runner.invoke(cli.main, ["replay", cid])

    assert result.exit_code == 0, result.output
    assert "This is the transcript of a meeting" in result.output
    assert "Meeting summary note" in result.output
    assert "create_todo" in result.output


def test_replay_unknown_conversation(runner, monkeypatch):
    memory = Memory(db_path=":memory:")
    monkeypatch.setattr(cli, "_make_memory", lambda: memory)

    result = runner.invoke(cli.main, ["replay", "does-not-exist"])

    assert result.exit_code != 0 or "no conversation found" in result.output.lower()


# ---------------------------------------------------------------------------
# config (masking)
# ---------------------------------------------------------------------------


def test_config_masks_sensitive_values(runner, monkeypatch):
    from assistant.config import config as cfg

    # Patch sensitive values so we can assert they are NOT shown in full
    monkeypatch.setattr(cfg.gcp, "project_id", "my-secret-project-id")
    monkeypatch.setattr(cfg.integrations.google, "oauth_credentials_path", "/home/user/.creds.json")
    monkeypatch.setattr(cfg.integrations.obsidian, "vault_path", "/Users/avin/vault")

    result = runner.invoke(cli.main, ["config"])

    assert result.exit_code == 0, result.output
    # Full secret strings must NOT appear
    assert "my-secret-project-id" not in result.output
    assert "/home/user/.creds.json" not in result.output
    assert "/Users/avin/vault" not in result.output
    # Section labels must appear
    assert "gcp" in result.output
    assert "integrations" in result.output.lower()


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------


def test_devices_shows_device_list(runner, monkeypatch):
    fake_devices = [
        {"id": 0, "name": "Built-in Microphone", "sample_rate": 44100.0},
        {"id": 3, "name": "USB Headset", "sample_rate": 48000.0},
    ]
    recorder = MagicMock()
    recorder.list_devices.return_value = fake_devices
    monkeypatch.setattr(cli, "_make_recorder", lambda: recorder)

    result = runner.invoke(cli.main, ["devices"])

    assert result.exit_code == 0, result.output
    assert "Built-in Microphone" in result.output
    assert "USB Headset" in result.output


def test_devices_no_devices(runner, monkeypatch):
    recorder = MagicMock()
    recorder.list_devices.return_value = []
    monkeypatch.setattr(cli, "_make_recorder", lambda: recorder)

    result = runner.invoke(cli.main, ["devices"])

    assert result.exit_code == 0, result.output
    assert "No audio input devices found" in result.output
