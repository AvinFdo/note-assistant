"""Tests for the avin CLI `listen` command and shared processing pipeline.

All tests run fully offline: the hardware/API factory functions
(`_make_recorder`, `_make_transcriber`, `_make_brain`, `_make_memory`) are
monkeypatched to return fakes, so no microphone, sounddevice stream, or live
Gemini call is ever made.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from assistant import cli
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

    def process(self, text: str):
        self.process_calls.append(text)
        cid = self._memory.save_conversation(text)
        for intent, details in self._actions:
            # execution_mode here is irrelevant — route_action reads mode from config
            self._memory.save_action(cid, intent, details, "pending")
        return None


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
