"""Tests for assistant.config: YAML loading, env var overrides, and error handling."""

from pathlib import Path

import pytest
import yaml

from assistant.config import Config, load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p


_MINIMAL_YAML = {
    "gcp": {"project_id": "test-project", "region": "us-east1"},
    "models": {"transcription": "gemini-test", "reasoning": "gemini-test"},
    "audio": {
        "sample_rate": 8000,
        "channels": 1,
        "format": "int16",
        "recordings_dir": "rec",
        "default_duration": 5,
    },
    "vad": {
        "enabled": False,
        "model": "silero",
        "threshold": 0.8,
        "min_speech_duration_ms": 100,
        "silence_duration_ms": 500,
        "buffer_duration_s": 1.0,
    },
    "memory": {"db_path": ":memory:", "context_window_size": 3, "max_context_tokens": 1000},
    "actions": {
        "confidence_threshold": 0.9,
        "create_todo": {"mode": "log_only"},
        "send_email": {"mode": "confirm_first"},
        "add_calendar": {"mode": "log_only"},
        "research_topic": {"mode": "log_only"},
    },
    "integrations": {
        "obsidian": {"vault_path": "/vault", "notes_folder": "notes"},
        "google": {"oauth_credentials_path": "/creds.json"},
    },
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_from_yaml(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, _MINIMAL_YAML)
    cfg = load_config(p)

    assert cfg.gcp.project_id == "test-project"
    assert cfg.gcp.region == "us-east1"
    assert cfg.models.transcription == "gemini-test"
    assert cfg.audio.sample_rate == 8000
    assert cfg.vad.enabled is False
    assert cfg.vad.threshold == 0.8
    assert cfg.memory.db_path == ":memory:"
    assert cfg.memory.context_window_size == 3
    assert cfg.actions.confidence_threshold == 0.9
    assert cfg.actions.create_todo.mode == "log_only"
    assert cfg.actions.send_email.mode == "confirm_first"
    assert cfg.integrations.obsidian.vault_path == "/vault"
    assert cfg.integrations.google.oauth_credentials_path == "/creds.json"


def test_returns_config_instance(tmp_path: Path) -> None:
    p = _write_yaml(tmp_path, _MINIMAL_YAML)
    cfg = load_config(p)
    assert isinstance(cfg, Config)


def test_env_var_override_gcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _write_yaml(tmp_path, _MINIMAL_YAML)
    monkeypatch.setenv("AVIN_GCP_PROJECT_ID", "env-project")
    monkeypatch.setenv("AVIN_GCP_REGION", "eu-west1")
    cfg = load_config(p)
    assert cfg.gcp.project_id == "env-project"
    assert cfg.gcp.region == "eu-west1"


def test_env_var_override_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _write_yaml(tmp_path, _MINIMAL_YAML)
    monkeypatch.setenv("AVIN_AUDIO_SAMPLE_RATE", "44100")
    cfg = load_config(p)
    assert cfg.audio.sample_rate == 44100


def test_env_var_override_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _write_yaml(tmp_path, _MINIMAL_YAML)
    monkeypatch.setenv("AVIN_MODELS_TRANSCRIPTION", "gemini-override")
    cfg = load_config(p)
    assert cfg.models.transcription == "gemini-override"


def test_avin_config_path_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _write_yaml(tmp_path, _MINIMAL_YAML)
    monkeypatch.setenv("AVIN_CONFIG_PATH", str(p))
    monkeypatch.delenv("AVIN_GCP_PROJECT_ID", raising=False)
    cfg = load_config()  # no path arg — should pick up AVIN_CONFIG_PATH
    assert cfg.gcp.project_id == "test-project"


def test_missing_config_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_empty_yaml_uses_defaults(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    cfg = load_config(p)
    assert cfg.gcp.project_id == ""
    assert cfg.audio.sample_rate == 16000
    assert cfg.actions.send_email.mode == "confirm_first"


def test_singleton_is_config_instance() -> None:
    from assistant.config import config

    assert isinstance(config, Config)
    assert config.gcp.project_id != ""  # loaded from real default.yaml
