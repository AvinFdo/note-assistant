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


def test_env_var_override_obsidian_github(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested integrations.obsidian.* GitHub fields are injectable via env vars.

    The token in particular must come from the env (it's a secret, never in YAML).
    """
    p = _write_yaml(tmp_path, _MINIMAL_YAML)
    monkeypatch.setenv("AVIN_OBSIDIAN_GITHUB_REPO", "owner/vault")
    monkeypatch.setenv("AVIN_OBSIDIAN_GITHUB_BRANCH", "notes")
    monkeypatch.setenv("AVIN_OBSIDIAN_GITHUB_TOKEN", "tok_secret")
    cfg = load_config(p)
    assert cfg.integrations.obsidian.github_repo == "owner/vault"
    assert cfg.integrations.obsidian.github_branch == "notes"
    assert cfg.integrations.obsidian.github_token == "tok_secret"


def test_env_var_override_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = _write_yaml(tmp_path, _MINIMAL_YAML)
    monkeypatch.setenv("AVIN_MODELS_TRANSCRIPTION", "gemini-override")
    cfg = load_config(p)
    assert cfg.models.transcription == "gemini-override"


def test_embed_notes_defaults_false(tmp_path: Path) -> None:
    cfg = load_config(_write_yaml(tmp_path, _MINIMAL_YAML))
    assert cfg.memory.embed_notes is False


def test_retrieval_defaults_to_recency(tmp_path: Path) -> None:
    cfg = load_config(_write_yaml(tmp_path, _MINIMAL_YAML))
    assert cfg.memory.retrieval.mode == "recency"
    assert cfg.memory.retrieval.top_k == 6


def test_retrieval_block_parsed(tmp_path: Path) -> None:
    data = dict(_MINIMAL_YAML)
    data["memory"] = {
        **_MINIMAL_YAML["memory"],
        "retrieval": {"mode": "scored", "top_k": 3, "weight_relevance": 2.5},
    }
    cfg = load_config(_write_yaml(tmp_path, data))
    assert cfg.memory.retrieval.mode == "scored"
    assert cfg.memory.retrieval.top_k == 3
    assert cfg.memory.retrieval.weight_relevance == 2.5


def test_embed_notes_env_string_is_parsed_as_bool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env vars arrive as strings; 'true'/'false' must coerce correctly."""
    p = _write_yaml(tmp_path, _MINIMAL_YAML)
    monkeypatch.setenv("AVIN_MEMORY_EMBED_NOTES", "true")
    assert load_config(p).memory.embed_notes is True
    monkeypatch.setenv("AVIN_MEMORY_EMBED_NOTES", "false")
    assert load_config(p).memory.embed_notes is False


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
