"""Loads config/default.yaml and exposes a typed Config singleton.

Override any value with an AVIN_<SECTION>_<KEY> environment variable,
e.g. AVIN_GCP_PROJECT_ID overrides gcp.project_id.
Point AVIN_CONFIG_PATH to a custom YAML file to replace the default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Dataclass hierarchy
# ---------------------------------------------------------------------------


@dataclass
class GcpConfig:
    project_id: str = ""
    region: str = "us-central1"


@dataclass
class ModelsConfig:
    transcription: str = "gemini-2.5-flash"
    reasoning: str = "gemini-2.5-flash"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    format: str = "int16"
    recordings_dir: str = "recordings"
    default_duration: int = 10


@dataclass
class VadConfig:
    enabled: bool = True
    model: str = "silero"
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    silence_duration_ms: int = 1500
    buffer_duration_s: float = 3.0


@dataclass
class MemoryConfig:
    db_path: str = "data/assistant.db"
    context_window_size: int = 5
    max_context_tokens: int = 4000
    min_transcript_words: int = 10


@dataclass
class ActionModeConfig:
    mode: str = "auto_execute"


@dataclass
class ActionsConfig:
    confidence_threshold: float = 0.7
    create_todo: ActionModeConfig = field(default_factory=lambda: ActionModeConfig("auto_execute"))
    send_email: ActionModeConfig = field(default_factory=lambda: ActionModeConfig("confirm_first"))
    add_calendar: ActionModeConfig = field(
        default_factory=lambda: ActionModeConfig("confirm_first")
    )
    research_topic: ActionModeConfig = field(
        default_factory=lambda: ActionModeConfig("auto_execute")
    )


@dataclass
class ObsidianConfig:
    vault_path: str = ""
    notes_folder: str = "assistant"


@dataclass
class GoogleIntegrationConfig:
    oauth_credentials_path: str = ""


@dataclass
class IntegrationsConfig:
    obsidian: ObsidianConfig = field(default_factory=ObsidianConfig)
    google: GoogleIntegrationConfig = field(default_factory=GoogleIntegrationConfig)


@dataclass
class ApiConfig:
    """API-level settings, including authentication keys.

    ``api_keys`` is the list of valid API keys accepted by the server.
    Leave it empty (the default) to disable authentication — useful for local
    development.  In production, inject keys via the ``AVIN_API_KEYS``
    environment variable (comma-separated) or the ``api.api_keys`` YAML list.
    """

    api_keys: list[str] = field(default_factory=list)


@dataclass
class Config:
    gcp: GcpConfig = field(default_factory=GcpConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    actions: ActionsConfig = field(default_factory=ActionsConfig)
    integrations: IntegrationsConfig = field(default_factory=IntegrationsConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "default.yaml"

# Top-level section names — used to parse AVIN_<SECTION>_<KEY> env vars.
_SECTIONS = ("integrations", "models", "memory", "actions", "audio", "gcp", "vad", "api")


def _apply_env_overrides(data: dict) -> None:
    """Mutate *data* in-place with any AVIN_* environment variable overrides.

    Special cases:
    - ``AVIN_API_KEYS``: comma-separated string parsed into a list and stored
      under ``data["api"]["api_keys"]``.  The generic mechanism would store a
      raw string, which is wrong for a list field, so we handle it explicitly
      here before the generic loop runs.
    """
    # --- Explicit list overrides ---
    avin_api_keys = os.environ.get("AVIN_API_KEYS")
    if avin_api_keys is not None:
        keys = [k.strip() for k in avin_api_keys.split(",") if k.strip()]
        data.setdefault("api", {})["api_keys"] = keys

    # --- Generic scalar overrides ---
    for env_key, value in os.environ.items():
        if not env_key.startswith("AVIN_"):
            continue
        rest = env_key[5:].lower()  # strip prefix, lowercase
        for section in _SECTIONS:
            prefix = section + "_"
            if rest.startswith(prefix):
                key = rest[len(prefix) :]
                # Skip api_keys — already handled above as a list.
                if section == "api" and key == "api_keys":
                    break
                data.setdefault(section, {})[key] = value
                break


def _parse_action_mode(raw: object, default: str = "auto_execute") -> ActionModeConfig:
    if isinstance(raw, dict):
        return ActionModeConfig(mode=str(raw.get("mode", default)))
    return ActionModeConfig(mode=default)


def _parse_config(data: dict) -> Config:
    gcp_raw = data.get("gcp", {})
    models_raw = data.get("models", {})
    audio_raw = data.get("audio", {})
    vad_raw = data.get("vad", {})
    mem_raw = data.get("memory", {})
    act_raw = data.get("actions", {})
    int_raw = data.get("integrations", {})
    obs_raw = int_raw.get("obsidian", {})
    goog_raw = int_raw.get("google", {})
    api_raw = data.get("api", {})

    return Config(
        gcp=GcpConfig(
            project_id=str(gcp_raw.get("project_id", "")),
            region=str(gcp_raw.get("region", "us-central1")),
        ),
        models=ModelsConfig(
            transcription=str(models_raw.get("transcription", "gemini-2.5-flash")),
            reasoning=str(models_raw.get("reasoning", "gemini-2.5-flash")),
        ),
        audio=AudioConfig(
            sample_rate=int(audio_raw.get("sample_rate", 16000)),
            channels=int(audio_raw.get("channels", 1)),
            format=str(audio_raw.get("format", "int16")),
            recordings_dir=str(audio_raw.get("recordings_dir", "recordings")),
            default_duration=int(audio_raw.get("default_duration", 10)),
        ),
        vad=VadConfig(
            enabled=bool(vad_raw.get("enabled", True)),
            model=str(vad_raw.get("model", "silero")),
            threshold=float(vad_raw.get("threshold", 0.5)),
            min_speech_duration_ms=int(vad_raw.get("min_speech_duration_ms", 250)),
            silence_duration_ms=int(vad_raw.get("silence_duration_ms", 1500)),
            buffer_duration_s=float(vad_raw.get("buffer_duration_s", 3.0)),
        ),
        memory=MemoryConfig(
            db_path=str(mem_raw.get("db_path", "data/assistant.db")),
            context_window_size=int(mem_raw.get("context_window_size", 5)),
            max_context_tokens=int(mem_raw.get("max_context_tokens", 4000)),
            min_transcript_words=int(mem_raw.get("min_transcript_words", 10)),
        ),
        actions=ActionsConfig(
            confidence_threshold=float(act_raw.get("confidence_threshold", 0.7)),
            create_todo=_parse_action_mode(act_raw.get("create_todo"), "auto_execute"),
            send_email=_parse_action_mode(act_raw.get("send_email"), "confirm_first"),
            add_calendar=_parse_action_mode(act_raw.get("add_calendar"), "confirm_first"),
            research_topic=_parse_action_mode(act_raw.get("research_topic"), "auto_execute"),
        ),
        integrations=IntegrationsConfig(
            obsidian=ObsidianConfig(
                vault_path=str(obs_raw.get("vault_path", "")),
                notes_folder=str(obs_raw.get("notes_folder", "assistant")),
            ),
            google=GoogleIntegrationConfig(
                oauth_credentials_path=str(goog_raw.get("oauth_credentials_path", "")),
            ),
        ),
        api=ApiConfig(
            # api_keys must be a list of strings; guard against YAML scalars.
            api_keys=list(api_raw.get("api_keys") or []),
        ),
    )


def load_config(path: Path | None = None) -> Config:
    """Load and return a Config from *path* (default: config/default.yaml)."""
    resolved = path or Path(os.environ.get("AVIN_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH)))
    with open(resolved) as f:
        data = yaml.safe_load(f) or {}
    _apply_env_overrides(data)
    return _parse_config(data)


# Singleton — imported everywhere as `from assistant.config import config`
config: Config = load_config()
