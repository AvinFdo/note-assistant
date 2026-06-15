# Avin — Personal Assistant Backlog

> [!IMPORTANT]
> **🎯 Current Focus: Google Workspace actions (2.2.1–2.2.6)** _(Full stack is LIVE end-to-end on the user's phone: Cloudflare Pages → Worker (proxy-secret) → Cloud Run + Firestore. Scored semantic retrieval (2.3.2) shipped + live. Notes: manual delete + retention/purge shipped. Next big gap: actions are detected but never executed — wire OAuth (2.2.1) then Calendar/Tasks/Gmail/Docs (email stays confirm-first). Also open: 2.4.3 billing/monitoring; Obsidian per-note-file vault refactor; 1.3.2 noise test needs human + mic.)_
> Update this marker when you complete a task. Work top-to-bottom, respecting `Depends on:`.

> [!NOTE]
> Each task includes **Context** (why it matters), **Details** (what to implement), and **Done When** (acceptance criteria).
> Dependencies between tasks are noted where relevant. Tasks within a sub-stage are ordered by dependency.

---

## Stage 1: Local Prototype (Solid Foundation)

### 1.1 — Project Hygiene & Structure
> **Why this first:** Every subsequent task depends on a clean project structure. Doing this first prevents accruing tech debt from day one.

---

#### `[x]` 1.1.1 — Initialize Git repo and `.gitignore`
**Context:** The project currently has no version control. We need Git from the start so every change is tracked and we can use branches/PRs.
**Details:**
- Run `git init` in `note-assistant/`
- Create `.gitignore` covering:
  ```
  venv/
  __pycache__/
  *.pyc
  *.wav
  data/
  recordings/
  .env
  .DS_Store
  dist/
  *.egg-info/
  ```
- Make initial commit with just `.gitignore` and empty `README.md`
- Create a GitHub repo and push

**Done when:** Repo exists on GitHub, `.gitignore` prevents committing generated files, `main` branch is protected.

---

#### `[x]` 1.1.2 — Create `pyproject.toml` with pinned dependencies
**Context:** The current venv was set up with ad-hoc `pip install` commands. There's no record of what versions are installed. Anyone cloning the repo can't reproduce the environment.
**Details:**
- Create `pyproject.toml` using the `[project]` table (PEP 621)
- Pin major+minor versions for core dependencies:
  ```toml
  dependencies = [
      "google-genai>=2.8,<3.0",
      "sounddevice>=0.5,<1.0",
      "numpy>=2.0,<3.0",
      "scipy>=1.13,<2.0",
      "pyyaml>=6.0,<7.0",
      "rich>=13.0,<14.0",
      "click>=8.0,<9.0",
  ]
  
  [project.optional-dependencies]
  dev = ["pytest>=8.0", "pytest-asyncio", "ruff"]
  
  [project.scripts]
  avin = "assistant.cli:main"
  ```
- Add `[tool.ruff]` config for linting
- Add `[tool.pytest.ini_options]` config
- Delete the old `venv/`, recreate with `python -m venv venv && pip install -e ".[dev]"`

**Done when:** `pip install -e ".[dev]"` works cleanly from a fresh venv. `avin` CLI entry point is registered.

---

#### `[x]` 1.1.3 — Scaffold package layout
**Context:** All source code currently lives as flat scripts in the project root. We need a proper Python package so modules can import each other cleanly and tests can run.
**Details:**
- Create the directory structure as defined in [Section 7 of the Project Brief](file:///Users/avinfernando/.gemini/antigravity/brain/fdc908c9-c3a0-4545-a576-e171e82fdd8f/implementation_plan.md):
  ```
  src/assistant/        — all source modules (empty __init__.py files for now)
  src/assistant/actions/ — action sub-package
  tests/                — test directory with conftest.py
  config/               — default.yaml
  recordings/           — empty dir with .gitkeep
  data/                 — empty dir with .gitkeep
  ```
- Every `.py` file should have a module docstring explaining its purpose
- Create placeholder classes with `pass` bodies — just enough to verify the import graph works

**Done when:** `python -c "from assistant.config import Config; from assistant.brain import Brain; from assistant.actions.todo import CreateTodoAction"` runs without errors.

---

#### `[x]` 1.1.4 — Create `config/default.yaml` and `config.py`
**Context:** The old code had hardcoded values for project ID, model name, region, sample rate, and language. Config must be externalized so we can change behavior without editing source code. This is especially important when we deploy to different environments (local vs cloud vs Pi).
**Details:**
- Create `config/default.yaml` with the full schema defined in [Section 5 of the Project Brief](file:///Users/avinfernando/.gemini/antigravity/brain/fdc908c9-c3a0-4545-a576-e171e82fdd8f/implementation_plan.md) (GCP settings, model names, audio params, VAD params, memory settings, action modes)
- Create `config.py` that:
  - Loads `default.yaml` using PyYAML
  - Supports environment variable overrides (e.g., `AVIN_GCP_PROJECT_ID` overrides `gcp.project_id`)
  - Exposes config as a typed dataclass or Pydantic model for IDE autocompletion
  - Is a singleton — loaded once, importable everywhere
- Support `AVIN_CONFIG_PATH` env var to point to a custom YAML file (for different environments)

**Done when:** `from assistant.config import config; print(config.gcp.project_id)` returns the value from YAML. Environment variable override works. Test in `test_config.py` passes.

**Depends on:** 1.1.3

---

#### `[x]` 1.1.5 — Write `README.md`
**Context:** Documentation for anyone (including future-you) picking up this project.
**Details:**
- Project overview and vision (1 paragraph)
- Architecture diagram (embed the mermaid diagram from the project brief)
- Prerequisites (Python 3.12+, GCP account, ADC auth)
- Setup instructions (clone, create venv, install, configure)
- Usage examples (`avin listen`, `avin history`, etc.)
- Project structure explanation
- Contributing guidelines (branch naming, PR process)

**Done when:** A new developer can follow the README from zero to running the CLI.

---

### 1.2 — Audio Capture
> **Why:** This is the entry point of the entire pipeline. If audio capture is unreliable, nothing downstream works.
> **Depends on:** 1.1 (config.py must exist so we read sample_rate etc. from config)

---

#### `[x]` 1.2.1 — Implement `audio.py` with configurable recording
**Context:** The old `audio.py` worked but had no error handling, saved files to the wrong directory, and had hardcoded values.
**Details:**
- Create `AudioRecorder` class that reads settings from `config.audio`
- `record(duration_seconds) -> Path` — records for a fixed duration, returns path to WAV file
- `list_devices() -> list[dict]` — returns available audio input devices with their IDs, names, and sample rates
- Save all recordings to `config.audio.recordings_dir` (default: `recordings/`)
- Filename format: `recording_YYYYMMDD_HHMMSS.wav`
- Error handling:
  - `NoMicrophoneError` — no input device found
  - `PermissionError` — microphone access denied (macOS security)
  - `DeviceBusyError` — device in use by another app
- Log audio level (RMS) during recording to help debug silence issues

**Done when:** 
- `recorder.record(5)` saves a WAV to `recordings/` with correct format
- `recorder.list_devices()` shows available microphones
- Running with no mic raises `NoMicrophoneError` with a helpful message
- Config values (sample_rate, channels) are read from YAML, not hardcoded

---

### 1.3 — Unified STT (via google-genai)
> **Why:** We currently depend on `google-cloud-speech`, a separate library with different auth patterns. The `google-genai` SDK can handle audio natively via Gemini's multimodal input, letting us drop an entire dependency tree.
> **Depends on:** 1.1 (config), 1.2 (audio.py produces WAV files)

---

#### `[x]` 1.3.1 — Implement `transcriber.py` using google-genai multimodal
**Context:** Gemini models accept audio files as input. We can send a WAV file and ask the model to transcribe it, replacing the need for a separate Speech-to-Text API.
**Details:**
- Create `Transcriber` class
- Initialize `genai.Client(vertexai=True, project=config.gcp.project_id, location=config.gcp.region)`
- `transcribe(audio_path: Path) -> TranscriptionResult` method:
  - Reads the WAV file
  - Sends to Gemini with a simple prompt: "Transcribe the following audio exactly as spoken. Return only the transcription, no commentary."
  - Returns a `TranscriptionResult` dataclass: `{ text: str, confidence: float, duration_ms: int }`
- Custom exception hierarchy:
  - `TranscriptionError` (base)
  - `AuthenticationError` — ADC not configured or expired
  - `NetworkError` — can't reach the API
  - `SilenceError` — audio contained no detectable speech
  - `QuotaExceededError` — API rate limit hit
- Retry logic: retry on `NetworkError` up to 3 times with exponential backoff

**Done when:**
- `transcriber.transcribe("recordings/test.wav")` returns accurate text
- Each error type is raised in the correct scenario (testable with mocks)
- `google-cloud-speech` is removed from `pyproject.toml` and venv

---

#### `[ ]` 1.3.2 — Test transcription accuracy in noisy environment
**Context:** The user works in an open office. We need to validate that Gemini multimodal can handle moderate background noise before building on top of it.
**Details:**
- Record 5-10 test samples with varying noise levels (quiet room, fan noise, multiple speakers)
- Transcribe each and measure Word Error Rate (WER) by comparing to known text
- Document results and thresholds — if accuracy drops below 80% in noisy conditions, we may need to add a noise reduction preprocessing step (e.g., `noisereduce` library)

**Done when:** Test results documented. Decision made on whether noise preprocessing is needed.

---

### 1.4 — Persistence & Memory
> **Why:** Without persistence, the assistant has no memory. Notes are lost after each session. This is the foundation of the "second brain" vision.
> **Depends on:** 1.1 (config for db_path)

---

#### `[x]` 1.4.1 — Design and implement SQLite schema
**Context:** We need to store conversations, notes, actions, and rolling context. The schema is defined in [Section 4 of the Project Brief](file:///Users/avinfernando/.gemini/antigravity/brain/fdc908c9-c3a0-4545-a576-e171e82fdd8f/implementation_plan.md).
**Details:**
- Create `memory.py` with a `Memory` class
- `__init__` creates the DB file at `config.memory.db_path` and runs schema migrations (create tables if not exist)
- CRUD methods:
  - `save_conversation(transcript, audio_path=None) -> conversation_id`
  - `save_note(conversation_id, summary, is_noteworthy) -> note_id`
  - `save_action(conversation_id, intent, details, execution_mode) -> action_id`
  - `update_action_status(action_id, status)`
  - `get_recent_conversations(limit=5) -> list[Conversation]`
  - `get_recent_notes(limit=20) -> list[Note]`
  - `get_pending_actions() -> list[Action]`
  - `search_notes(query: str) -> list[Note]` — basic LIKE search for now, vector search in Stage 2
  - `set_context(key, value)` / `get_context(key) -> str` / `get_all_context() -> dict`
- Use dataclasses for return types: `Conversation`, `Note`, `Action`
- Use Python's built-in `sqlite3` module (no ORM overhead for this scale)
- UUIDs generated via `uuid.uuid4()` as strings

**Done when:**
- All CRUD operations work (verified by `test_memory.py` using an in-memory SQLite DB)
- DB file is auto-created on first run
- Schema is idempotent (running twice doesn't error)

---

#### `[x]` 1.4.2 — Implement context retrieval for LLM prompt enrichment
**Context:** The LLM needs context from previous conversations to provide continuity. This method assembles the "memory" section of the prompt.
**Details:**
- Add method `assemble_context() -> str` to `Memory` class
- Retrieves:
  1. Last N conversation summaries (from `notes` table, limited by `config.memory.context_window_size`)
  2. Pending/recent actions (last 10 from `actions` table)
  3. All key-value pairs from `context_window` table
- Formats them as a structured string suitable for inclusion in the LLM prompt (see [Section 6 of the Project Brief](file:///Users/avinfernando/.gemini/antigravity/brain/fdc908c9-c3a0-4545-a576-e171e82fdd8f/implementation_plan.md))
- Respects `config.memory.max_context_tokens` — truncates oldest entries if the assembled context exceeds the token budget (approximate with char count / 4)

**Done when:** `memory.assemble_context()` returns a formatted string with recent history, pending actions, and context keys. Truncation works when history is large.

**Depends on:** 1.4.1

---

### 1.5 — Context-Aware LLM Processing
> **Why:** This is the brain of the assistant. It takes raw transcripts and produces structured, actionable output with awareness of past conversations.
> **Depends on:** 1.1 (config), 1.3 (transcriber), 1.4 (memory for context retrieval)

---

#### `[x]` 1.5.1 — Implement `brain.py` with context-aware prompting
**Context:** The old `llm.py` sent each transcript in isolation with no history. The new `brain.py` must assemble context from memory and use Gemini's structured output mode.
**Details:**
- Create `Brain` class that takes `Memory` and `config` as dependencies
- `process(transcript: str) -> ProcessingResult` method:
  1. Calls `memory.assemble_context()` to get recent history
  2. Assembles the full prompt using the template from [Section 6 of the Project Brief](file:///Users/avinfernando/.gemini/antigravity/brain/fdc908c9-c3a0-4545-a576-e171e82fdd8f/implementation_plan.md)
  3. Sends to Gemini with `response_mime_type="application/json"` and the response schema defined in the brief
  4. Parses the structured JSON response into a `ProcessingResult` dataclass:
     ```python
     @dataclass
     class ActionItem:
         intent: str       # 'create_todo' | 'send_email' | 'add_calendar' | 'research_topic'
         confidence: float  # 0.0 to 1.0
         details: dict      # action-specific fields
     
     @dataclass
     class ProcessingResult:
         is_noteworthy: bool
         summary_note: str
         actions: list[ActionItem]
     ```
  5. Saves conversation, note (if noteworthy), and actions to memory
- Add a `confidence_threshold` config value (default: 0.7) — actions below this threshold are saved with `status="low_confidence"` and not routed for execution

**Done when:**
- Given a transcript + mock context, produces correct structured output
- Context from previous conversations influences the output (testable with sequential calls)
- Low-confidence actions are logged but not executed
- Unit tests pass with mocked Gemini responses (no live API calls in tests)

---

#### `[x]` 1.5.2 — Implement relevance filter
**Context:** Not everything spoken is worth saving. "Hey, how's it going?" doesn't need to be a note. The LLM's `is_noteworthy` field handles this, but we also need guardrails.
**Details:**
- If `is_noteworthy` is `false`, still save the conversation transcript (for raw history) but don't create a note or extract actions
- Add a minimum transcript length threshold (e.g., < 10 words → skip processing entirely, save transcript only)
- Log filtered-out transcripts at DEBUG level for review

**Done when:** Short/trivial transcripts don't generate notes. Conversation is still saved for context continuity.

**Depends on:** 1.5.1

---

### 1.6 — Action Framework
> **Why:** Actions are how the assistant actually does things. The framework must be extensible (easy to add new action types) and safe (configurable execution modes).
> **Depends on:** 1.4 (memory for persistence), 1.5 (brain produces action items)

---

#### `[x]` 1.6.1 — Create abstract `Action` base class and registry
**Context:** We need a pluggable action system. Each action type (todo, email, calendar, research) implements the same interface, and a router dispatches to the right one.
**Details:**
- `base.py`:
  ```python
  class Action(ABC):
      @abstractmethod
      def execute(self, details: dict) -> str:
          """Execute the action. Returns a human-readable result message."""
          
      @abstractmethod
      def describe(self, details: dict) -> str:
          """Return a human-readable description for confirmation prompts."""
  ```
- `__init__.py` contains:
  - `ACTION_REGISTRY: dict[str, Action]` — maps intent strings to Action instances
  - `route_action(action_item: ActionItem, memory: Memory) -> str` — looks up the action in the registry, checks execution mode from config, and either executes, prompts for confirmation, or logs
  - Auto-registers all Action subclasses on import

**Done when:** `route_action(ActionItem(intent="create_todo", ...))` dispatches to `CreateTodoAction.execute()`. Unknown intents raise `UnknownActionError`.

---

#### `[x]` 1.6.2 — Implement mock actions with SQLite persistence
**Context:** In Stage 1, actions don't connect to real APIs. They log to console and update their status in SQLite.
**Details:**
- `todo.py`: `CreateTodoAction` — prints `[TODO] {task}`, updates action status to "executed"
- `email.py`: `SendEmailAction` — prints `[EMAIL] To: {recipient}, Subject: {subject}`, updates status
- `calendar.py`: `AddCalendarAction` — prints `[CALENDAR] {title} at {time}`, updates status
- `research.py`: `ResearchTopicAction` — prints `[RESEARCH] {topic}`, updates status
- Each mock action also saves a human-readable description to the action's `details` in SQLite
- Respect `execution_mode` from config:
  - `auto_execute`: execute immediately, print result
  - `confirm_first`: print description, prompt user for y/n in CLI, then execute or dismiss
  - `log_only`: save to DB with status="logged", don't execute

**Done when:** Actions are saved to SQLite with correct status. `confirm_first` mode prompts user. All testable without live APIs.

**Depends on:** 1.6.1

---

### 1.7 — Continuous Listening (VAD)
> **Why:** Fixed-duration recording (e.g., 5 seconds) is useless for a real assistant. We need continuous listening with intelligent speech segmentation.
> **Depends on:** 1.2 (audio capture), 1.3 (transcriber)

---

#### `[x]` 1.7.1 — Integrate Silero VAD
**Context:** Silero VAD is a lightweight voice activity detector (~2MB PyTorch model) that runs in real-time on CPU. It determines whether each audio frame contains speech or silence.
**Details:**
- Add `torch` and `torchaudio` to dependencies (CPU-only build to keep size small)
- Create `vad.py` with a `VADProcessor` class:
  - `__init__`: loads Silero model, reads config from `config.vad`
  - `process_frame(audio_frame: np.ndarray) -> bool` — returns True if speech detected
  - Configurable `threshold` (0.0-1.0), `min_speech_duration_ms`, `silence_duration_ms`
- The VAD operates on 30ms frames (480 samples at 16kHz)

**Done when:** `vad.process_frame(frame)` correctly returns True for speech and False for silence. Configurable via YAML.

---

#### `[x]` 1.7.2 — Implement continuous capture with speech segmentation
**Context:** Instead of recording for N seconds, the assistant should listen continuously and produce audio segments whenever a speech-to-silence transition occurs.
**Details:**
- Add `listen_continuous(callback: Callable[[Path], None])` method to `AudioRecorder`:
  1. Opens an audio stream in callback mode (non-blocking)
  2. Maintains a rolling buffer of `config.vad.buffer_duration_s` seconds (captures pre-speech audio so we don't miss sentence beginnings)
  3. Feeds each frame to `VADProcessor`
  4. When speech starts: begin accumulating frames (prepend the rolling buffer)
  5. When silence exceeds `config.vad.silence_duration_ms`: save accumulated audio as a WAV, call `callback(wav_path)`
  6. Reset and continue listening
- The callback is where the rest of the pipeline hooks in (transcribe → brain → actions)
- Add graceful shutdown on Ctrl+C

**Done when:** Running in continuous mode, speaking a sentence, then going silent produces a WAV file containing the full sentence (including the beginning). Multiple sentences produce multiple files. Ctrl+C exits cleanly. (state machine unit-tested offline; live-mic end-to-end validation pending)

**Depends on:** 1.7.1, 1.2

---

#### `[x]` 1.7.3 — Add `--continuous` and `--duration` CLI flags _(CLI + pipeline unit-tested offline; live end-to-end pending)_
**Context:** Users should be able to choose between continuous listening and fixed-duration recording.
**Details:**
- `avin listen` — default, records for `config.audio.default_duration` seconds (e.g., 10s)
- `avin listen --continuous` — continuous VAD-based listening until Ctrl+C
- `avin listen --duration 30` — record for exactly 30 seconds
- Both modes feed into the same processing pipeline (transcribe → brain → actions)

**Done when:** Both modes work end-to-end. `--continuous` processes multiple speech segments in sequence.

**Depends on:** 1.7.2

---

### 1.8 — CLI Polish & Testing
> **Why:** A good CLI makes the tool pleasant to use daily. Tests prevent regressions as we iterate.
> **Depends on:** All previous 1.x tasks

---

#### `[x]` 1.8.1 — Build CLI using click + rich
**Context:** The current entry point is a bare `main.py` with `argparse`. We need a proper CLI with subcommands, colors, and structured output.
**Details:**
- Use `click` for command structure and argument parsing
- Use `rich` for colored/formatted output (tables, panels, spinners)
- Commands:
  - `avin listen` — record and process (with `--continuous` and `--duration` flags)
  - `avin history` — show recent notes in a rich table (timestamp, summary, action count)
  - `avin search <query>` — search notes by keyword, display results
  - `avin actions` — show pending/recent actions with status
  - `avin actions confirm <id>` — confirm a pending action
  - `avin replay <conversation_id>` — show full transcript and extracted data for a past conversation
  - `avin config` — print current configuration (with sensitive values masked)
  - `avin devices` — list available audio input devices
- Show a spinner while waiting for API responses
- Show a live indicator during recording (e.g., "🔴 Recording..." with elapsed time)

**Done when:** All commands work. Output is colorized and well-formatted. `avin --help` shows all commands with descriptions.

---

#### `[x]` 1.8.2 — Write unit tests _(90% total coverage; conftest fixtures added; enforced via --cov-fail-under=80)_
**Context:** We need automated tests to catch regressions, especially for the brain (LLM parsing) and memory (database operations) which are the most complex and fragile parts.
**Details:**
- `tests/conftest.py`:
  - Fixture: in-memory SQLite database (`:memory:`)
  - Fixture: mock config loaded from a test YAML
  - Fixture: mock `genai.Client` that returns predefined responses
- `tests/test_memory.py`:
  - Test CRUD for all 4 tables
  - Test `assemble_context()` with various history lengths
  - Test `search_notes()` with matching and non-matching queries
  - Test context window size truncation
- `tests/test_brain.py`:
  - Test with mock LLM response containing multiple actions
  - Test with mock LLM response where `is_noteworthy=false`
  - Test confidence threshold filtering
  - Test that context is included in the prompt (inspect the assembled prompt)
- `tests/test_actions.py`:
  - Test action routing for each intent type
  - Test `auto_execute`, `confirm_first`, `log_only` modes
  - Test unknown intent handling
- `tests/test_config.py`:
  - Test loading from YAML
  - Test environment variable override
  - Test missing config file error

**Done when:** `pytest` passes with >80% code coverage on `brain.py`, `memory.py`, `actions/`, and `config.py`.

---

#### `[x]` 1.8.3 — Set up GitHub Actions CI
**Context:** Automated testing on every push prevents broken code from being merged.
**Details:**
- Create `.github/workflows/ci.yml`:
  - Triggers on push to `main` and on PRs
  - Matrix: Python 3.12, 3.13
  - Steps: checkout → setup Python → install deps → lint (ruff) → test (pytest)
  - No live API calls in CI (all tests use mocks)
- Add status badge to README.md

**Done when:** Push to GitHub triggers CI. Green badge shows on README.

---

## Stage 2: Cloud Service & Integrations

### 2.1 — API Server

#### `[x]` 2.1.1 — Create FastAPI application
**Context:** We need to expose the processing pipeline as an API so clients (web, mobile, Raspberry Pi) can send audio and receive notes remotely.
**Details:**
- Create `src/assistant/api/` package with `app.py`, `routes.py`, `schemas.py`, `auth.py`
- Mount the existing `Brain`, `Memory`, and action router as FastAPI dependencies
- REST endpoints as defined in [Section 9 of the Project Brief](file:///Users/avinfernando/.gemini/antigravity/brain/fdc908c9-c3a0-4545-a576-e171e82fdd8f/implementation_plan.md): `GET /notes`, `GET /notes/{id}`, `GET /actions`, `PATCH /actions/{id}`, `POST /search`, `GET /context`
- Pydantic models for request/response validation
- Auto-generated OpenAPI docs at `/docs`

**Done when:** `uvicorn assistant.api.app:app` starts. All REST endpoints return correct data from SQLite. `/docs` shows interactive API docs.

---

#### `[x]` 2.1.2 — WebSocket endpoint for real-time audio streaming (server-side VAD segmentation reused; logic unit-tested offline)
**Context:** For the Raspberry Pi and future clients, we need real-time streaming rather than file upload.
**Details:**
- `WS /api/v1/stream` endpoint
- Accepts binary frames (raw PCM audio) and text frames (control messages)
- Server-side: buffers audio, runs VAD, segments on silence, processes through pipeline
- Sends back JSON messages: `transcript`, `note`, `action`, `error` (see [Section 9 WebSocket contract](file:///Users/avinfernando/.gemini/antigravity/brain/fdc908c9-c3a0-4545-a576-e171e82fdd8f/implementation_plan.md))
- Handle client disconnection gracefully (flush pending audio)

**Done when:** A test client can stream audio over WebSocket and receive real-time notes/actions back.

**Depends on:** 2.1.1

---

#### `[x]` 2.1.3 — Add authentication
**Context:** The API will be internet-facing. We need to prevent unauthorized access.
**Details:**
- Start simple: API key authentication via `X-API-Key` header
- Store valid API keys in Firestore or environment variable
- Optionally: Firebase Auth (JWT-based) for more robust user management
- All endpoints require auth except `/health`

**Done when:** Requests without a valid API key return 401. Requests with a valid key succeed.

**Depends on:** 2.1.1

---

### 2.2 — Real Integrations (Replace Mocks)

#### `[ ]` 2.2.1 — Implement OAuth2 flow for Google Workspace APIs
**Context:** Calendar, Tasks, Gmail, and Docs all require OAuth2 user consent. This is a one-time setup per user.
**Details:**
- Create OAuth2 client credentials in Google Cloud Console
- Implement the consent flow: user visits a URL, grants access, callback stores refresh token
- Store refresh tokens securely (Firestore or encrypted local file)
- Use `google-auth` library's `InstalledAppFlow` for the CLI, and redirect-based flow for the web API
- Scopes needed: `calendar.events`, `tasks`, `gmail.compose`, `gmail.send`, `documents`

**Done when:** Running the OAuth flow grants access. Refresh token persists across restarts.

---

#### `[ ]` 2.2.2 — Google Calendar integration
**Context:** Replace `AddCalendarAction` mock with real Google Calendar API calls.
**Details:**
- Use `google-api-python-client` with the stored OAuth2 credentials
- `execute()`: creates a calendar event with title, start time, end time, attendees
- Parse natural language time expressions from the LLM (e.g., "tomorrow at 2pm") into ISO 8601 datetime — use `dateutil.parser` or ask the LLM to output ISO format
- `describe()`: "Create event: 'Team Meeting' on June 10 at 2:00 PM"
- Handle edge cases: all-day events, recurring events, timezone handling

**Done when:** Saying "schedule a meeting with John tomorrow at 3pm" creates a real Google Calendar event.

**Depends on:** 2.2.1

---

#### `[ ]` 2.2.3 — Google Tasks integration
**Context:** Replace `CreateTodoAction` mock with real Google Tasks API calls.
**Details:**
- Create tasks in the user's default task list
- Support due dates if mentioned
- `describe()`: "Add to-do: 'Buy groceries' (due: tomorrow)"

**Done when:** Saying "remind me to buy groceries" creates a real Google Tasks item.

**Depends on:** 2.2.1

---

#### `[ ]` 2.2.4 — Gmail integration
**Context:** Replace `SendEmailAction` mock. **This is the most dangerous action** — sending emails on behalf of the user.
**Details:**
- Default execution mode: `confirm_first` (NEVER auto-send)
- `execute()`: creates and sends an email via Gmail API
- `describe()`: shows full email preview (to, subject, body) before confirmation
- Support drafts: optionally save as draft instead of sending
- Add a config flag: `gmail.allow_auto_send: false` (global safety switch)

**Done when:** Saying "email John about the proposal" creates a draft or sends (after confirmation). Safety switch prevents accidental sends.

**Depends on:** 2.2.1

---

#### `[x]` 2.2.5 — Obsidian integration (local markdown writer; pipeline hook gated on configured vault)
**Context:** Save notes as markdown files into the user's Obsidian vault.
**Details:**
- Read vault path from `config.integrations.obsidian.vault_path`
- Write markdown files to `{vault_path}/{notes_folder}/YYYY-MM-DD.md`
- Append to daily note if it already exists
- Format: timestamp, summary, actions list
- If vault is synced via iCloud/Git, the notes will automatically appear in Obsidian

**Done when:** Processing a transcript creates/appends a markdown file in the Obsidian vault with the note content.

---

#### `[ ]` 2.2.6 — Google Docs integration
**Context:** Create or append to Google Docs for longer-form notes or meeting summaries.
**Details:**
- `execute()`: creates a new Google Doc or appends to an existing one
- Support a `doc_id` field in action details to append to a specific document
- Format content with proper headings, timestamps, and bullet points

**Done when:** Processing a transcript can create a new Google Doc with the summary content.

**Depends on:** 2.2.1

---

### 2.3 — Persistence Migration

#### `[x]` 2.3.1 — Migrate from SQLite to Firestore _(FirestoreMemory + sqlite|firestore switch; 23 offline tests w/ fake client. Real-Firestore/emulator parity + enabling the TTL policy pending.)_
**Context:** SQLite doesn't work in serverless environments (Cloud Run instances are ephemeral). Firestore is serverless, GCP-native, and has a generous free tier.
**Details:**
- Create a `FirestoreMemory` class implementing the same interface as `Memory`
- Schema as defined in [Section 4 of the Project Brief](file:///Users/avinfernando/.gemini/antigravity/brain/fdc908c9-c3a0-4545-a576-e171e82fdd8f/implementation_plan.md) (Firestore section)
- Use `google-cloud-firestore` library with ADC
- Add TTL policy on old conversations (e.g., auto-delete after 90 days) to manage storage costs
- Make the storage backend configurable: `memory.backend: "sqlite" | "firestore"` in YAML

**Done when:** All existing tests pass with `FirestoreMemory`. Config switch toggles between SQLite and Firestore.

---

#### `[x]` 2.3.2 — Add vector embeddings for semantic search _(Embedder via text-embedding-004; notes carry importance+embedding; scored context retrieval (recency+importance+relevance) wired behind memory.retrieval.mode; semantic /search with keyword fallback; backfill via `python -m assistant.backfill`. LIVE on Cloud Run via AVIN_MEMORY_EMBED_NOTES + AVIN_MEMORY_RETRIEVAL_MODE.)_
**Context:** Keyword search (`LIKE '%query%'`) is brittle. Semantic search lets users find notes by meaning (e.g., searching "budget discussion" finds a note about "cost estimates").
**Details:**
- Use Vertex AI Embeddings API to generate embeddings for each note summary
- Store embeddings in Firestore (or a dedicated vector store if scale demands it)
- `search_notes()` now computes the query embedding and finds nearest neighbors
- Fall back to keyword search if embeddings aren't available

**Done when:** `search_notes("budget")` returns notes about cost estimates, financials, etc. even if the word "budget" doesn't appear in the note text.

**Depends on:** 2.3.1

---

### 2.4 — Containerization & Deployment

#### `[x]` 2.4.1 — Write Dockerfile and docker-compose.yml _(files written + compose validated; live `docker compose up` build pending — start Docker daemon)_
**Context:** We need a reproducible, deployable container image.
**Details:**
- Multi-stage Dockerfile: build stage installs deps, runtime stage copies only what's needed
- `docker-compose.yml` for local development (app + optional local Firestore emulator)
- Health check endpoint at `GET /health`
- Non-root user in container

**Done when:** `docker compose up` starts the API server locally. Health check passes.

---

#### `[x]` 2.4.2 — Deploy to Cloud Run _(live at https://avin-333767001298.us-central1.run.app; Firestore backend; proxy-secret + API-key auth; --allow-unauthenticated + 2Gi, max-instances 1. SA granted cloudbuild.builds.builder, storage.objectViewer, datastore.user. WS/torch path not yet load-tested.)_
**Context:** Cloud Run is ideal for our budget — it scales to zero when idle (no cost), supports WebSockets, and is GCP-native.
**Details:**
- Build and push container image to Artifact Registry
- Deploy with: `gcloud run deploy avin --image ... --allow-unauthenticated=false`
- Set environment variables for config overrides
- Set memory limit (512MB should suffice) and max instances (1, for budget control)
- Test WebSocket connectivity through Cloud Run

**Done when:** API is accessible at a Cloud Run URL. WebSocket streaming works. Cost is $0 when idle.

**Depends on:** 2.4.1

---

#### `[ ]` 2.4.3 — Set up monitoring and billing alerts
**Context:** We need visibility into costs and errors to stay within the $20/month budget.
**Details:**
- Set up Cloud Logging (automatic with Cloud Run)
- Create billing alerts at $10 and $20
- Add basic error alerting (e.g., notify on 5xx error spike)
- Add a `/metrics` endpoint or use Cloud Monitoring for request count and latency tracking

**Done when:** Billing alerts are active. Errors appear in Cloud Logging. Can see request metrics.

---

### 2.5 — Web Client (phone frontend)

#### `[x]` 2.5.1 — Cloudflare Pages web client _(added per the phone-web-app goal; browser/live validation pending)_
**Context:** A phone-accessible web app (the user's target) that streams microphone audio to the Cloud Run backend and surfaces notes/actions in real time. Hosted on Cloudflare Pages; the Python backend stays on Cloud Run (Workers can't run the torch/audio pipeline).
**Details:**
- `web/` static app (vanilla HTML/CSS/JS, zero build step): `index.html`, `app.js`, `styles.css`, `README.md`.
- Mic capture via `getUserMedia` + Web Audio API; downsamples to 16 kHz mono int16 PCM and streams binary frames to `WS /api/v1/stream`; sends `control/stop` on Stop.
- Renders `transcript`/`note`/`action`/`error` messages; action cards needing confirmation show Confirm/Dismiss → `PATCH /api/v1/actions/{id}` (never auto-confirms — email guardrail preserved at the UI).
- Settings (backend URL + API key) persisted to `localStorage`; auth via `X-API-Key` header / `?api_key=` for WS.
- Backend: added CORS middleware (`AVIN_CORS_ORIGINS`, default `*`) so the Pages origin can call the REST API.

**Done when:** Web client built and committed; JS syntax-checked; backend CORS in place. _Live phone test (mic → notes) is the user's manual step; needs the backend deployed (2.4.2) + CORS origin set._

---

## Stage 3: Raspberry Pi Desk Companion

### 3.1 — Hardware Setup

#### `[ ]` 3.1.1 — Select and procure hardware
**Context:** We need a Raspberry Pi with a good microphone array for an open office environment.
**Details:**
- Recommended: Raspberry Pi 4 (2GB+ RAM) or Pi 5
- Microphone: ReSpeaker 4-Mic Array (beamforming, improves accuracy in noisy environments) or USB conference microphone
- MicroSD card (32GB+), power supply, case
- Optional: LED ring (NeoPixel) for visual feedback

**Done when:** Hardware is procured and physically assembled.

---

#### `[ ]` 3.1.2 — Set up Raspberry Pi OS and validate audio
**Context:** The Pi needs a headless OS setup with working audio drivers.
**Details:**
- Flash Raspberry Pi OS Lite (64-bit, no desktop) onto MicroSD
- Enable SSH, configure WiFi
- Install audio drivers for the selected microphone (ALSA/PulseAudio)
- Test recording with `arecord` and verify audio quality
- Install Python 3.12+, create venv, install project dependencies

**Done when:** `arecord -d 5 test.wav && aplay test.wav` captures and plays back clear audio on the Pi.

**Depends on:** 3.1.1

---

### 3.2 — Client Application

#### `[ ]` 3.2.1 — Build lightweight streaming client
**Context:** The Pi runs only the audio capture + VAD. All heavy processing happens in the cloud.
**Details:**
- Reuse `audio.py` and `vad.py` from the main project
- New `pi_client.py`:
  - Opens WebSocket connection to Cloud Run backend
  - Streams audio chunks when VAD detects speech
  - Receives and logs notes/actions from the server
  - Reconnects automatically on disconnect (with backoff)
  - Local queue: if disconnected, buffer audio segments and send when reconnected
- Runs as a systemd service (starts on boot, auto-restarts on crash)

**Done when:** Speaking near the Pi produces notes in the cloud database. Network disconnection doesn't crash the client.

**Depends on:** 2.1.2, 3.1.2

---

#### `[ ]` 3.2.2 — Add LED feedback
**Context:** Users need visual feedback to know the assistant's state (especially important in shared spaces).
**Details:**
- Blue: listening / idle
- Yellow: processing (speech detected, streaming to cloud)
- Green pulse: action taken successfully
- Red: error (network down, auth failure)
- Use GPIO + NeoPixel library or simple LEDs with `gpiozero`

**Done when:** LED colors change in sync with assistant state transitions.

**Depends on:** 3.2.1

---

#### `[ ]` 3.2.3 — Implement wake word detection
**Context:** In an open office, always-processing all speech is wasteful and potentially privacy-invasive. A wake word lets the user control when the assistant pays attention.
**Details:**
- Evaluate options:
  - **OpenWakeWord** (free, open source, runs locally on Pi)
  - **Picovoice Porcupine** (free tier available, very accurate, custom wake words)
- Run wake word detection locally on the Pi (no cloud calls)
- When wake word is detected: begin streaming the next speech segment to the cloud
- Config option: `wake_word.enabled: true/false` — when false, process all speech (useful at home)

**Done when:** Saying "Hey Avin" (or configured wake word) activates the assistant. Speech before the wake word is ignored.

**Depends on:** 3.2.1

---

### 3.3 — Reliability & Edge Cases

#### `[ ]` 3.3.1 — Network resilience and auto-restart
**Context:** The Pi will run 24/7. It must handle network outages, process crashes, and power cycles gracefully.
**Details:**
- systemd service file with `Restart=always` and `RestartSec=5`
- Watchdog: if the main process hangs (no audio processed for >5 minutes while speech is detected), force restart
- Network: exponential backoff reconnection (1s, 2s, 4s, 8s... max 60s)
- Audio queue: store up to 10 minutes of speech segments locally during network outage, flush when reconnected
- Logging to local file (rotated, max 50MB) for debugging

**Done when:** Pulling the network cable → client buffers audio → plugging back in → buffered audio is processed. Killing the process → systemd restarts it within 5 seconds.

---

#### `[ ]` 3.3.2 — OTA update mechanism
**Context:** We don't want to SSH into the Pi every time we update the client code.
**Details:**
- Simple approach: Git-based. Pi pulls from a `release` branch on a schedule (cron job, every hour)
- If new code is detected: `git pull && pip install -e . && systemctl restart avin`
- More robust approach (later): proper OTA with versioning and rollback

**Done when:** Pushing a commit to the `release` branch on GitHub causes the Pi to update itself within an hour.

**Depends on:** 3.3.1

---

#### `[ ]` 3.3.3 — End-to-end latency testing
**Context:** The assistant should feel responsive. Target: < 3 seconds from end-of-speech to note appearing.
**Details:**
- Measure each pipeline stage: VAD processing, audio upload, transcription, LLM processing, action routing
- Identify bottlenecks
- Optimize: batch small segments, use streaming transcription if available, tune VAD silence threshold

**Done when:** Measured latency documented. Optimizations applied to meet < 3 second target.

---

## Stage 4: Custom Portable Hardware

### 4.1 — Hardware Research

#### `[ ]` 4.1.1 — Evaluate hardware candidates
**Context:** For a portable device, we need something smaller, lower-power, and more durable than a Raspberry Pi.
**Details:**
- Candidates to evaluate:
  - **ESP32-S3** (with PSRAM): WiFi + BLE, very low power, MicroPython or C, cheap (~$10)
  - **Google Coral Dev Board Mini**: has TPU for edge ML, can run wake word / VAD on-device
  - **nRF5340**: BLE, extremely low power, but limited compute
  - **Raspberry Pi Zero 2W**: smaller Pi, still runs Linux/Python, but limited RAM
- Evaluation criteria matrix: battery life, mic quality, compute power, wireless options, size, cost, dev tooling
- Order 1-2 dev boards for prototyping

**Done when:** Evaluation matrix completed. 1-2 boards selected and ordered for prototyping.

---

#### `[ ]` 4.1.2 — Prototype on dev board
**Context:** Before committing to a custom PCB, validate that the chosen board can handle audio capture + WiFi streaming reliably.
**Details:**
- Port the audio capture + WiFi streaming logic to the dev board
- Test: battery life under continuous listening, audio quality, WiFi reliability
- Benchmark: can it run wake word detection locally?

**Done when:** Dev board captures audio and streams to Cloud Run backend successfully. Battery life and audio quality documented.

**Depends on:** 4.1.1

---

### 4.2 — Firmware & Client

#### `[ ]` 4.2.1 — Port audio client to target hardware
**Context:** Depending on the chosen board, we may need to rewrite the client in C or MicroPython.
**Details:**
- Implement audio capture using the board's ADC/I2S microphone interface
- Implement WiFi/BLE connectivity
- Implement local wake word detection if the board has sufficient compute
- Battery management: sleep when idle, wake on sound/button

**Done when:** Device captures audio, detects wake word, and streams to cloud backend.

**Depends on:** 4.1.2

---

### 4.3 — Enclosure & Finish

#### `[ ]` 4.3.1 — Design and build enclosure
**Context:** The final device needs to be something you can clip to your shirt or set on a desk.
**Details:**
- Design in CAD (Fusion 360 or similar)
- 3D print prototype enclosures, iterate on fit
- Include: mic opening, LED window, USB-C charging port, mute button
- Design for wearability: clip, lanyard loop, or pocket-friendly shape

**Done when:** Device sits in a custom enclosure. Looks and feels like a finished product, not a dev board.

---

#### `[ ]` 4.3.2 — Final integration testing
**Context:** End-to-end testing in real-world environments with the final hardware.
**Details:**
- Test in: quiet office, open office, walking outdoors, coffee shop
- Measure: transcription accuracy, latency, battery life per environment
- Fix issues found during testing
- Document final specs and user guide

**Done when:** Device works reliably across all target environments. Battery lasts a full workday (8+ hours).

---

## Backlog Summary

| Stage | Sub-stages | Tasks | Status |
|-------|-----------|-------|--------|
| Stage 1: Local Prototype | 8 | 38 | 🔴 Not started |
| Stage 2: Cloud & Integrations | 4 | 19 | 🔴 Not started |
| Stage 3: Raspberry Pi | 3 | 9 | 🔴 Not started |
| Stage 4: Custom Hardware | 3 | 6 | 🔴 Not started |
| **Total** | **18** | **72** | |
