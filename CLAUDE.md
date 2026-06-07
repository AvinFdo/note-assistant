# CLAUDE.md — Avin Note-Taking Assistant

Context-aware, always-listening voice assistant: audio → VAD → transcription → LLM reasoning → actions. A "second brain" that captures conversations and routes actions to Google Workspace + Obsidian.

**Read these before starting work** (deep detail lives here, not in this file):
- [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md) — architecture, schemas, config, prompt design, API contracts
- [docs/BACKLOG.md](docs/BACKLOG.md) — all 72 tasks with Context / Details / Done When. **Start at the "Current Focus" marker at the top.**

## How to work a task

1. Pick the task at **Current Focus** in `docs/BACKLOG.md` (respect any `Depends on:`). Don't jump ahead.
2. Branch: `git checkout -b stage-<id>-<slug>` (e.g. `stage-1.1.2-pyproject`).
3. Implement to satisfy the task's **Done When** — that's the acceptance contract. No more, no less.
4. Run lint + tests (see Commands). They must pass before you open a PR.
5. In the same PR: flip the task `[ ]` → `[x]` in `docs/BACKLOG.md` and advance the Current Focus marker.
6. Open a PR; reference the task ID in the title. CI must be green to merge.

One task = one branch = one PR. Keep PRs small and reviewable.

## Commands

> Scaffolding (pyproject, package, tests) is built in task 1.1.2–1.1.3. Until then these are the *target* commands.

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"     # install with dev deps
pytest                       # run tests (uses mocks — never live APIs)
ruff check .                 # lint
ruff format .                # format
avin --help                  # CLI entry point (built in 1.8.1)
```

## Layout (target — see Brief §7)

```
src/assistant/        config.py, audio.py, transcriber.py, brain.py, memory.py, vad.py, cli.py
src/assistant/actions/ base.py + registry, todo/email/calendar/research
tests/                conftest.py (in-memory SQLite, mock config, mock genai client)
config/default.yaml   all tunables — see Brief §5
docs/                 PROJECT_BRIEF.md, BACKLOG.md
```

## Conventions

- **No hardcoded values.** Everything tunable goes through `config/default.yaml` → `config.py`. Read from `config.*`, never inline constants.
- **Dataclasses for return types** (`Conversation`, `Note`, `Action`, `ProcessingResult`, etc.).
- **Custom exception hierarchies** per module (e.g. `TranscriptionError` → `AuthenticationError`, `NetworkError`, `SilenceError`).
- **Every `.py` has a module docstring** explaining its purpose.
- **Tests use mocks, never live APIs** — mock the `genai.Client`, use `:memory:` SQLite. CI runs with no credentials.
- **`memory.py` abstracts storage** so the Stage 2 SQLite→Firestore swap is an implementation change, not a consumer rewrite.
- Single unified SDK: `google-genai` for both transcription (multimodal audio) and reasoning. No `google-cloud-speech`.

## Guardrails (do not violate)

- **Never commit secrets.** `.env`, OAuth client JSON, and `data/`/`recordings/` are gitignored — keep it that way.
- **Email is the most dangerous action.** `send_email` defaults to `confirm_first` and must NEVER auto-send. Respect the `confidence` threshold (default 0.7) — low-confidence actions are logged, not executed.
- **Budget is <$20/mo.** VAD filtering (skip silence) is the main cost lever — don't process or transmit silence.
- Pin dependency versions in `pyproject.toml`; the unified SDK can break.
