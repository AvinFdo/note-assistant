"""Brain: assembles memory context, calls Gemini with structured output, and persists results.

This module is the core reasoning layer of the Avin assistant. It takes a raw transcript,
enriches it with context from memory, sends it to Gemini for note and action extraction,
and persists every artifact (conversation, note, actions) back to memory.

Exception hierarchy
-------------------
BrainError   — base for all Brain failures
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from assistant.config import config
from assistant.memory import Memory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class BrainError(Exception):
    """Base exception for all Brain processing failures."""


# ---------------------------------------------------------------------------
# Return-type dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ActionItem:
    """A single extracted action from a transcript.

    Attributes:
        intent:     One of 'create_todo', 'send_email', 'add_calendar', 'research_topic'.
        confidence: Float in [0.0, 1.0] expressing the model's confidence.
        details:    Action-specific payload (task, recipient, subject, title, etc.).
    """

    intent: str
    confidence: float
    details: dict


@dataclass
class ProcessingResult:
    """Structured output returned by :meth:`Brain.process`.

    Attributes:
        is_noteworthy: Whether the transcript contained noteworthy information.
        summary_note:  Concise summary if noteworthy, empty string otherwise.
        actions:       List of extracted :class:`ActionItem` objects (all confidences).
    """

    is_noteworthy: bool
    summary_note: str
    actions: list[ActionItem] = field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON response schema for Gemini structured output
# ---------------------------------------------------------------------------

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_noteworthy": {
            "type": "boolean",
            "description": "Whether this transcript contains information worth saving as a note",
        },
        "summary_note": {
            "type": "string",
            "description": "A concise summary of the key points. Empty string if not noteworthy.",
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": [
                            "create_todo",
                            "send_email",
                            "add_calendar",
                            "research_topic",
                        ],
                    },
                    "confidence": {
                        "type": "number",
                        "description": "0.0 to 1.0 confidence that this action was genuinely intended",
                    },
                    "details": {
                        "type": "object",
                        "description": (
                            "Action-specific fields (task, recipient, subject, body, "
                            "title, time, topic, etc.)"
                        ),
                    },
                },
                "required": ["intent", "confidence", "details"],
            },
        },
    },
    "required": ["is_noteworthy", "summary_note", "actions"],
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_prompt(context: str, transcript: str) -> str:
    """Assemble the full prompt following the template from PROJECT_BRIEF §6."""
    return (
        "SYSTEM:\n"
        "You are Avin, an intelligent note-taking and action-extraction assistant.\n"
        "You are analyzing a transcribed conversation segment from your user's day.\n\n"
        "Your responsibilities:\n"
        "1. Determine if this segment contains noteworthy information"
        " (not all speech is important).\n"
        "2. If noteworthy, write a concise summary note capturing the key points.\n"
        "3. Extract any actionable intents from the following categories:\n"
        "   - create_todo: Tasks the user needs to do\n"
        "   - send_email: Intent to communicate via email"
        " (extract recipient, subject, body)\n"
        "   - add_calendar: Intent to schedule something"
        " (extract title, datetime, attendees)\n"
        "   - research_topic: Topics the user wants to learn about\n\n"
        f"{context}\n\n"
        f'CURRENT TRANSCRIPT:\n"{transcript}"\n\n'
        "Respond using the provided JSON schema."
    )


# ---------------------------------------------------------------------------
# Brain class
# ---------------------------------------------------------------------------


class Brain:
    """Reasoning layer: enrich transcript with context, call Gemini, persist results.

    Args:
        memory: A :class:`~assistant.memory.Memory` instance for context retrieval
                and persistence.
        client: An optional pre-constructed ``genai.Client``.  When *None* the real
                Vertex AI client is built from :data:`assistant.config.config`.
                Pass a mock client in tests to avoid live API calls.
    """

    def __init__(self, memory: Memory, client: genai.Client | None = None) -> None:
        self._memory = memory
        if client is not None:
            self._client = client
        else:
            self._client = genai.Client(
                vertexai=True,
                project=config.gcp.project_id,
                location=config.gcp.region,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, transcript: str) -> ProcessingResult:
        """Process a transcript: enrich with context, call Gemini, persist artifacts.

        Steps
        -----
        1. Assemble context from memory (recent history, pending actions, known info).
        2. Build the full prompt (SYSTEM + context + transcript).
        3. Call Gemini with structured-output JSON mode.
        4. Parse the JSON response into a :class:`ProcessingResult`.
        5. Persist conversation, note (if noteworthy), and all actions to memory.
        6. Apply the confidence guardrail: actions below
           ``config.actions.confidence_threshold`` are saved then marked
           ``'low_confidence'`` and NOT routed for execution. High-confidence
           actions remain ``'pending'`` for the 1.6 action framework.

        Args:
            transcript: Raw text of the conversation segment to process.

        Returns:
            :class:`ProcessingResult` with all fields populated.

        Raises:
            BrainError: If the model returns malformed or invalid JSON.
        """
        # --- Length pre-filter (before any LLM call) ---
        word_count = len(transcript.split())
        min_words = config.memory.min_transcript_words
        if word_count < min_words:
            logger.debug(
                "Transcript filtered out (word_count=%d < min_transcript_words=%d): %r",
                word_count,
                min_words,
                transcript,
            )
            self._memory.save_conversation(transcript)
            return ProcessingResult(is_noteworthy=False, summary_note="", actions=[])

        # 1. Assemble context
        context = self._memory.assemble_context()

        # 2. Build the full prompt
        prompt = _build_prompt(context, transcript)

        # 3. Call Gemini with structured output
        response = self._client.models.generate_content(
            model=config.models.reasoning,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
            ),
        )

        # 4. Parse JSON response
        raw_text = response.text if response.text else ""
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise BrainError(
                f"Brain received malformed JSON from the model: {exc!s}\nRaw response: {raw_text!r}"
            ) from exc

        if not isinstance(data, dict):
            raise BrainError(
                f"Brain expected a JSON object from the model, got: {type(data).__name__}"
            )

        is_noteworthy: bool = bool(data.get("is_noteworthy", False))
        summary_note: str = str(data.get("summary_note", ""))
        raw_actions: list[dict] = data.get("actions", [])

        action_items: list[ActionItem] = []
        for item in raw_actions:
            action_items.append(
                ActionItem(
                    intent=str(item.get("intent", "")),
                    confidence=float(item.get("confidence", 0.0)),
                    details=item.get("details", {}),
                )
            )

        result = ProcessingResult(
            is_noteworthy=is_noteworthy,
            summary_note=summary_note,
            actions=action_items,
        )

        # 5. Persist — conversation is always saved for context continuity
        cid = self._memory.save_conversation(transcript)

        if is_noteworthy:
            self._memory.save_note(cid, summary_note, is_noteworthy=True)

            threshold: float = config.actions.confidence_threshold

            for action in action_items:
                # Resolve execution_mode from config per intent
                execution_mode = self._get_execution_mode(action.intent)
                aid = self._memory.save_action(cid, action.intent, action.details, execution_mode)

                # 6. Confidence guardrail
                if action.confidence < threshold:
                    self._memory.update_action_status(aid, "low_confidence")
                # High-confidence actions remain 'pending' (the default)
        else:
            logger.debug(
                "Transcript not noteworthy — skipping note and action persistence: %r",
                transcript,
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_execution_mode(self, intent: str) -> str:
        """Return the configured execution_mode for the given intent.

        Falls back to 'log_only' for unknown intents to be safe.
        """
        mode_map = {
            "create_todo": config.actions.create_todo.mode,
            "send_email": config.actions.send_email.mode,
            "add_calendar": config.actions.add_calendar.mode,
            "research_topic": config.actions.research_topic.mode,
        }
        return mode_map.get(intent, "log_only")
