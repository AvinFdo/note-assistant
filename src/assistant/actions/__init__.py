"""Action registry and router — maps intent strings to Action instances and dispatches execution.

Auto-registration
-----------------
Importing this module triggers imports of all four concrete action modules. Because each
module defines an Action subclass with an ``intent`` class attribute, the registry is
built by calling ``Action.__subclasses__()`` after all imports complete.  Any new
subclass added to this package auto-registers simply by being imported here.

Routing logic
-------------
``route_action`` reads the execution mode for the given intent from config (defaulting
to 'log_only' for unknown intents — safe default).  It then dispatches:

- ``auto_execute``   — call action.execute() immediately, return its message.
                       If *action_id* is provided, status is updated to "executed".
- ``confirm_first``  — call action.describe() and invoke the ``confirm`` callback.
  Status transitions:
    - ``confirm`` is None  → action remains "pending" (awaiting confirmation); not executed.
    - ``confirm`` returns False → status updated to "dismissed"; not executed.
    - ``confirm`` returns True  → execute(); status updated to "executed".
  GUARDRAIL: never auto-execute when confirm is absent — critical path for send_email.
- ``log_only``       — do NOT execute; if *action_id* is provided, status is updated to
                       "logged".

The ``confirm`` callback is injectable so tests can pass a stub and the CLI can pass a
real y/n prompt without this module ever calling input() directly.

SQLite persistence
------------------
When the optional *action_id* parameter is supplied (the UUID returned by
``memory.save_action``), ``route_action`` calls ``memory.update_action_status`` to
persist the final status of the action row.  Passing ``action_id=None`` (the default)
skips all DB writes — existing callers from task 1.6.1 continue to work unchanged.
"""

from __future__ import annotations

from collections.abc import Callable

from assistant.actions.base import Action, ActionError, UnknownActionError
from assistant.actions.calendar import AddCalendarAction
from assistant.actions.email import SendEmailAction
from assistant.actions.research import ResearchTopicAction
from assistant.actions.todo import CreateTodoAction
from assistant.brain import ActionItem
from assistant.config import config
from assistant.memory import Memory

# ---------------------------------------------------------------------------
# Auto-build registry from all Action subclasses
# ---------------------------------------------------------------------------

ACTION_REGISTRY: dict[str, Action] = {
    cls.intent: cls() for cls in Action.__subclasses__() if hasattr(cls, "intent")
}


# ---------------------------------------------------------------------------
# Execution-mode lookup
# ---------------------------------------------------------------------------

_MODE_MAP = {
    "create_todo": lambda: config.actions.create_todo.mode,
    "send_email": lambda: config.actions.send_email.mode,
    "add_calendar": lambda: config.actions.add_calendar.mode,
    "research_topic": lambda: config.actions.research_topic.mode,
}


def _get_mode(intent: str) -> str:
    """Return the configured execution mode for *intent*, defaulting to 'log_only'."""
    getter = _MODE_MAP.get(intent)
    return getter() if getter is not None else "log_only"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def route_action(
    action_item: ActionItem,
    memory: Memory,
    action_id: str | None = None,
    confirm: Callable[[str], bool] | None = None,
) -> str:
    """Look up the action for *action_item.intent* and dispatch based on config mode.

    Args:
        action_item: The :class:`~assistant.brain.ActionItem` to route.
        memory:      The :class:`~assistant.memory.Memory` instance used for status
                     persistence when *action_id* is provided.
        action_id:   Optional UUID string returned by ``memory.save_action``.  When
                     supplied, the corresponding DB row's status is updated after
                     routing.  Pass ``None`` (default) to skip all DB writes — existing
                     callers remain unaffected.
        confirm:     Optional callback ``(description: str) -> bool``.  When the mode
                     is ``confirm_first``, this callback is called with a human-readable
                     description.  If *None*, confirmation is treated as NOT given —
                     the action is never executed (status remains "pending").  Tests
                     inject a stub; the CLI injects a real prompt.

    Returns:
        A human-readable result / status message string.

    Raises:
        UnknownActionError: If no registered Action handles *action_item.intent*.
    """
    intent = action_item.intent
    details = action_item.details

    action = ACTION_REGISTRY.get(intent)
    if action is None:
        raise UnknownActionError(
            f"No registered action handler for intent: {intent!r}. "
            f"Known intents: {sorted(ACTION_REGISTRY)}"
        )

    mode = _get_mode(intent)

    if mode == "auto_execute":
        result = action.execute(details)
        if action_id is not None:
            memory.update_action_status(action_id, "executed")
        return result

    if mode == "confirm_first":
        description = action.describe(details)
        if confirm is None:
            # GUARDRAIL: never auto-execute when confirm is absent —
            # this is the critical path that protects send_email.
            # Status remains "pending" (awaiting confirmation) — no DB update.
            return (
                f"Action '{intent}' awaiting confirmation — not executed. "
                f"Description: {description}"
            )
        if confirm(description):
            result = action.execute(details)
            if action_id is not None:
                memory.update_action_status(action_id, "executed")
            return result
        # confirm returned False — user dismissed the action
        if action_id is not None:
            memory.update_action_status(action_id, "dismissed")
        return f"Action '{intent}' dismissed — not confirmed."

    # log_only (and any unrecognised future mode falls through safely)
    if action_id is not None:
        memory.update_action_status(action_id, "logged")
    return f"Action '{intent}' logged — not executed (mode: {mode})."


__all__ = [
    "ACTION_REGISTRY",
    "ActionError",
    "UnknownActionError",
    "route_action",
    # concrete classes re-exported for convenience
    "CreateTodoAction",
    "SendEmailAction",
    "AddCalendarAction",
    "ResearchTopicAction",
]
