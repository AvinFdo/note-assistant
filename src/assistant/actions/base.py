"""Abstract Action base class and exception hierarchy for the pluggable action framework.

Every concrete action type (todo, email, calendar, research) inherits from Action and
implements execute() and describe(). The registry in __init__.py auto-discovers all
subclasses and maps them by their intent class attribute.

Exception hierarchy
-------------------
ActionError         — base for all action framework failures
UnknownActionError  — raised when no registered Action handles the given intent
"""

from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class ActionError(Exception):
    """Base exception for all action framework errors."""


class UnknownActionError(ActionError):
    """Raised when route_action receives an intent with no registered handler."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class Action(ABC):
    """Abstract base class for all executable actions.

    Subclasses must set the class attribute ``intent`` (e.g. ``"create_todo"``)
    and implement both :meth:`execute` and :meth:`describe`.
    """

    intent: str  # subclasses set this class attribute, e.g. "create_todo"

    @abstractmethod
    def execute(self, details: dict) -> str:
        """Execute the action. Returns a human-readable result message."""

    @abstractmethod
    def describe(self, details: dict) -> str:
        """Return a human-readable description for confirmation prompts."""
