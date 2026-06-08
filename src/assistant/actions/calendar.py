"""AddCalendarAction: creates a calendar event (mock in Stage 1, Google Calendar in Stage 2).

Minimal implementation for task 1.6.1 — routing and registry wiring.
Task 1.6.2 adds printing, SQLite status persistence, and confirmation flow.
"""

from assistant.actions.base import Action


class AddCalendarAction(Action):
    """Action that adds a calendar event from extracted intent details."""

    intent = "add_calendar"

    def execute(self, details: dict) -> str:
        """Return a human-readable confirmation that the calendar event was created."""
        title = details.get("title", "")
        time = details.get("time", "")
        return f"[CALENDAR] {title} at {time}"

    def describe(self, details: dict) -> str:
        """Return a one-line description suitable for a confirmation prompt."""
        title = details.get("title", "(no title)")
        time = details.get("time", "(no time)")
        return f"Add calendar event: {title} at {time}"
