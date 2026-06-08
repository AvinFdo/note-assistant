"""AddCalendarAction: creates a calendar event (mock in Stage 1, Google Calendar in Stage 2).

In Stage 1 the action prints to the console and returns a result message; in Stage 2
this will instead call the Google Calendar API.  SQLite status persistence is handled by
route_action in __init__.py which receives the action_id returned by memory.save_action.
"""

from assistant.actions.base import Action


class AddCalendarAction(Action):
    """Action that adds a calendar event from extracted intent details."""

    intent = "add_calendar"

    def execute(self, details: dict) -> str:
        """Print the mock calendar event to the console and return a confirmation message."""
        title = details.get("title", "")
        time = details.get("time", "")
        print(f"[CALENDAR] {title} at {time}")
        return f"[CALENDAR] {title} at {time}"

    def describe(self, details: dict) -> str:
        """Return a one-line description suitable for a confirmation prompt."""
        title = details.get("title", "(no title)")
        time = details.get("time", "(no time)")
        return f"Add calendar event: {title} at {time}"
