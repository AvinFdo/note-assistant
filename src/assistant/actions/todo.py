"""CreateTodoAction: logs a to-do item (mock in Stage 1, Google Tasks in Stage 2).

In Stage 1 the action prints to the console and returns a result message; in Stage 2
this will instead call the Google Tasks API.  SQLite status persistence is handled by
route_action in __init__.py which receives the action_id returned by memory.save_action.
"""

from assistant.actions.base import Action


class CreateTodoAction(Action):
    """Action that creates a to-do item from extracted intent details."""

    intent = "create_todo"

    def execute(self, details: dict) -> str:
        """Print the mock todo to the console and return a confirmation message."""
        task = details.get("task", "")
        print(f"[TODO] {task}")
        return f"[TODO] {task}"

    def describe(self, details: dict) -> str:
        """Return a one-line description suitable for a confirmation prompt."""
        task = details.get("task", "(no task specified)")
        return f"Create to-do: {task}"
