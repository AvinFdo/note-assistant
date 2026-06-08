"""CreateTodoAction: logs a to-do item (mock in Stage 1, Google Tasks in Stage 2).

Minimal implementation for task 1.6.1 — routing and registry wiring.
Task 1.6.2 adds printing, SQLite status persistence, and confirmation flow.
"""

from assistant.actions.base import Action


class CreateTodoAction(Action):
    """Action that creates a to-do item from extracted intent details."""

    intent = "create_todo"

    def execute(self, details: dict) -> str:
        """Return a human-readable confirmation that the todo was created."""
        task = details.get("task", "")
        return f"[TODO] {task}"

    def describe(self, details: dict) -> str:
        """Return a one-line description suitable for a confirmation prompt."""
        task = details.get("task", "(no task specified)")
        return f"Create to-do: {task}"
