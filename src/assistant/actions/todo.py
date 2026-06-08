"""CreateTodoAction: logs a to-do item (mock in Stage 1, Google Tasks in Stage 2) — implemented in task 1.6.2."""

from assistant.actions.base import Action


class CreateTodoAction(Action):
    def execute(self, details: dict) -> str:
        pass

    def describe(self, details: dict) -> str:
        pass
