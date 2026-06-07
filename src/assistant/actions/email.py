"""SendEmailAction: drafts or sends an email (mock in Stage 1, Gmail in Stage 2) — implemented in task 1.6.2."""
from assistant.actions.base import Action


class SendEmailAction(Action):
    def execute(self, details: dict) -> str:
        pass

    def describe(self, details: dict) -> str:
        pass
