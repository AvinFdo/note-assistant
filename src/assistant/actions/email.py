"""SendEmailAction: drafts or sends an email (mock in Stage 1, Gmail in Stage 2).

Minimal implementation for task 1.6.1 — routing and registry wiring.
Task 1.6.2 adds printing, SQLite status persistence, and confirmation flow.

GUARDRAIL: send_email is always confirm_first in config — it must never auto-execute.
"""

from assistant.actions.base import Action


class SendEmailAction(Action):
    """Action that sends an email from extracted intent details."""

    intent = "send_email"

    def execute(self, details: dict) -> str:
        """Return a human-readable confirmation that the email was sent."""
        recipient = details.get("recipient", "")
        subject = details.get("subject", "")
        return f"[EMAIL] To: {recipient}, Subject: {subject}"

    def describe(self, details: dict) -> str:
        """Return a one-line description suitable for a confirmation prompt."""
        recipient = details.get("recipient", "(no recipient)")
        subject = details.get("subject", "(no subject)")
        return f"Send email to {recipient} — subject: {subject}"
