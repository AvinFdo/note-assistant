"""SendEmailAction: drafts or sends an email (mock in Stage 1, Gmail in Stage 2).

In Stage 1 the action prints to the console and returns a result message; in Stage 2
this will instead call the Gmail API.  SQLite status persistence is handled by
route_action in __init__.py which receives the action_id returned by memory.save_action.

GUARDRAIL: send_email is always confirm_first in config — it must never auto-execute.
The route_action router enforces this: execute() is only reached after explicit positive
confirmation from the caller's confirm callback.
"""

from assistant.actions.base import Action


class SendEmailAction(Action):
    """Action that sends an email from extracted intent details."""

    intent = "send_email"

    def execute(self, details: dict) -> str:
        """Print the mock email to the console and return a confirmation message."""
        recipient = details.get("recipient", "")
        subject = details.get("subject", "")
        print(f"[EMAIL] To: {recipient}, Subject: {subject}")
        return f"[EMAIL] To: {recipient}, Subject: {subject}"

    def describe(self, details: dict) -> str:
        """Return a one-line description suitable for a confirmation prompt."""
        recipient = details.get("recipient", "(no recipient)")
        subject = details.get("subject", "(no subject)")
        return f"Send email to {recipient} — subject: {subject}"
