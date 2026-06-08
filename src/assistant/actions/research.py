"""ResearchTopicAction: queues a research task (mock in Stage 1, external search in Stage 2).

Minimal implementation for task 1.6.1 — routing and registry wiring.
Task 1.6.2 adds printing, SQLite status persistence, and confirmation flow.
"""

from assistant.actions.base import Action


class ResearchTopicAction(Action):
    """Action that queues a research topic from extracted intent details."""

    intent = "research_topic"

    def execute(self, details: dict) -> str:
        """Return a human-readable confirmation that the research topic was queued."""
        topic = details.get("topic", "")
        return f"[RESEARCH] {topic}"

    def describe(self, details: dict) -> str:
        """Return a one-line description suitable for a confirmation prompt."""
        topic = details.get("topic", "(no topic)")
        return f"Research topic: {topic}"
