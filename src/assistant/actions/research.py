"""ResearchTopicAction: queues a research task (mock in Stage 1, external search in Stage 2).

In Stage 1 the action prints to the console and returns a result message; in Stage 2
this will instead dispatch to an external search/research API.  SQLite status persistence
is handled by route_action in __init__.py which receives the action_id returned by
memory.save_action.
"""

from assistant.actions.base import Action


class ResearchTopicAction(Action):
    """Action that queues a research topic from extracted intent details."""

    intent = "research_topic"

    def execute(self, details: dict) -> str:
        """Print the mock research task to the console and return a confirmation message."""
        topic = details.get("topic", "")
        print(f"[RESEARCH] {topic}")
        return f"[RESEARCH] {topic}"

    def describe(self, details: dict) -> str:
        """Return a one-line description suitable for a confirmation prompt."""
        topic = details.get("topic", "(no topic)")
        return f"Research topic: {topic}"
