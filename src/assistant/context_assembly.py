"""Shared context-assembly logic for both SQLite Memory and FirestoreMemory.

Provides a pure function that assembles the LLM prompt "memory" section from
pre-fetched data (history strings, action lines, context key-value pairs) and
truncates it to a character budget.  Both Memory and FirestoreMemory call this
helper so the output format and truncation algorithm remain byte-identical
regardless of the backing store.
"""

from __future__ import annotations


def assemble_context_string(
    history: list[str],
    action_lines: list[str],
    context: dict,
    max_tokens: int,
) -> str:
    """Build and return the formatted context string for the LLM prompt.

    Parameters
    ----------
    history:
        Noteworthy note summaries, newest-first.  The list is mutated in-place
        during truncation (oldest entries are popped from the tail) — callers
        should pass a copy if they need the original order preserved.
    action_lines:
        Pre-formatted ``"intent: short_detail"`` strings for recent actions.
    context:
        All context key-value pairs from the context store.
    max_tokens:
        Approximate token budget.  The assembled string is truncated to
        ``max_tokens * 4`` characters by dropping the oldest history entries
        one at a time until it fits.

    Returns
    -------
    str
        Formatted context string with three sections:
        ``CONTEXT (Recent History)``, ``CONTEXT (Known Information)``,
        and ``CONTEXT (Pending Actions)``.
    """
    char_budget = max_tokens * 4

    def _build(hist: list[str]) -> str:
        history_section = "\n".join(f"- {s}" for s in hist) if hist else "- (none)"
        info_section = (
            "\n".join(f"- {k}: {v}" for k, v in context.items()) if context else "- (none)"
        )
        actions_section = (
            "\n".join(f"- {line}" for line in action_lines) if action_lines else "- (none)"
        )
        return (
            f"CONTEXT (Recent History):\n{history_section}\n\n"
            f"CONTEXT (Known Information):\n{info_section}\n\n"
            f"CONTEXT (Pending Actions):\n{actions_section}"
        )

    result = _build(history)

    # Truncate by dropping oldest history entries until within budget.
    # history[0] = newest, history[-1] = oldest → pop from the tail.
    while len(result) > char_budget and history:
        history.pop()  # remove oldest entry
        result = _build(history)

    return result
