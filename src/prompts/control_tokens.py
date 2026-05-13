"""Control-token helpers for model-specific prompt switches."""

from __future__ import annotations


NO_THINK = "/no_think"


def strip_no_think(prompt: str) -> str:
    """Remove one or more leading /no_think control tokens."""
    text = (prompt or "").strip()
    while text.startswith(NO_THINK):
        text = text[len(NO_THINK):].lstrip()
    return text


def ensure_no_think(prompt: str, enabled: bool = True) -> str:
    """Idempotently add /no_think at the start of a prompt."""
    text = strip_no_think(prompt)
    if not enabled:
        return text
    return f"{NO_THINK}\n{text}".strip()
