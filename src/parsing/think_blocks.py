"""Helpers for removing model-internal thinking blocks before parsing."""

from __future__ import annotations

import re


def strip_think_blocks(text: str) -> str:
    """Remove complete and unterminated <think> blocks from generated text."""
    raw = text or ""
    raw = re.sub(r"(?is)<think\b[^>]*>.*?</think\s*>", "\n", raw)
    raw = re.sub(r"(?is)<think\b[^>]*>.*$", "\n", raw)
    if re.search(r"(?is)</think\s*>", raw):
        raw = re.split(r"(?is)</think\s*>", raw)[-1]
    return raw.strip()
