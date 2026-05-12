"""Quality gates for Actor responses used as training targets."""

from __future__ import annotations

import re

from src.parsing.answer_extractor import ExtractedAnswer, extract_answer


MIN_REASONING_WORDS = 5


def is_trainable_actor_response(
    response: str,
    task_type: str,
    min_reasoning_words: int = MIN_REASONING_WORDS,
) -> bool:
    """Return whether an Actor response is suitable as a DPO chosen target.

    This is intentionally not a strict format check.  The response only needs
    a parseable answer plus enough natural-language content before the final
    answer anchor to avoid training on answer-only completions.
    """
    text = (response or "").strip()
    if not text:
        return False

    extracted = extract_answer(text, task_type)
    if extracted.answer is None:
        return False

    reasoning = _reasoning_text_before_answer(text, extracted)
    return _word_count(reasoning) >= min_reasoning_words


def _reasoning_text_before_answer(text: str, extracted: ExtractedAnswer) -> str:
    raw_span = (extracted.raw_span or "").strip()
    if raw_span:
        idx = text.lower().rfind(raw_span.lower())
        if idx >= 0:
            return text[:idx].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return ""
    return "\n".join(lines[:-1]).strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", text or ""))
