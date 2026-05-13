"""Field-level parser for natural-language Critic responses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from src.parsing.think_blocks import strip_think_blocks


AnswerCorrect = Literal["yes", "no", "uncertain", "unknown"]


@dataclass(frozen=True)
class ParsedCritic:
    """Parsed critic response with independent usability flags."""

    critique: str
    answer_correct: AnswerCorrect
    suggested_answer: Optional[str]
    confidence: Optional[float]
    has_confidence: bool
    has_suggested_answer: bool
    has_answer_correct: bool
    usable_for_feedback: bool
    usable_for_routing: bool
    usable_for_consensus: bool
    parse_errors: list[str] = field(default_factory=list)


ANSWER_CORRECT_PATTERNS = [
    re.compile(r"(?i)\banswer\s+correct\s*[:=]\s*(yes|no|uncertain)\b"),
    re.compile(r"(?i)\[\s*answer_correct\s*[:=]\s*(yes|no|uncertain)\s*\]"),
    re.compile(r"(?i)\bactor\s+(?:is|was)\s+(correct|incorrect|wrong)\b"),
]

SUGGESTED_PATTERNS = [
    re.compile(r"(?i)\bsuggested\s+(?:final\s+)?answer\s*[:=]\s*\(?\s*(A|B|C|D|Yes|No|unknown)\s*\)?"),
    re.compile(r"(?i)\bthe\s+suggested\s+answer\s+is\s*\(?\s*(A|B|C|D|Yes|No|unknown)\s*\)?"),
    re.compile(r"(?i)\[\s*suggested_final_answer\s*[:=]\s*(A|B|C|D|Yes|No|unknown)\s*\]"),
]

SUGGESTED_MATH_PATTERNS = [
    re.compile(r"(?im)^\s*suggested\s+(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$"),
    re.compile(r"(?im)^\s*the\s+suggested\s+answer\s+is\s*(.+?)\s*$"),
    re.compile(r"(?im)^\s*\[\s*suggested_final_answer\s*[:=]\s*(.+?)\s*\]\s*$"),
]

CONFIDENCE_PATTERNS = [
    re.compile(r"(?i)\bconfidence\s*[:=]\s*([01](?:\.\d+)?)\b"),
    re.compile(r"(?i)\[\s*confidence\s*[:=]\s*([01](?:\.\d+)?)\s*\]"),
]

JUDGEMENT_HEADER = re.compile(
    r"(?im)^\s*(?:final\s+)?judge?ment\s*:?\s*$|^\s*evaluation\s*:?\s*$"
)


def parse_critic_response(text: str, task_type: str = "multiple_choice") -> ParsedCritic:
    """Parse a Critic response without requiring a rigid schema."""
    raw = strip_think_blocks(text or "")
    judgement_block = find_judgement_block(raw)
    target = judgement_block if judgement_block is not None else get_tail(raw, n=10)

    parse_errors: list[str] = []
    answer_correct = parse_answer_correct(target)
    suggested_answer = parse_suggested_answer(target, task_type)
    confidence = parse_confidence(target)

    if answer_correct is None:
        parse_errors.append("missing_answer_correct")
    if suggested_answer is None:
        parse_errors.append("missing_suggested_answer")
    if confidence is None:
        parse_errors.append("missing_confidence")

    critique = strip_judgement_block(raw).strip()
    if not critique and raw.strip():
        critique = raw.strip()
    if not critique:
        parse_errors.append("missing_critique")

    has_confidence = confidence is not None
    has_suggested_answer = suggested_answer is not None and suggested_answer != "unknown"
    has_answer_correct = answer_correct is not None
    resolved_correct: AnswerCorrect = answer_correct or "unknown"

    return ParsedCritic(
        critique=critique,
        answer_correct=resolved_correct,
        suggested_answer=suggested_answer,
        confidence=confidence,
        has_confidence=has_confidence,
        has_suggested_answer=has_suggested_answer,
        has_answer_correct=has_answer_correct,
        usable_for_feedback=bool(critique.strip()),
        usable_for_routing=bool(critique.strip()) and has_confidence,
        usable_for_consensus=has_suggested_answer and has_confidence,
        parse_errors=parse_errors,
    )


def find_judgement_block(text: str) -> Optional[str]:
    """Return the final Judgement/Judgment block, if present."""
    matches = list(JUDGEMENT_HEADER.finditer(text or ""))
    if not matches:
        return None
    start = matches[-1].start()
    return text[start:]


def strip_judgement_block(text: str) -> str:
    """Remove the final judgement block from displayed natural critique."""
    raw = text or ""
    matches = list(JUDGEMENT_HEADER.finditer(raw))
    if matches:
        return raw[:matches[-1].start()].strip()

    # Old bracket schema sometimes appears at the top.  Remove known field
    # lines but keep any human-readable text.
    lines = []
    for line in raw.splitlines():
        clean = line.strip()
        if re.match(
            r"(?i)^\[\s*(answer_correct|suggested_final_answer|error_type|confidence)\s*:",
            clean,
        ):
            continue
        if re.match(r"(?i)^\[\s*critique\s*\]$", clean):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def parse_confidence(text: str) -> Optional[float]:
    """Parse and clamp a 0.0-1.0 confidence value."""
    found = _last_group(text, CONFIDENCE_PATTERNS)
    if found is None:
        return None
    try:
        return max(0.0, min(1.0, float(found)))
    except ValueError:
        return None


def parse_suggested_answer(text: str, task_type: str = "multiple_choice") -> Optional[str]:
    """Parse a suggested answer token."""
    patterns = SUGGESTED_MATH_PATTERNS if task_type == "math" else SUGGESTED_PATTERNS
    found = _last_group(text, patterns)
    if found is None:
        return None
    if str(found).strip().lower() == "unknown":
        return "unknown"
    return _normalize_answer_token(found, task_type)


def parse_answer_correct(text: str) -> Optional[AnswerCorrect]:
    """Parse the Critic's judgement of whether the Actor answer is correct."""
    last: Optional[tuple[int, str]] = None
    for pattern in ANSWER_CORRECT_PATTERNS:
        for match in pattern.finditer(text or ""):
            if last is None or match.start() >= last[0]:
                last = (match.start(), match.group(1).strip().lower())
    if last is None:
        return None
    value = last[1]
    if value == "correct":
        return "yes"
    if value in {"incorrect", "wrong"}:
        return "no"
    if value in {"yes", "no", "uncertain"}:
        return value  # type: ignore[return-value]
    return None


def get_tail(text: str, n: int = 10) -> str:
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    return "\n".join(lines[-n:])


def _last_group(text: str, patterns: list[re.Pattern[str]]) -> Optional[str]:
    last: Optional[tuple[int, str]] = None
    for pattern in patterns:
        for match in pattern.finditer(text or ""):
            if last is None or match.start() >= last[0]:
                last = (match.start(), match.group(1))
    return last[1] if last is not None else None


def _normalize_answer_token(answer: object, task_type: str) -> Optional[str]:
    if answer is None:
        return None
    token = str(answer).strip().strip("().").strip()
    if not token:
        return None
    upper = token.upper()
    if upper == "UNKNOWN":
        return None
    if task_type == "yes_no":
        if upper in {"YES", "Y"}:
            return "YES"
        if upper in {"NO", "N"}:
            return "NO"
        return None
    if task_type == "multiple_choice":
        return upper if upper in {"A", "B", "C", "D"} else None
    if task_type == "mixed":
        if upper in {"A", "B", "C", "D"}:
            return upper
        if upper in {"YES", "Y"}:
            return "YES"
        if upper in {"NO", "N"}:
            return "NO"
        return None
    return token
