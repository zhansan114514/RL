"""
Structured Critic judgement contract.

This module is the single source of truth for Critic output formatting,
parsing, validation, and training-sample rendering.  The Router and Critic
training code must use these helpers instead of maintaining local regexes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


VALID_SUGGESTED_ANSWERS = {"A", "B", "C", "D", "YES", "NO", "UNKNOWN"}
VALID_ERROR_TYPES = {
    "computation",
    "reasoning",
    "knowledge",
    "grounding",
    "verification",
    "none",
}

CRITIC_JUDGEMENT_CONTRACT = """
You MUST start your response with exactly these verdict fields:

[Answer_Correct: yes or no]
[Suggested_Final_Answer: A or B or C or D or Yes or No or unknown]
[Error_Type: computation or reasoning or knowledge or grounding or verification or none]
[Confidence: 0.0-1.0]
[Critique]
Provide at most three concise sentences of evidence.
"""


@dataclass
class CriticJudgement:
    """Parsed Critic judgement with schema-validation diagnostics."""

    answer_correct: Optional[bool]
    suggested_final_answer: Optional[str]
    error_type: Optional[str]
    confidence: Optional[float]
    critique: str
    schema_valid: bool
    schema_errors: list[str] = field(default_factory=list)


ANSWER_CORRECT_LINE = re.compile(
    r"^\[Answer_Correct:\s*(yes|no)\]$",
    re.IGNORECASE,
)
SUGGESTED_ANSWER_LINE = re.compile(
    r"^\[Suggested_Final_Answer:\s*(A|B|C|D|Yes|No|unknown)\]$",
    re.IGNORECASE,
)
ERROR_TYPE_LINE = re.compile(
    r"^\[Error_Type:\s*"
    r"(computation|reasoning|knowledge|grounding|verification|none)\]$",
    re.IGNORECASE,
)
CONFIDENCE_LINE = re.compile(
    r"^\[Confidence:\s*([0-9]*\.?[0-9]+)\]$",
    re.IGNORECASE,
)
CRITIQUE_LINE = re.compile(r"^\[Critique\]$", re.IGNORECASE)


def render_critic_judgement(
    answer_correct: bool,
    suggested_final_answer: str,
    error_type: str,
    confidence: float,
    critique: str,
) -> str:
    """Render a schema-valid Critic judgement for DPO chosen responses."""

    suggested = _normalize_answer_token(suggested_final_answer) or "unknown"
    error = _normalize_error_type(error_type) or "none"
    conf = max(0.0, min(1.0, float(confidence)))
    return (
        f"[Answer_Correct: {'yes' if answer_correct else 'no'}]\n"
        f"[Suggested_Final_Answer: {suggested}]\n"
        f"[Error_Type: {error}]\n"
        f"[Confidence: {conf:.2f}]\n"
        "[Critique]\n"
        f"{critique.strip()}\n"
    )


def parse_critic_judgement(
    text: str,
    actor_answer: str | None = None,
) -> CriticJudgement:
    """Parse and validate a Critic judgement response."""

    raw = text or ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    answer_correct: Optional[bool] = None
    suggested_final_answer: Optional[str] = None
    error_type: Optional[str] = None
    confidence: Optional[float] = None
    critique = ""

    if lines:
        m = ANSWER_CORRECT_LINE.match(lines[0])
        if m:
            answer_correct = m.group(1).lower() == "yes"
    if len(lines) > 1:
        m = SUGGESTED_ANSWER_LINE.match(lines[1])
        if m:
            suggested_final_answer = _normalize_answer_token(m.group(1))
    if len(lines) > 2:
        m = ERROR_TYPE_LINE.match(lines[2])
        if m:
            error_type = _normalize_error_type(m.group(1))
    if len(lines) > 3:
        m = CONFIDENCE_LINE.match(lines[3])
        if m:
            try:
                confidence = max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                confidence = None
    if len(lines) > 4 and CRITIQUE_LINE.match(lines[4]):
        critique = "\n".join(lines[5:]).strip()
    elif len(lines) > 4:
        critique = "\n".join(lines[4:]).strip()

    judgement = CriticJudgement(
        answer_correct=answer_correct,
        suggested_final_answer=suggested_final_answer,
        error_type=error_type,
        confidence=confidence,
        critique=critique,
        schema_valid=False,
        schema_errors=[],
    )
    judgement.schema_valid, judgement.schema_errors = validate_critic_judgement(
        judgement,
        actor_answer=actor_answer,
        raw_text=raw,
    )
    return judgement


def validate_critic_judgement(
    judgement: CriticJudgement,
    actor_answer: str | None = None,
    raw_text: str | None = None,
) -> tuple[bool, list[str]]:
    """Validate field presence, field position, legal values, and consistency."""

    errors: list[str] = []

    if raw_text is not None:
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        expected = [
            ("answer_correct_not_in_verdict_position", ANSWER_CORRECT_LINE),
            ("suggested_answer_not_in_verdict_position", SUGGESTED_ANSWER_LINE),
            ("error_type_not_in_verdict_position", ERROR_TYPE_LINE),
            ("confidence_not_in_verdict_position", CONFIDENCE_LINE),
            ("critique_header_not_in_verdict_position", CRITIQUE_LINE),
        ]
        if len(lines) < len(expected):
            errors.append("missing_verdict_block")
        for idx, (error_name, pattern) in enumerate(expected):
            if idx >= len(lines) or not pattern.match(lines[idx]):
                errors.append(error_name)

    if judgement.answer_correct is None:
        errors.append("missing_answer_correct")
    suggested = _normalize_answer_token(judgement.suggested_final_answer)
    if suggested is None:
        errors.append("missing_suggested_answer")
    elif suggested not in VALID_SUGGESTED_ANSWERS:
        errors.append("invalid_suggested_answer")

    error_type = _normalize_error_type(judgement.error_type)
    if error_type is None:
        errors.append("missing_error_type")
    elif error_type not in VALID_ERROR_TYPES:
        errors.append("invalid_error_type")

    if judgement.confidence is None:
        errors.append("missing_confidence")
    elif not 0.0 <= judgement.confidence <= 1.0:
        errors.append("confidence_out_of_range")

    actor = _normalize_answer_token(actor_answer)
    if actor and suggested and suggested != "UNKNOWN":
        if judgement.answer_correct is True and suggested != actor:
            errors.append("contradiction_correct_with_different_suggestion")
        if judgement.answer_correct is False and suggested == actor:
            errors.append("contradiction_incorrect_with_same_suggestion")

    unique_errors = list(dict.fromkeys(errors))
    return not unique_errors, unique_errors


def strip_critic_judgement_fields(text: str) -> str:
    """Remove verdict lines while keeping the human-readable critique."""

    lines = [line.rstrip() for line in (text or "").splitlines()]
    stripped: list[str] = []
    for idx, line in enumerate(lines):
        clean = line.strip()
        if idx < 5 and (
            ANSWER_CORRECT_LINE.match(clean)
            or SUGGESTED_ANSWER_LINE.match(clean)
            or ERROR_TYPE_LINE.match(clean)
            or CONFIDENCE_LINE.match(clean)
            or CRITIQUE_LINE.match(clean)
        ):
            continue
        stripped.append(line)
    return "\n".join(stripped).strip()


def parse_confidence(text: str) -> Optional[float]:
    return parse_critic_judgement(text).confidence


def parse_answer_correct(text: str) -> Optional[bool]:
    return parse_critic_judgement(text).answer_correct


def parse_suggested_answer(text: str) -> Optional[str]:
    answer = parse_critic_judgement(text).suggested_final_answer
    return None if answer == "UNKNOWN" else answer


def parse_error_type(text: str) -> Optional[str]:
    return parse_critic_judgement(text).error_type


def _normalize_answer_token(answer: object) -> Optional[str]:
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    upper = text.strip("().").upper()
    if upper in VALID_SUGGESTED_ANSWERS:
        return upper
    return None


def _normalize_error_type(error_type: object) -> Optional[str]:
    if error_type is None:
        return None
    text = str(error_type).strip().lower()
    if not text:
        return None
    return text
