"""Stateful answer resolution and mixed-task evaluation helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from src.algorithms.reward import (
    extract_answer_with_source,
    math_answers_equal,
    normalize_answer,
)


MAINTAIN_PREVIOUS_PATTERNS = [
    re.compile(r"(?i)\bprevious answer\b.*\b(correct|right|still|remain|remains)\b"),
    re.compile(r"(?i)\b(correct|right|still|remain|remains)\b.*\bprevious answer\b"),
    re.compile(r"(?i)\bI\s+(still\s+)?(?:think|believe|maintain)\b.*\b(previous|original)\b"),
    re.compile(r"(?i)\bmy\s+(?:previous|original)\s+answer\s+(?:is|remains)\s+(?:correct|right)\b"),
    re.compile(r"我(?:仍然|依然|还是)?认为(?:之前|原来|上一轮)的?答案(?:是)?(?:正确|对的)"),
    re.compile(r"(?:维持|保持)(?:之前|原来|上一轮)的?答案"),
]


@dataclass(frozen=True)
class ResolvedAnswer:
    """Answer state for one generation round."""

    raw_extracted_answer: Optional[str]
    resolved_answer: Optional[str]
    extract_source: str
    format_valid: bool


def mentions_maintain_previous(response: str) -> bool:
    """Return whether a response explicitly keeps the previous answer."""
    text = response or ""
    return any(pattern.search(text) for pattern in MAINTAIN_PREVIOUS_PATTERNS)


def normalize_task_answer(answer: Optional[str], task_type: str) -> Optional[str]:
    """Normalize an answer token for stateful voting without collapsing yes/no."""
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None

    if task_type == "math":
        return text

    upper = text.strip("().").strip(".").upper()
    if task_type == "yes_no":
        if upper in {"YES", "Y"}:
            return "YES"
        if upper in {"NO", "N"}:
            return "NO"
        return None

    if task_type in {"multiple_choice", "mixed"}:
        if upper in {"A", "B", "C", "D"}:
            return upper
        if task_type == "mixed" and upper in {"YES", "Y"}:
            return "YES"
        if task_type == "mixed" and upper in {"NO", "N"}:
            return "NO"
        return None

    return text


def resolve_answer_for_round(
    response: str,
    task_type: str,
    previous_answer: Optional[str] = None,
    extracted_answer: Optional[str] = None,
    extraction_source: Optional[str] = None,
) -> ResolvedAnswer:
    """Resolve a round answer using extraction first, then explicit carry-forward."""
    if extracted_answer is None and extraction_source is None:
        extraction = extract_answer_with_source(response, task_type)
        extracted_answer = extraction.answer
        extraction_source = extraction.source

    normalized = normalize_task_answer(extracted_answer, task_type)
    if normalized:
        source = extraction_source or "fallback"
        return ResolvedAnswer(
            raw_extracted_answer=extracted_answer,
            resolved_answer=normalized,
            extract_source=source,
            format_valid=source == "strict",
        )

    previous = normalize_task_answer(previous_answer, task_type)
    if previous and mentions_maintain_previous(response):
        return ResolvedAnswer(
            raw_extracted_answer=extracted_answer,
            resolved_answer=previous,
            extract_source="carried_forward",
            format_valid=False,
        )

    return ResolvedAnswer(
        raw_extracted_answer=extracted_answer,
        resolved_answer=None,
        extract_source="none",
        format_valid=False,
    )


def answers_match(pred: Optional[str], label: Optional[str], task_type: str) -> bool:
    """Compare one prediction/label pair using its own task type."""
    pred_text = "" if pred is None else str(pred)
    label_text = "" if label is None else str(label)
    if task_type == "math":
        return math_answers_equal(pred_text, label_text)
    return normalize_answer(pred_text, task_type) == normalize_answer(label_text, task_type)


def compute_accuracy_mixed(
    predictions: Sequence[Optional[str]],
    labels: Sequence[Optional[str]],
    task_types: Sequence[str],
) -> float:
    """Compute accuracy with per-sample task types."""
    n = len(labels)
    if n == 0:
        return 0.0
    correct = sum(
        answers_match(
            predictions[i] if i < len(predictions) else None,
            labels[i],
            task_types[i] if i < len(task_types) else "yes_no",
        )
        for i in range(n)
    )
    return correct / n


def compute_accuracy_with_ci_mixed(
    predictions: Sequence[Optional[str]],
    labels: Sequence[Optional[str]],
    task_types: Sequence[str],
    z: float = 1.96,
) -> tuple[float, float]:
    """Compute mixed-task accuracy and Wilson score margin."""
    n = len(labels)
    if n == 0:
        return 0.0, 0.0
    accuracy = compute_accuracy_mixed(predictions, labels, task_types)
    denom = 1 + z**2 / n
    margin = z * math.sqrt(
        (accuracy * (1 - accuracy) + z**2 / (4 * n)) / n
    ) / denom
    return accuracy, margin


def source_counts(sources: Sequence[str]) -> dict[str, int]:
    """Count answer-source categories with stable keys."""
    keys = ["strict", "fallback", "carried_forward", "none"]
    return {key: sum(1 for source in sources if source == key) for key in keys}


def source_rates(sources: Sequence[str]) -> dict[str, float | int]:
    """Return strict/fallback/carried-forward/unresolved counts and rates."""
    total = len(sources)
    counts = source_counts(sources)
    result: dict[str, float | int] = {"total": total}
    for key, count in counts.items():
        result[f"{key}_count"] = count
        result[f"{key}_rate"] = count / total if total else 0.0
    return result
