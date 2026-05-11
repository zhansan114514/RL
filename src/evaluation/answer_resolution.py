"""Answer normalization and mixed-task evaluation helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from src.algorithms.reward import math_answers_equal, normalize_answer
from src.parsing.answer_extractor import extract_answer


ANSWER_SOURCES = [
    "final_result",
    "final_answer",
    "boxed",
    "tail_claim",
    "weak_tail",
    "none",
]


@dataclass(frozen=True)
class ResolvedAnswer:
    """Answer state for one generation."""

    raw_extracted_answer: Optional[str]
    resolved_answer: Optional[str]
    extract_source: str
    parse_confidence: float


def normalize_task_answer(answer: Optional[str], task_type: str) -> Optional[str]:
    """Normalize an answer token for voting without collapsing yes/no labels."""
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
    parse_confidence: Optional[float] = None,
) -> ResolvedAnswer:
    """Resolve a round answer from extraction only.

    The previous answer parameter is accepted by legacy callers but no longer
    used.  Actor outputs are not repaired or carried forward implicitly.
    """
    del previous_answer
    if extracted_answer is None and extraction_source is None:
        extraction = extract_answer(response, task_type)
        extracted_answer = extraction.answer
        extraction_source = extraction.source
        parse_confidence = extraction.confidence

    normalized = normalize_task_answer(extracted_answer, task_type)
    return ResolvedAnswer(
        raw_extracted_answer=extracted_answer,
        resolved_answer=normalized,
        extract_source=extraction_source or "none",
        parse_confidence=0.0 if parse_confidence is None else parse_confidence,
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
    """Count parser source categories."""
    return {key: sum(1 for source in sources if source == key) for key in ANSWER_SOURCES}


def source_rates(sources: Sequence[str]) -> dict[str, float | int]:
    """Return parser source counts and rates."""
    total = len(sources)
    counts = source_counts(sources)
    result: dict[str, float | int] = {"total": total}
    for key, count in counts.items():
        result[f"{key}_count"] = count
        result[f"{key}_rate"] = count / total if total else 0.0
    result["parse_success_count"] = total - counts["none"]
    result["parse_success_rate"] = (
        result["parse_success_count"] / total if total else 0.0
    )
    flexible_count = (
        counts["final_answer"]
        + counts["boxed"]
        + counts["tail_claim"]
        + counts["weak_tail"]
    )
    result["flexible_parse_count"] = flexible_count
    result["flexible_parse_rate"] = flexible_count / total if total else 0.0
    return result
