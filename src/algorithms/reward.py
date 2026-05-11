"""Reward and accuracy utilities.

Answer extraction lives in :mod:`src.parsing.answer_extractor`.  This module
keeps task normalization, accuracy, confidence intervals, and reward deltas.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

from src.parsing.answer_extractor import ExtractedAnswer, extract_answer as _extract_answer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnswerExtraction:
    """Compatibility-shaped extraction result backed by the new parser."""

    answer: Optional[str]
    source: str
    confidence: float = 0.0
    raw_span: str = ""


def extract_answer(response: str, task_type: str = "yes_no") -> Optional[str]:
    """Extract a task answer from natural model output."""
    return _extract_answer(response, task_type).answer


def extract_answer_with_source(response: str, task_type: str = "yes_no") -> AnswerExtraction:
    """Extract an answer and return parser-source diagnostics."""
    extracted: ExtractedAnswer = _extract_answer(response, task_type)
    return AnswerExtraction(
        answer=extracted.answer,
        source=extracted.source,
        confidence=extracted.confidence,
        raw_span=extracted.raw_span,
    )


def normalize_answer(answer: str, task_type: str = "yes_no") -> str:
    """Normalize answer for comparison."""
    if not answer:
        return ""
    text = str(answer).strip()
    if task_type == "math":
        return _normalize_math_answer(text)

    upper = text.strip("()").strip(".").upper()
    if task_type == "yes_no":
        if upper in {"YES", "Y"}:
            return "Y"
        if upper in {"NO", "N"}:
            return "N"
        return upper[:1]
    if task_type == "mixed":
        if upper in {"YES", "Y"}:
            return "Y"
        if upper in {"NO", "N"}:
            return "N"
    return upper[:1]


def _normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    text = answer.strip()
    text_match = re.match(r"^\\text\{(.+)\}$", text)
    if text_match:
        text = text_match.group(1).strip()

    frac_match = re.match(r"^\\d?frac\{(.+?)\}\{(.+?)\}$", text)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                result = num / den
                if result == int(result):
                    return str(int(result))
                return str(result)
        except (ValueError, OverflowError):
            pass

    try:
        num = float(text)
        if num == int(num) and "." not in text and "e" not in text.lower():
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        pass

    return re.sub(r"\s+", " ", text)


def math_answers_equal(pred: str, label: str) -> bool:
    """Compare two math answers with a small numeric tolerance."""
    if not pred or not label:
        return pred == label
    norm_pred = _normalize_math_answer(pred.strip())
    norm_label = _normalize_math_answer(label.strip())
    if norm_pred == norm_label:
        return True
    try:
        return abs(float(norm_pred) - float(norm_label)) < 1e-6
    except (ValueError, OverflowError):
        return False


def compute_extraction_success_rates(
    responses: list[str],
    task_type: str | list[str] = "yes_no",
) -> dict[str, float | int | dict[str, int]]:
    """Compute parser diagnostics by extraction source."""
    total = len(responses)
    source_counts = {
        "final_result": 0,
        "final_answer": 0,
        "boxed": 0,
        "tail_claim": 0,
        "weak_tail": 0,
        "none": 0,
    }
    confidence_sum = 0.0
    for idx, response in enumerate(responses):
        current_task_type = task_type[idx] if isinstance(task_type, list) else task_type
        extraction = _extract_answer(response, current_task_type)
        source_counts[extraction.source] += 1
        confidence_sum += extraction.confidence

    extracted = total - source_counts["none"]
    return {
        "extract_success_rate": extracted / total if total else 0.0,
        "extract_success_count": extracted,
        "extract_failure_count": source_counts["none"],
        "avg_parse_confidence": confidence_sum / total if total else 0.0,
        "source_counts": source_counts,
        "source_rates": {
            key: count / total if total else 0.0
            for key, count in source_counts.items()
        },
    }


def compute_accuracy(
    predictions: list[str],
    labels: list[str],
    task_type: str = "yes_no",
) -> float:
    """Compute simple accuracy."""
    if not predictions or not labels:
        return 0.0
    if task_type == "math":
        correct = sum(
            math_answers_equal(prediction, label)
            for prediction, label in zip(predictions, labels)
        )
    else:
        correct = sum(
            normalize_answer(prediction, task_type) == normalize_answer(label, task_type)
            for prediction, label in zip(predictions, labels)
        )
    return correct / len(predictions)


def compute_accuracy_with_ci(
    predictions: list[str],
    labels: list[str],
    confidence: float = 0.95,
    task_type: str = "yes_no",
) -> tuple[float, float]:
    """Compute accuracy and Wilson score margin."""
    acc = compute_accuracy(predictions, labels, task_type=task_type)
    n = len(predictions)
    if n == 0:
        return 0.0, 0.0
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    spread = z * np.sqrt(acc * (1 - acc) / n + z**2 / (4 * n**2)) / denom
    return acc, spread


def compute_per_round_accuracy(
    all_rounds_predictions: list[list[str]],
    labels: list[str],
    task_type: str = "yes_no",
) -> list[float]:
    """Compute accuracy at each deliberation round."""
    return [
        compute_accuracy(round_preds, labels, task_type=task_type)
        for round_preds in all_rounds_predictions
    ]


def compute_improvement_rate(acc_final: float, acc_initial: float) -> float:
    """Relative improvement over initial accuracy."""
    if acc_initial == 0:
        return 0.0
    return (acc_final - acc_initial) / acc_initial


def compute_reward_delta(reward_guided: float, reward_natural: float) -> float:
    """Preference reward delta."""
    return reward_guided - reward_natural
