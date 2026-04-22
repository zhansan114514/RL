"""
Reward computation: answer extraction, accuracy, and reward delta.

This is the canonical (authoritative) implementation of:
- zeta(response): extract structured answer from LLM text
- compute_accuracy(): batch accuracy with Wilson confidence intervals
- compute_reward_delta(): reward difference for preference pairs

All modules should import from here, not from src/reward/ or src/data/preprocessor.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ============================================================
# Answer extraction (zeta function)
# ============================================================

def extract_answer(response: str, task_type: str = "yes_no") -> Optional[str]:
    """
    Extract structured answer from LLM text response (zeta function).

    Args:
        response: Raw text response from the LLM.
        task_type: "yes_no", "multiple_choice", "math", or "mixed".

    Returns:
        Extracted answer string (YES/NO for boolq, A/B/C/D for MC, numeric/math expression for math), or None.
    """
    if not response or not response.strip():
        return None

    if task_type == "yes_no":
        return _extract_yes_no(response)
    elif task_type == "multiple_choice":
        return _extract_mc(response)
    elif task_type == "math":
        return _extract_math(response)
    elif task_type == "mixed":
        result = _extract_mc(response)
        if result:
            return result
        return _extract_yes_no(response)
    else:
        logger.warning(f"Unknown task_type: {task_type}")
        return None


def _extract_yes_no(text: str) -> Optional[str]:
    """Extract Yes/No from response (returns uppercase)."""
    patterns = [
        r"[Ff]inal [Aa]nswer:?\s*(Yes|No)",
        r"[Aa]nswer:?\s*(Yes|No)",
        r"[Tt]he answer is\s*(Yes|No)",
        r"I (?:think|believe) (?:the answer is )?(Yes|No)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().upper()

    # Fallback: last Yes/No occurrence
    matches = re.findall(r"\b(Yes|No)\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].strip().upper()

    return None


def _extract_mc(text: str) -> Optional[str]:
    """Extract A/B/C/D from multiple choice response."""
    patterns = [
        r"[Ff]inal [Aa]nswer:?\s*\(?([A-D])\)?",
        r"[Aa]nswer:?\s*\(?([A-D])\)?",
        r"\(([A-D])\)",
        r"[Oo]ption\s*([A-D])",
        r"[Cc]hoice\s*([A-D])",
        r"[Tt]he (?:correct )?(?:answer|option) is\s*\(?\s*([A-D])\s*\)?",
        r"\b([A-D])\b(?=[\.\,\;\:]?\s*$|\s*(?:is|\.))",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().upper()
    return None


def extract_balanced_braces(text: str, start: int) -> Optional[str]:
    """Extract content inside balanced curly braces starting at position start.

    Handles nested braces like \\boxed{\\frac{1}{2}}.
    """
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
        i += 1
    return None


def _extract_math(text: str) -> Optional[str]:
    """Extract mathematical answer from response (supports \\boxed{} and numeric answers)."""
    # First try to extract from \boxed{...} with balanced brace matching
    boxed_prefixes = [
        r'\\boxed\{',
        r'boxed\{',
        r'\{\\boxed\{',
    ]
    for prefix in boxed_prefixes:
        for m in re.finditer(prefix, text):
            brace_start = m.end() - 1  # position of the opening '{'
            content = extract_balanced_braces(text, brace_start)
            if content is not None:
                return content.strip()

    # Try to extract from "Final Answer:" or "Answer:" patterns
    answer_patterns = [
        r'[Ff]inal [Aa]nswer:?\s*([0-9]+(?:\.[0-9]+)?)',
        r'[Aa]nswer:?\s*([0-9]+(?:\.[0-9]+)?)',
        r'[Tt]he answer is\s+([0-9]+(?:\.[0-9]+)?)',
        r'=\s*([0-9]+(?:\.[0-9]+)?)(?:\s|$|\.|,)',
    ]
    for pat in answer_patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()

    # Fallback: try to extract any mathematical expression near the end
    # Look for patterns like "therefore X" or "equals X"
    fallback_patterns = [
        r'(?:therefore|so|thus|equals?|is)\s+([0-9]+(?:\.[0-9]+)?|\([^)]+\))',
        r'=\s*([0-9]+(?:\.[0-9]+)?|\([^)]+\))',
    ]
    for pat in fallback_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    return None


def normalize_answer(answer: str, task_type: str = "yes_no") -> str:
    """Normalize answer for comparison.

    For yes_no / multiple_choice: strip, upper, first char (Y/N/A/B/C/D).
    For math: strip whitespace, normalize numeric representation.
    """
    if not answer:
        return ""
    s = str(answer).strip()
    if task_type == "math":
        return _normalize_math_answer(s)
    return s.strip("()").strip(".").upper()[:1]


def _normalize_math_answer(answer: str) -> str:
    """Normalize a math answer for comparison.

    Strips whitespace and LaTeX formatting, normalizes numeric values.
    """
    s = answer.strip()
    # Strip surrounding \text{} if present
    text_match = re.match(r'^\\text\{(.+)\}$', s)
    if text_match:
        s = text_match.group(1).strip()
    # Try numeric comparison: parse as float and normalize
    try:
        num = float(s)
        # If it's an integer value, return as int string to avoid "42.0" vs "42"
        if num == int(num) and '.' not in s and 'e' not in s.lower():
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        pass
    # Non-numeric: normalize LaTeX whitespace
    s = re.sub(r'\s+', ' ', s)
    return s


def math_answers_equal(pred: str, label: str) -> bool:
    """Compare two math answers with numeric tolerance.

    Handles cases like "42" vs "42.0", "\\frac{1}{2}" vs "0.5", etc.
    """
    if not pred or not label:
        return pred == label
    norm_pred = _normalize_math_answer(pred.strip())
    norm_label = _normalize_math_answer(label.strip())
    # Direct string match after normalization
    if norm_pred == norm_label:
        return True
    # Try numeric comparison with tolerance
    try:
        pred_num = float(norm_pred)
        label_num = float(norm_label)
        return abs(pred_num - label_num) < 1e-6
    except (ValueError, OverflowError):
        pass
    return norm_pred == norm_label


# ============================================================
# Accuracy computation
# ============================================================

def compute_accuracy(
    predictions: list[str],
    labels: list[str],
    task_type: str = "yes_no",
) -> float:
    """
    Compute simple accuracy.

    Args:
        predictions: List of predicted answers.
        labels: List of ground truth answers.
        task_type: Task type for answer normalization ("yes_no", "multiple_choice", "math").

    Returns:
        Accuracy as float in [0, 1].
    """
    if not predictions or not labels:
        return 0.0

    if task_type == "math":
        correct = sum(
            math_answers_equal(p, l)
            for p, l in zip(predictions, labels)
        )
    else:
        correct = sum(
            normalize_answer(p, task_type) == normalize_answer(l, task_type)
            for p, l in zip(predictions, labels)
        )
    return correct / len(predictions)


def compute_accuracy_with_ci(
    predictions: list[str],
    labels: list[str],
    confidence: float = 0.95,
    task_type: str = "yes_no",
) -> tuple[float, float]:
    """
    Compute accuracy with Wilson score confidence interval.

    Args:
        predictions: Predicted answers.
        labels: Ground truth answers.
        confidence: Confidence level (default 0.95).
        task_type: Task type for answer normalization.

    Returns:
        Tuple of (accuracy, margin_of_error).
    """
    acc = compute_accuracy(predictions, labels, task_type=task_type)
    n = len(predictions)

    if n == 0:
        return 0.0, 0.0

    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    center = (acc + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(acc * (1 - acc) / n + z**2 / (4 * n**2)) / denom

    margin = spread
    return acc, margin


def compute_per_round_accuracy(
    all_rounds_predictions: list[list[str]],
    labels: list[str],
    task_type: str = "yes_no",
) -> list[float]:
    """
    Compute accuracy at each deliberation round.

    Args:
        all_rounds_predictions: predictions[round][sample_idx].
        labels: Ground truth answers.
        task_type: Task type for answer normalization.

    Returns:
        List of accuracy values, one per round.
    """
    return [
        compute_accuracy(round_preds, labels, task_type=task_type)
        for round_preds in all_rounds_predictions
    ]


def compute_improvement_rate(acc_final: float, acc_initial: float) -> float:
    """
    Compute percent improvement: (acc_final - acc_initial) / acc_initial.

    As defined in Eq. 7 of the paper.
    """
    if acc_initial == 0:
        return 0.0
    return (acc_final - acc_initial) / acc_initial


# ============================================================
# Reward delta
# ============================================================

def compute_reward_delta(reward_guided: float, reward_natural: float) -> float:
    """
    Compute delta = reward_guided - reward_natural.

    For delta_y: guided towards correct answer vs natural.
    For delta_not_y: natural vs guided away from correct answer.
    """
    return reward_guided - reward_natural
