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


def _extract_math(text: str) -> Optional[str]:
    """Extract mathematical answer from response (supports \boxed{} and numeric answers)."""
    # First try to extract from \boxed{...}
    boxed_patterns = [
        r'\\boxed\{([^}]+)\}',
        r'boxed\{([^}]+)\}',
        r'\{\\boxed\{([^}]+)\}\}',
    ]
    for pat in boxed_patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()

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


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (strip, upper, first char)."""
    if not answer:
        return ""
    return str(answer).strip().strip("()").strip(".").upper()[:1]


# ============================================================
# Accuracy computation
# ============================================================

def compute_accuracy(
    predictions: list[str],
    labels: list[str],
) -> float:
    """
    Compute simple accuracy.

    Args:
        predictions: List of predicted answers.
        labels: List of ground truth answers.

    Returns:
        Accuracy as float in [0, 1].
    """
    if not predictions or not labels:
        return 0.0

    correct = sum(
        normalize_answer(p) == normalize_answer(l)
        for p, l in zip(predictions, labels)
    )
    return correct / len(predictions)


def compute_accuracy_with_ci(
    predictions: list[str],
    labels: list[str],
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute accuracy with Wilson score confidence interval.

    Args:
        predictions: Predicted answers.
        labels: Ground truth answers.
        confidence: Confidence level (default 0.95).

    Returns:
        Tuple of (accuracy, margin_of_error).
    """
    acc = compute_accuracy(predictions, labels)
    n = len(predictions)

    if n == 0:
        return 0.0, 0.0

    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    center = (acc + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(acc * (1 - acc) / n + z**2 / (4 * n**2)) / denom

    margin = min(center - (center - spread), (center + spread) - center)
    return acc, margin


def compute_per_round_accuracy(
    all_rounds_predictions: list[list[str]],
    labels: list[str],
) -> list[float]:
    """
    Compute accuracy at each deliberation round.

    Args:
        all_rounds_predictions: predictions[round][sample_idx].
        labels: Ground truth answers.

    Returns:
        List of accuracy values, one per round.
    """
    return [
        compute_accuracy(round_preds, labels)
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
