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
    """Extract Yes/No from response (returns uppercase).

    Priority:
      1. FINAL_ANSWER: marker (strict)
      2. Final Answer / Answer near end (medium)
      3. Last standalone Yes/No in tail (weak)
    """
    if not text:
        return None

    # Layer 1: strict FINAL_ANSWER marker (uppercase only)
    strict_patterns = [
        r"(?m)^\s*FINAL[_\s]ANSWER\s*:\s*(Yes|No)\s*$",
        r"(?m)^\s*FINAL_ANSWER\s*:\s*(Yes|No)\s*$",
    ]
    for pat in strict_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip().upper()

    # Tail region for weaker patterns
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    tail = "\n".join(lines[-5:])

    # Layer 2: Final Answer / Answer near end
    medium_patterns = [
        r"(?i)(?:final\s+answer|answer)\s*:\s*(Yes|No)",
        r"(?i)the\s+(?:correct\s+)?answer\s+is\s*(Yes|No)",
        r"(?i)I\s+(?:think|believe)\s+(?:the answer is\s*)?(Yes|No)",
    ]
    for pat in medium_patterns:
        matches = re.findall(pat, tail, re.IGNORECASE)
        if matches:
            return matches[-1].strip().upper()

    # Layer 3: weak fallback — last Yes/No in the tail only
    matches = re.findall(r"\b(Yes|No)\b", tail, re.IGNORECASE)
    if matches:
        return matches[-1].strip().upper()

    return None


def _extract_mc(text: str) -> Optional[str]:
    """Extract A/B/C/D from multiple choice response.

    Three-layer priority (conservative — avoids grabbing option letters
    from reasoning / critic feedback text):
      1. FINAL_ANSWER: X  (strict, full text scan)
      2. Final Answer / Answer / the correct answer is X  (last 5 lines)
      3. Very weak: standalone letter at line end (last 5 lines only)
    """
    if not text:
        return None

    text = text.replace("\uff1a", ":")  # fullwidth colon
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    tail = "\n".join(lines[-5:])

    # Layer 1: strict FINAL_ANSWER marker (scan full text)
    # Only matches the exact uppercase marker we inject into prompts.
    strict_patterns = [
        r"(?m)^\s*FINAL[_\s]ANSWER\s*:\s*\(?\s*([A-D])\s*\)?\s*$",
        r"(?m)^\s*FINAL_ANSWER\s*:\s*\(?\s*([A-D])\s*\)?\s*$",
    ]
    for pat in strict_patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Layer 2: medium-confidence patterns in tail only
    medium_patterns = [
        r"(?i)(?:final\s+answer|answer)\s*:\s*\(?\s*([A-D])\s*\)?",
        r"(?i)the\s+(?:correct\s+)?(?:answer|option|choice)\s+is\s*\(?\s*([A-D])\s*\)?",
        r"(?i)(?:option|choice)\s*([A-D])",
        r"\(([A-D])\)",
    ]
    for pat in medium_patterns:
        matches = re.findall(pat, tail, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Layer 3: weak fallback — restricted to tail only (last 5 lines).
    # The old \b([A-D])\b was dangerous because it scanned the full text,
    # but restricting to the tail avoids picking up option letters from
    # earlier reasoning / critic feedback.
    weak_patterns = [
        r"(?im)^\s*\(?\s*([A-D])\s*\)?\s*$",
        r"(?im)^\s*(?:therefore|thus|so)\s*,?\s*\(?\s*([A-D])\s*\)?\s*\.?\s*$",
    ]
    for pat in weak_patterns:
        matches = re.findall(pat, tail, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Last resort: find the last standalone A-D letter in the tail only.
    matches = re.findall(r"\b([A-D])\b", tail)
    if matches:
        return matches[-1].upper()

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
    """Extract mathematical answer from response (supports \\boxed{}, FINAL_ANSWER, and numeric answers).

    Priority:
      1. FINAL_ANSWER: marker (strict)
      2. \\boxed{...}
      3. Final Answer / Answer patterns
      4. Weak fallback near end
    """
    if not text:
        return None

    # Layer 1: strict FINAL_ANSWER marker (uppercase only, exact format)
    final_answer_match = re.search(
        r"(?m)^\s*FINAL[_\s]ANSWER\s*:\s*(.+?)\s*$", text,
    )
    if not final_answer_match:
        final_answer_match = re.search(
            r"(?m)^\s*FINAL_ANSWER\s*:\s*(.+?)\s*$", text,
        )
    if final_answer_match:
        return final_answer_match.group(1).strip()

    # Layer 2: \boxed{...} with balanced brace matching
    boxed_prefixes = [
        r'\\boxed\s*\{',
        r'boxed\s*\{',
        r'\{\\boxed\s*\{',
    ]
    for prefix in boxed_prefixes:
        for m in re.finditer(prefix, text):
            brace_start = m.end() - 1  # position of the opening '{'
            content = extract_balanced_braces(text, brace_start)
            if content is not None:
                return content.strip()

    # Layer 3: Answer patterns (tail only)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    tail = "\n".join(lines[-5:])

    answer_patterns = [
        r'[Ff]inal [Aa]nswer:?\s*(-?[0-9]+(?:\.[0-9]+)?)',
        r'[Aa]nswer:?\s*(-?[0-9]+(?:\.[0-9]+)?)',
        r'[Tt]he answer is\s+(-?[0-9]+(?:\.[0-9]+)?)',
        r'=\s*(-?[0-9]+(?:\.[0-9]+)?)(?:\s|$|\.|,)',
    ]
    for pat in answer_patterns:
        m = re.search(pat, tail)
        if m:
            return m.group(1).strip()

    # Layer 4: weak fallback
    fallback_patterns = [
        r'(?:therefore|so|thus|equals?|is)\s+(-?[0-9]+(?:\.[0-9]+)?|\([^)]+\))',
        r'=\s*(-?[0-9]+(?:\.[0-9]+)?|\([^)]+\))',
    ]
    for pat in fallback_patterns:
        matches = re.findall(pat, tail, re.IGNORECASE)
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
    Handles \\frac{a}{b} and \\dfrac{a}{b} by evaluating to float.
    """
    s = answer.strip()
    # Strip surrounding \text{} if present
    text_match = re.match(r'^\\text\{(.+)\}$', s)
    if text_match:
        s = text_match.group(1).strip()

    # Evaluate \frac{a}{b} and \dfrac{a}{b} to a numeric string
    # Handles nested fractions like \frac{1}{2} -> "0.5"
    frac_match = re.match(r'^\\d?frac\{(.+?)\}\{(.+?)\}$', s)
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
