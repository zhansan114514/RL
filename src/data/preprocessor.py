"""Data preprocessing utilities: answer extraction, normalization, format standardization."""

from __future__ import annotations

import re
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_answer(response: str, task_type: str = "yes_no") -> Optional[str]:
    """
    Extract structured answer from LLM text response.

    This implements the zeta function zeta(z_a^(T)) from the paper.

    Args:
        response: Raw text response from the LLM.
        task_type: One of "yes_no", "multiple_choice", "mixed".

    Returns:
        Extracted answer string, or None if extraction fails.
    """
    if not response or not response.strip():
        return None

    if task_type == "yes_no":
        return _extract_yes_no(response)
    elif task_type == "multiple_choice":
        return _extract_multiple_choice(response)
    elif task_type == "mixed":
        # Try multiple choice first, then yes/no
        result = _extract_multiple_choice(response)
        if result:
            return result
        return _extract_yes_no(response)
    else:
        logger.warning(f"Unknown task_type: {task_type}")
        return None


def _extract_yes_no(response: str) -> Optional[str]:
    """Extract Yes/No answer from response."""
    # Clean response
    text = response.strip()

    # Priority patterns (ordered by specificity)
    patterns = [
        r"[Ff]inal [Aa]nswer:?\s*(Yes|No)",
        r"[Aa]nswer:?\s*(Yes|No)",
        r"[Tt]he answer is\s*(Yes|No)",
        r"I (?:think|believe) (?:the answer is )?(?:Yes|No)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    # Fallback: find the last Yes/No in the text
    matches = re.findall(r"\b(Yes|No)\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].lower()

    return None


def _extract_multiple_choice(response: str) -> Optional[str]:
    """Extract A/B/C/D option from response."""
    text = response.strip()

    # Priority patterns
    patterns = [
        r"[Ff]inal [Aa]nswer:?\s*\(([A-D])\)",
        r"[Ff]inal [Aa]nswer:?\s*([A-D])[\.\s]",
        r"[Aa]nswer:?\s*\(([A-D])\)",
        r"[Aa]nswer:?\s*([A-D])[\.\s]",
        r"\(([A-D])\)",
        r"[Oo]ption\s*([A-D])",
        r"[Cc]hoice\s*([A-D])",
        r"[Tt]he (?:correct )?(?:answer|option) is\s*\(?\s*([A-D])\s*\)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Fallback: find standalone A/B/C/D near "answer" context
    match = re.search(r"\b([A-D])\b(?=[\.\,\;\:]?\s*$|\s*(?:is|\.))", text, re.MULTILINE)
    if match:
        return match.group(1).upper()

    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for consistent comparison.

    Args:
        answer: Raw answer string.

    Returns:
        Normalized answer string.
    """
    if answer is None:
        return ""
    return str(answer).strip().lower().strip("()").strip(".").upper()[:1] if answer else ""


def standardize_sample(
    sample: dict[str, Any],
    task_type: str = "yes_no",
) -> dict[str, Any]:
    """
    Convert dataset-specific format to unified format.

    Args:
        sample: Raw sample from a dataset.
        task_type: Type of task.

    Returns:
        Standardized sample with keys: question, passage, answer, choices.
    """
    result = {
        "question": "",
        "passage": "",
        "answer": "",
        "choices": [],
        "task_type": task_type,
    }

    if task_type == "yes_no":
        # BoolQ format
        result["question"] = sample.get("question", "")
        result["passage"] = sample.get("passage", "")
        answer = sample.get("answer", "")
        result["answer"] = "yes" if answer is True or str(answer).lower() == "true" else "no"
        result["choices"] = ["Yes", "No"]

    elif task_type == "multiple_choice":
        # MMLU/SCIQ/ARC format
        result["question"] = sample.get("question", "")
        result["passage"] = sample.get("passage", "")

        choices = sample.get("choices", [])
        if isinstance(choices, list):
            result["choices"] = choices
            # Map choices to A/B/C/D
            labels = [chr(65 + i) for i in range(len(choices))]
            result["choice_labels"] = labels

        # Handle answer as index or string
        answer = sample.get("answer", sample.get("answerKey", ""))
        if isinstance(answer, int):
            result["answer"] = chr(65 + answer) if answer < len(choices) else ""
        elif isinstance(answer, str):
            result["answer"] = answer.upper().strip("()")

    return result


def generate_wrong_answer(
    correct_answer: str,
    choices: list[str] | None = None,
    rng: "random.Random | None" = None,
) -> str:
    """
    Generate a wrong answer for guided trajectory (!y).

    For yes/no: flip the answer.
    For multiple choice: pick a random wrong option.

    Args:
        correct_answer: The correct answer.
        choices: Available choices for multiple choice tasks.
        rng: Optional random.Random instance for reproducibility.

    Returns:
        A wrong answer string.
    """
    import random

    if rng is None:
        rng = random.Random()

    correct = correct_answer.lower().strip()

    if correct in ("yes", "no"):
        return "no" if correct == "yes" else "yes"

    if choices:
        wrong_options = [c for c in choices if c.upper() != correct.upper()]
        if wrong_options:
            return rng.choice(wrong_options).upper()

    # Fallback
    options = ["A", "B", "C", "D"]
    wrong = [o for o in options if o != correct.upper()]
    if wrong:
        return rng.choice(wrong)

    return "A" if correct.upper() != "A" else "B"
