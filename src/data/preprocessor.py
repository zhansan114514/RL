"""Data preprocessing utilities: answer extraction, normalization, format standardization."""

from __future__ import annotations

import re
import logging
from typing import Any, Optional

# Re-export from the canonical reward module (authoritative zeta function)
from src.algorithms.reward import extract_answer, normalize_answer

logger = logging.getLogger(__name__)


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
        if isinstance(choices, dict):
            # ARC format: {"text": [...], "label": ["A", "B", ...]}
            choice_texts = choices.get("text", [])
            choice_labels = choices.get("label", [])
            result["choices"] = choice_texts
            result["choice_labels"] = choice_labels if choice_labels else [chr(65 + i) for i in range(len(choice_texts))]
        elif isinstance(choices, list):
            result["choices"] = choices
            # Map choices to A/B/C/D
            labels = [chr(65 + i) for i in range(len(choices))]
            result["choice_labels"] = labels

        # Handle answer as index or string
        answer = sample.get("answer", sample.get("answerKey", ""))
        if isinstance(answer, int):
            n_choices = len(result["choices"]) if result["choices"] else 4
            result["answer"] = chr(65 + answer) if answer < n_choices else ""
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
