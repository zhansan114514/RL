"""Data preprocessing utilities: answer extraction, normalization, format standardization."""

from __future__ import annotations

import re
import logging
from typing import Any, Optional

# Re-export from the canonical reward module (authoritative zeta function)
from src.algorithms.reward import extract_answer, normalize_answer, math_answers_equal

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

    elif task_type == "math":
        # MATH/GSM8K format with \boxed{} or "#### number" answers
        result["question"] = sample.get("question", sample.get("problem", ""))
        result["passage"] = ""

        # Extract answer from various formats
        raw_answer = sample.get("answer", sample.get("solution", ""))
        if isinstance(raw_answer, str):
            # Try to extract from \boxed{...} with balanced brace matching
            from src.algorithms.reward import _extract_balanced_braces
            boxed_match = re.search(r'\\boxed\{', raw_answer)
            if boxed_match:
                brace_start = boxed_match.end() - 1
                content = _extract_balanced_braces(raw_answer, brace_start)
                result["answer"] = content.strip() if content else raw_answer.strip()
            # Try GSM8K "#### number" format
            elif "####" in raw_answer:
                after_hash = raw_answer.split("####")[-1].strip()
                # Extract numeric value
                num_match = re.search(r'-?[0-9]+(?:\.[0-9]+)?', after_hash)
                result["answer"] = num_match.group(0) if num_match else after_hash
            else:
                # Extract last number as fallback
                nums = re.findall(r'-?[0-9]+(?:\.[0-9]+)?', raw_answer)
                result["answer"] = nums[-1] if nums else raw_answer.strip()
        else:
            result["answer"] = str(raw_answer).strip()

        result["choices"] = []

    return result


def generate_wrong_answer(
    correct_answer: str,
    choices: list[str] | None = None,
    task_type: str = "multiple_choice",
    rng: "random.Random | None" = None,
) -> str:
    """
    Generate a wrong answer for guided trajectory (!y).

    For yes/no: flip the answer.
    For multiple choice: pick a random wrong option.
    For math: perturb the numeric answer.

    Args:
        correct_answer: The correct answer.
        choices: Available choices for multiple choice tasks.
        task_type: Task type (yes_no, multiple_choice, math).
        rng: Optional random.Random instance for reproducibility.

    Returns:
        A wrong answer string.
    """
    import random

    if rng is None:
        rng = random.Random()

    correct = correct_answer.strip()

    if task_type == "math":
        # For math, perturb the numeric answer
        try:
            num = float(correct)
            is_int = '.' not in correct and 'e' not in correct.lower() and num == int(num)
            strategies = ["negate", "perturb_up", "perturb_down", "add_offset"]
            strategy = rng.choice(strategies)
            if strategy == "negate":
                result = -num
            elif strategy == "perturb_up":
                result = round(num * rng.uniform(1.1, 1.5), 2)
            elif strategy == "perturb_down":
                result = round(num * rng.uniform(0.5, 0.9), 2)
            else:  # add_offset
                offset = rng.choice([-1, 1, 2, -2, 3, -3])
                result = int(num) + offset if is_int else num + offset
            # Preserve integer format if original was integer
            if is_int:
                return str(int(result)) if result == int(result) else str(result)
            return str(result)
        except ValueError:
            # Non-numeric answer, return a generic wrong value
            return "0"

    if correct.lower() in ("yes", "no"):
        return "no" if correct.lower() == "yes" else "yes"

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
