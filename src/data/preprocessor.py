"""Data preprocessing utilities: answer extraction, normalization, format standardization."""

from __future__ import annotations

import re
import logging
from typing import Any, Optional

from src.algorithms.reward import extract_answer, normalize_answer, math_answers_equal, extract_balanced_braces

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
        elif isinstance(choices, list) and choices:
            result["choices"] = choices
            # Map choices to A/B/C/D
            labels = [chr(65 + i) for i in range(len(choices))]
            result["choice_labels"] = labels
        else:
            # No choices provided: SCIQ format (correct_answer + distractor1/2/3)
            # Also handles choices=[] (empty list) from datasets without explicit choices.
            correct = sample.get("correct_answer", "")
            distractors = [
                sample.get(f"distractor{i}", "")
                for i in range(1, 4)
                if sample.get(f"distractor{i}")
            ]
            if correct and distractors:
                import random as _rng
                import hashlib
                # Deterministic seed from sample content so the same SCIQ
                # sample always produces the same option order and answer
                # label across processes and runs.
                seed_bytes = hashlib.md5(
                    (correct + "".join(distractors)).encode()
                ).digest()
                seed_int = int.from_bytes(seed_bytes[:4], "little")
                _r = _rng.Random(seed_int)
                all_options = [correct] + distractors
                _r.shuffle(all_options)
                result["choices"] = all_options
                result["choice_labels"] = [chr(65 + i) for i in range(len(all_options))]
                correct_idx = all_options.index(correct)
                result["answer"] = chr(65 + correct_idx)

        # Handle answer as index or string (skip if already set by SCIQ handler)
        if not result["answer"]:
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
            boxed_match = re.search(r'\\boxed\{', raw_answer)
            if boxed_match:
                brace_start = boxed_match.end() - 1
                content = extract_balanced_braces(raw_answer, brace_start)
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

    elif task_type == "mixed":
        # BBH format: could be yes/no, multiple choice, or free-form
        result["question"] = sample.get("question", sample.get("input", ""))
        result["passage"] = ""

        # Try to extract choices if present
        choices = sample.get("choices", [])
        if isinstance(choices, list) and choices:
            result["choices"] = choices
            result["choice_labels"] = [chr(65 + i) for i in range(len(choices))]
        elif isinstance(choices, dict):
            choice_texts = choices.get("text", [])
            result["choices"] = choice_texts
            result["choice_labels"] = choices.get("label", [chr(65 + i) for i in range(len(choice_texts))])

        # Handle answer
        answer = sample.get("answer", sample.get("target", ""))
        if isinstance(answer, str):
            result["answer"] = answer.strip()
        elif isinstance(answer, bool):
            result["answer"] = "yes" if answer else "no"
            result["choices"] = ["Yes", "No"]
        else:
            result["answer"] = str(answer).strip()

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

            # For zero, only use strategies that guarantee a non-zero result
            if abs(num) < 1e-12:
                offset = rng.choice([1, -1, 2, -2, 3, -3])
                result = float(offset)
            else:
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
        # correct is a label like "A"/"B"/"C"/"D"; choices is text like ["Paris", ...]
        # First try: correct is a label, pick a wrong label
        correct_upper = correct.upper()
        if correct_upper in ("A", "B", "C", "D", "E", "F"):
            correct_idx = ord(correct_upper) - ord("A")
            wrong_labels = [chr(65 + i) for i in range(len(choices)) if i != correct_idx]
            if wrong_labels:
                return rng.choice(wrong_labels)
        # Fallback: correct is option text, find it and pick another
        wrong_options = [c for c in choices if c.upper() != correct_upper]
        if wrong_options:
            return rng.choice(wrong_options)

    # Fallback
    options = ["A", "B", "C", "D"]
    wrong = [o for o in options if o != correct.upper()]
    if wrong:
        return rng.choice(wrong)

    return "A" if correct.upper() != "A" else "B"
