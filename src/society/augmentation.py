"""
Controlled augmentation helpers for minority Actor/Critic training buckets.

Augmentation is explicit and reported by callers.  It never silently fabricates
classification labels: generated Actor responses are checked for answer
correctness, strict output format, and style classification before use.
"""

from __future__ import annotations

import logging
from typing import Any

from src.algorithms.reward import extract_answer, math_answers_equal, normalize_answer
from src.prompts.prompt_builder import build_guided_actor_prompt
from src.society.agent_registry import ACTOR_STYLE_PROMPTS, ReasoningStyle
from src.society.data_classifier import (
    ClassificationError,
    _call_api,
    classify_reasoning_style,
)

logger = logging.getLogger(__name__)


def generate_style_conditioned_responses(
    samples: list[dict[str, Any]],
    target_style: str,
    dataset_name: str,
    max_generations: int,
    api_key: str,
    api_base: str,
    api_model: str,
    cache_dir: str,
    style_confidence_threshold: float = 0.6,
    request_timeout: int | float = 60,
    max_retries: int = 5,
    retry_delay: int | float = 5,
) -> list[dict[str, Any]]:
    """Generate and validate style-conditioned correct Actor responses."""

    if max_generations <= 0:
        return []
    if not api_key:
        logger.warning("Skipping synthetic Actor augmentation: API key is missing")
        return []

    try:
        style = ReasoningStyle(target_style)
    except ValueError:
        raise ValueError(f"Unknown target style for augmentation: {target_style}")

    synthetic: list[dict[str, Any]] = []
    seen_questions: set[str] = set()

    for sample in samples:
        if len(synthetic) >= max_generations:
            break
        question = sample.get("question", "")
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)

        prompt = _build_style_generation_prompt(sample, style, dataset_name)
        try:
            response = _call_api(
                prompt,
                api_url=api_base,
                api_key=api_key,
                model=api_model,
                max_tokens=512,
                request_timeout=request_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        except ClassificationError as e:
            logger.warning("Synthetic generation failed: %s", e)
            continue

        if not _response_answer_is_correct(response, sample):
            continue
        try:
            label = classify_reasoning_style(
                response=response,
                question=question,
                correct_answer=sample.get("answer", ""),
                use_api=True,
                cache_dir=cache_dir,
                api_key=api_key,
                api_base=api_base,
                api_model=api_model,
                request_timeout=request_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        except ClassificationError as e:
            logger.warning("Synthetic style validation failed: %s", e)
            continue

        if (
            label.primary_style != style
            or label.confidence < style_confidence_threshold
        ):
            continue

        synthetic.append({
            "sample": sample,
            "response": response,
            "primary_style": style.value,
            "secondary_styles": [s.value for s in label.secondary_styles],
            "style_confidence": label.confidence,
            "synthetic": True,
        })

    return synthetic


def make_synthetic_rejected_response(sample: dict[str, Any]) -> str:
    """Create a simple wrong-answer rejected response."""

    task_type = sample.get("task_type", "multiple_choice")
    correct = normalize_answer(str(sample.get("answer", "")), task_type)
    if task_type in {"multiple_choice", "mixed"}:
        for option in ("A", "B", "C", "D"):
            if option != correct:
                return f"This response gives an unsupported answer.\n\nThe final result is {option}."
    if task_type == "yes_no":
        wrong = "NO" if correct == "YES" else "YES"
        return f"This response gives an unsupported answer.\n\nThe final result is {wrong.title()}."
    return "This response gives an unsupported numeric result.\n\nThe final result is 0."


def _build_style_generation_prompt(
    sample: dict[str, Any],
    style: ReasoningStyle,
    dataset_name: str,
) -> str:
    prompt = build_guided_actor_prompt(
        sample,
        dataset_name,
        target_answer=sample.get("answer", ""),
        style=style,
    )
    return (
        "/no_think\n"
        f"{ACTOR_STYLE_PROMPTS[style]}\n\n"
        "Generate one correct response for training an Actor with this style.\n"
        "Reason naturally and end with exactly one sentence of the form: "
        "The final result is <answer>.\n"
        "Do not mention that this is synthetic training data.\n\n"
        f"{prompt}\n\n"
        f"Gold answer for this augmentation task: {sample.get('answer', '')}\n"
        "Return only the model response."
    )


def _response_answer_is_correct(response: str, sample: dict[str, Any]) -> bool:
    task_type = sample.get("task_type", "multiple_choice")
    correct = str(sample.get("answer", ""))
    extracted = extract_answer(response, task_type)
    if task_type == "math":
        return math_answers_equal(extracted or "", correct)
    return normalize_answer(extracted or "", task_type) == normalize_answer(correct, task_type)
