"""
Prompt formatting: fill templates with sample data.
"""

from __future__ import annotations

from typing import Any

from src.prompts.templates import (
    ACTOR_PROMPT_TYPES,
    PromptType,
    append_answer_contract,
    get_prompt_template,
)


def format_prompt(
    dataset_name: str,
    prompt_type: PromptType,
    sample: dict[str, Any],
    include_answer_contract: bool | None = None,
    **kwargs: Any,
) -> str:
    """
    Format a prompt template with sample data and optional overrides.

    Args:
        dataset_name: Dataset name (boolq, mmlu, etc.).
        prompt_type: Type of prompt.
        sample: Standardized sample dict with question, passage, answer, etc.
        include_answer_contract: Append the strict FINAL_ANSWER contract.
            Defaults to True for Actor prompt types and False for Critic types.
        **kwargs: Additional variables (target_answer, responses, actor_response).

    Returns:
        Formatted prompt string.
    """
    template = get_prompt_template(dataset_name, prompt_type)

    # Build format variables from sample + kwargs
    fmt_vars: dict[str, Any] = {
        "question": sample.get("question", ""),
        "passage": sample.get("passage", ""),
    }

    # Multiple choice fields
    choices = sample.get("choices", [])
    if choices:
        for i, label in enumerate("ABCD"):
            if i < len(choices):
                key = f"choice_{label.lower()}"
                fmt_vars[key] = choices[i]
    else:
        # Default empty choices
        for label in "abcd":
            fmt_vars[f"choice_{label}"] = ""

    # Common optional template variables.  Defaults are intentional: prompt
    # templates should never leak unresolved placeholders into model input.
    fmt_vars["responses_text"] = _format_responses(kwargs.get("responses", []))
    fmt_vars["actor_response"] = kwargs.get("actor_response", "")
    fmt_vars["target_answer"] = kwargs.get("target_answer", "")

    # Fill remaining kwargs
    for key, value in kwargs.items():
        if key not in fmt_vars:
            fmt_vars[key] = value

    # Safe format: ignore missing keys
    try:
        result = template.format_map(fmt_vars)
    except KeyError:
        # Fallback: manual substitution for missing keys
        result = template
        for key, value in fmt_vars.items():
            result = result.replace("{" + key + "}", str(value))

    should_append_contract = (
        prompt_type in ACTOR_PROMPT_TYPES
        if include_answer_contract is None
        else include_answer_contract
    )
    if should_append_contract:
        result = append_answer_contract(result, sample)

    return result


def _format_responses(responses: list[str] | dict[int, str] | str) -> str:
    """
    Format multi-agent responses for deliberation prompts.

    Args:
        responses: List or dict of agent responses, or a pre-formatted string.

    Returns:
        Formatted response text (Person 1 said: ..., Person 2 said: ...).
    """
    if isinstance(responses, str):
        return responses
    if isinstance(responses, dict):
        items = sorted(responses.items())
    else:
        items = list(enumerate(responses, start=1))

    parts = []
    for idx, resp in items:
        parts.append(f"\nPerson {idx} said: {resp}")

    return "".join(parts)
