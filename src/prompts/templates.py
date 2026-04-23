"""
Prompt templates for ACC-Collab.

Based on Section C of the ACC-Collab paper (ICLR 2025).
6 template types for Actor/Critic, each with dataset-specific variants.

Template types:
1. single_shot: Actor initial response (no deliberation)
2. guided_single_shot: Actor guided toward target answer
3. deliberation_actor: Actor response during deliberation rounds
4. guided_deliberation_actor: Actor guided deliberation
5. deliberation_critic: Critic feedback during deliberation
6. guided_deliberation_critic: Critic guided feedback
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class PromptType(str, Enum):
    """Types of prompts used in ACC-Collab."""

    SINGLE_SHOT = "single_shot"
    GUIDED_SINGLE_SHOT = "guided_single_shot"
    DELIBERATION_ACTOR = "deliberation_actor"
    GUIDED_DELIBERATION_ACTOR = "guided_deliberation_actor"
    DELIBERATION_CRITIC = "deliberation_critic"
    GUIDED_DELIBERATION_CRITIC = "guided_deliberation_critic"


# =============================================================================
# BoolQ Templates (Yes/No reading comprehension)
# =============================================================================

BOOLQ_TEMPLATES = {
    PromptType.SINGLE_SHOT: (
        "You will be given a yes-no question which is based on a passage. "
        "You should use the passage to help you answer the question. "
        "You should give a brief justification for your answer, "
        "and you must provide a final answer of either Yes or No."
        "\nQuestion: {question}?"
        "\nPassage: {passage}"
    ),
    PromptType.GUIDED_SINGLE_SHOT: (
        "You will be given a yes-no question which is based on a passage. "
        "You should use the passage to help you answer the question "
        "with a {target_answer}. "
        "You should give a brief justification for your answer of {target_answer}, "
        "and you must state that your final answer is {target_answer}."
        "\nQuestion: {question}?"
        "\nPassage: {passage}"
    ),
    PromptType.DELIBERATION_ACTOR: (
        "Several people have provided answers to a yes-no question. "
        "Below are their responses:"
        "{responses_text}"
        "\n\nYou should take these answers into consideration when answering "
        "the following yes-no question, which is based on a passage. "
        "You should give a brief justification for your answer, and you must "
        "provide a final answer of either Yes or No."
        "\nQuestion: {question}?"
        "\nPassage: {passage}"
    ),
    PromptType.GUIDED_DELIBERATION_ACTOR: (
        "Several people have provided answers to a yes-no question. "
        "Below are their responses:"
        "{responses_text}"
        "\n\nYou should take these answers and the passage into consideration when "
        "answering the following question with {target_answer}. "
        "You should give a brief justification for your answer of {target_answer}, "
        "and you must state that your final answer is {target_answer}."
        "\nQuestion: {question}?"
        "\nPassage: {passage}"
    ),
    PromptType.DELIBERATION_CRITIC: (
        "I am answering a question based on a passage. "
        "I would like you to help me improve the correctness of my response "
        "by briefly providing some additional details my original response "
        "may have missed. "
        "{responses_text}"
        "\nQuestion: {question}?"
        "\nPassage: {passage}"
        "\nMy Response: {actor_response}"
    ),
    PromptType.GUIDED_DELIBERATION_CRITIC: (
        "I would like you to be a deliberation assistant. "
        "You will be given a question based on a passage and my response "
        "to the question. "
        "You should use the passage and my response to develop additional details "
        "for why the correct answer is {target_answer}. "
        "Your details must be brief and must support the fact that the "
        "correct answer is {target_answer}."
        "{responses_text}"
        "\nQuestion: {question}?"
        "\nPassage: {passage}"
        "\nMy Response: {actor_response}"
    ),
}


# =============================================================================
# MMLU Templates (Multiple choice across subjects)
# =============================================================================

MMLU_TEMPLATES = {
    PromptType.SINGLE_SHOT: (
        "You will be given a multiple choice question. "
        "You should select the correct answer from the given options. "
        "You should give a brief justification for your answer, "
        "and you must provide a final answer with the letter of the correct option."
        "\nQuestion: {question}"
        "\nOptions:"
        "\n(A) {choice_a}"
        "\n(B) {choice_b}"
        "\n(C) {choice_c}"
        "\n(D) {choice_d}"
    ),
    PromptType.GUIDED_SINGLE_SHOT: (
        "You will be given a multiple choice question. "
        "You should use the information to help you select the answer "
        "that leads to option ({target_answer}). "
        "You should give a brief justification for your answer of ({target_answer}), "
        "and you must state that your final answer is ({target_answer})."
        "\nQuestion: {question}"
        "\nOptions:"
        "\n(A) {choice_a}"
        "\n(B) {choice_b}"
        "\n(C) {choice_c}"
        "\n(D) {choice_d}"
    ),
    PromptType.DELIBERATION_ACTOR: (
        "Several people have provided answers to a multiple choice question. "
        "Below are their responses:"
        "{responses_text}"
        "\n\nYou should take these answers into consideration when answering "
        "the following question. "
        "You should give a brief justification for your answer, and you must "
        "provide a final answer with the letter of the correct option."
        "\nQuestion: {question}"
        "\nOptions:"
        "\n(A) {choice_a}"
        "\n(B) {choice_b}"
        "\n(C) {choice_c}"
        "\n(D) {choice_d}"
    ),
    PromptType.GUIDED_DELIBERATION_ACTOR: (
        "Several people have provided answers to a multiple choice question. "
        "Below are their responses:"
        "{responses_text}"
        "\n\nYou should take these answers into consideration when "
        "answering the following question with ({target_answer}). "
        "You should give a brief justification for your answer of ({target_answer}), "
        "and you must state that your final answer is ({target_answer})."
        "\nQuestion: {question}"
        "\nOptions:"
        "\n(A) {choice_a}"
        "\n(B) {choice_b}"
        "\n(C) {choice_c}"
        "\n(D) {choice_d}"
    ),
    PromptType.DELIBERATION_CRITIC: (
        "I am answering a multiple choice question. "
        "I would like you to help me improve the correctness of my response "
        "by briefly providing some additional details my original response "
        "may have missed."
        "{responses_text}"
        "\nQuestion: {question}"
        "\nOptions:"
        "\n(A) {choice_a}"
        "\n(B) {choice_b}"
        "\n(C) {choice_c}"
        "\n(D) {choice_d}"
        "\nMy Response: {actor_response}"
    ),
    PromptType.GUIDED_DELIBERATION_CRITIC: (
        "I would like you to be a deliberation assistant. "
        "You will be given a multiple choice question and my response "
        "to the question. "
        "You should use the question and my response to develop additional details "
        "for why the correct answer is ({target_answer}). "
        "Your details must be brief and must support the fact that the "
        "correct answer is ({target_answer})."
        "{responses_text}"
        "\nQuestion: {question}"
        "\nOptions:"
        "\n(A) {choice_a}"
        "\n(B) {choice_b}"
        "\n(C) {choice_c}"
        "\n(D) {choice_d}"
        "\nMy Response: {actor_response}"
    ),
}

# BBH, SCIQ, ARC share the same structure as MMLU (multiple choice / mixed).
# BBH contains yes/no and free-form subtasks, but they are handled at runtime
# via task_type="mixed" in extract_answer(), not via template specialization.
BBH_TEMPLATES = MMLU_TEMPLATES
SCIQ_TEMPLATES = MMLU_TEMPLATES
ARC_TEMPLATES = MMLU_TEMPLATES


# =============================================================================
# MATH Templates (Mathematical problem solving with \boxed{} format)
# =============================================================================

MATH_TEMPLATES = {
    PromptType.SINGLE_SHOT: (
        "You will be given a mathematics problem. "
        "Solve the problem step by step, showing your work clearly. "
        "You must provide your final answer within \\boxed{{...}} format."
        "\nProblem: {question}"
    ),
    PromptType.GUIDED_SINGLE_SHOT: (
        "You will be given a mathematics problem. "
        "Solve the problem step by step, showing your work clearly. "
        "The correct answer is {target_answer}. "
        "You must provide your final answer within \\boxed{{{target_answer}}} format."
        "\nProblem: {question}"
    ),
    PromptType.DELIBERATION_ACTOR: (
        "Several people have provided solutions to a mathematics problem. "
        "Below are their responses:"
        "{responses_text}"
        "\n\nYou should take these solutions into consideration when solving "
        "the following problem. "
        "Solve the problem step by step, showing your work clearly. "
        "You must provide your final answer within \\boxed{{...}} format."
        "\nProblem: {question}"
    ),
    PromptType.GUIDED_DELIBERATION_ACTOR: (
        "Several people have provided solutions to a mathematics problem. "
        "Below are their responses:"
        "{responses_text}"
        "\n\nYou should take these solutions into consideration when solving "
        "the following problem with answer {target_answer}. "
        "Solve the problem step by step, showing your work clearly. "
        "You must provide your final answer within \\boxed{{{target_answer}}} format."
        "\nProblem: {question}"
    ),
    PromptType.DELIBERATION_CRITIC: (
        "I am solving a mathematics problem. "
        "I would like you to help me improve my solution "
        "by briefly providing any steps or reasoning I may have missed. "
        "\nProblem: {question}"
        "\nMy Solution: {actor_response}"
    ),
    PromptType.GUIDED_DELIBERATION_CRITIC: (
        "I would like you to be a deliberation assistant. "
        "You will be given a mathematics problem and my solution. "
        "You should use the problem and my solution to provide additional details "
        "for why the correct answer is {target_answer}. "
        "Your details must be brief and must support the fact that the "
        "correct answer is {target_answer}."
        "\nProblem: {question}"
        "\nMy Solution: {actor_response}"
    ),
}


# =============================================================================
# GSM8K Templates (Grade school math problems with step-by-step reasoning)
# =============================================================================

GSM_TEMPLATES = {
    PromptType.SINGLE_SHOT: (
        "You will be given a grade school mathematics problem. "
        "Solve the problem step by step, showing your work clearly. "
        "You must provide your final answer as a numeric value."
        "\nProblem: {question}"
    ),
    PromptType.GUIDED_SINGLE_SHOT: (
        "You will be given a grade school mathematics problem. "
        "Solve the problem step by step, showing your work clearly. "
        "The correct answer is {target_answer}. "
        "You must provide your final answer as {target_answer}."
        "\nProblem: {question}"
    ),
    PromptType.DELIBERATION_ACTOR: (
        "Several people have provided solutions to a mathematics problem. "
        "Below are their responses:"
        "{responses_text}"
        "\n\nYou should take these solutions into consideration when solving "
        "the following problem. "
        "Solve the problem step by step, showing your work clearly. "
        "You must provide your final answer as a numeric value."
        "\nProblem: {question}"
    ),
    PromptType.GUIDED_DELIBERATION_ACTOR: (
        "Several people have provided solutions to a mathematics problem. "
        "Below are their responses:"
        "{responses_text}"
        "\n\nYou should take these solutions into consideration when solving "
        "the following problem with answer {target_answer}. "
        "Solve the problem step by step, showing your work clearly. "
        "You must provide your final answer as {target_answer}."
        "\nProblem: {question}"
    ),
    PromptType.DELIBERATION_CRITIC: (
        "I am solving a grade school mathematics problem. "
        "I would like you to help me improve my solution "
        "by briefly providing any steps or reasoning I may have missed. "
        "\nProblem: {question}"
        "\nMy Solution: {actor_response}"
    ),
    PromptType.GUIDED_DELIBERATION_CRITIC: (
        "I would like you to be a deliberation assistant. "
        "You will be given a mathematics problem and my solution. "
        "You should use the problem and my solution to provide additional details "
        "for why the correct answer is {target_answer}. "
        "Your details must be brief and must support the fact that the "
        "correct answer is {target_answer}."
        "\nProblem: {question}"
        "\nMy Solution: {actor_response}"
    ),
}


# Registry: dataset name -> template dict
DATASET_TEMPLATES = {
    "boolq": BOOLQ_TEMPLATES,
    "mmlu": MMLU_TEMPLATES,
    "bbh": BBH_TEMPLATES,
    "sciq": SCIQ_TEMPLATES,
    "arc": ARC_TEMPLATES,
    "math": MATH_TEMPLATES,
    "gsm8k": GSM_TEMPLATES,
}


def get_prompt_template(
    dataset_name: str,
    prompt_type: PromptType,
) -> str:
    """
    Get a prompt template for a specific dataset and type.

    Args:
        dataset_name: One of boolq, mmlu, bbh, sciq, arc, math, gsm8k.
        prompt_type: Type of prompt to retrieve.

    Returns:
        Template string with {variable} placeholders.

    Raises:
        ValueError: If dataset or prompt type not found.
    """
    if dataset_name not in DATASET_TEMPLATES:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: {list(DATASET_TEMPLATES.keys())}"
        )
    templates = DATASET_TEMPLATES[dataset_name]
    if prompt_type not in templates:
        raise ValueError(
            f"Unknown prompt type: {prompt_type}. "
            f"Supported: {list(templates.keys())}"
        )
    return templates[prompt_type]


def get_available_datasets() -> list[str]:
    """Return list of supported dataset names."""
    return list(DATASET_TEMPLATES.keys())


def get_available_prompt_types(dataset_name: str) -> list[PromptType]:
    """Return available prompt types for a dataset."""
    if dataset_name not in DATASET_TEMPLATES:
        return []
    return list(DATASET_TEMPLATES[dataset_name].keys())
