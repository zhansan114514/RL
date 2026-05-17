"""Tests for natural prompt construction."""

from src.prompts.actor_prompts import build_initial_actor_prompt, build_revision_actor_prompt
from src.prompts.critic_prompts import build_critic_prompt
from src.prompts.control_tokens import ensure_no_think
from src.prompts.prompt_builder import (
    build_guided_actor_prompt,
    build_guided_critic_prompt,
    build_problem_text,
    build_simple_actor_prompt,
    build_simple_critic_prompt,
)
from src.society.agent_registry import CriticSkill, ReasoningStyle


def test_problem_text_renders_boolq_context_without_format_contract():
    sample = {
        "question": "Is the sky blue?",
        "passage": "The sky appears blue during the day.",
        "task_type": "yes_no",
    }

    text = build_problem_text(sample, "boolq")

    assert "Question:\nIs the sky blue?" in text
    assert "Passage:\nThe sky appears blue" in text
    assert "FINAL_ANSWER" not in text


def test_problem_text_renders_multiple_choice_options():
    sample = {
        "question": "What is 2+2?",
        "choices": ["3", "4", "5", "6"],
        "task_type": "multiple_choice",
    }

    text = build_problem_text(sample, "mmlu")

    assert "(A) 3" in text
    assert "(B) 4" in text
    assert "(C) 5" in text
    assert "(D) 6" in text


def test_initial_actor_prompt_uses_direct_contract():
    prompt = build_initial_actor_prompt(
        ReasoningStyle.DIRECT,
        "Question:\nWhat is 2+2?",
    )

    assert "You are Actor-direct" in prompt
    assert "direct style" in prompt
    assert "Direct reason:" in prompt
    assert "no option comparison and no evidence framework" in prompt
    assert "The final result is <answer>." in prompt
    assert prompt.endswith("The final result is <answer>.")
    assert "FINAL_ANSWER" not in prompt
    assert "RATIONALE" not in prompt


def test_initial_actor_prompt_uses_distinct_style_contracts():
    evidence = build_initial_actor_prompt(
        ReasoningStyle.EVIDENCE,
        "Question:\nWhat is 2+2?",
    )
    elimination = build_initial_actor_prompt(
        ReasoningStyle.ELIMINATION,
        "Question:\nWhat is 2+2?",
    )

    assert "Key evidence:" in evidence
    assert "Application:" in evidence
    assert "Option analysis:" not in evidence
    assert "Option analysis:" in elimination
    assert "Elimination:" in elimination
    assert "Key evidence:" not in elimination


def test_revision_actor_prompt_shows_natural_feedback_only():
    prompt = build_revision_actor_prompt(
        ReasoningStyle.ELIMINATION,
        "Question:\nPick the best option.",
        previous_actor_response="I chose A. The final result is A.",
        critic_feedback="The option mapping is wrong; compare B and C.",
    )

    assert "You previously gave this response:" in prompt
    assert "Critics provided the following feedback:" in prompt
    assert "The option mapping is wrong" in prompt
    assert "Option analysis:" in prompt
    assert "Elimination:" in prompt
    assert "weight=" not in prompt
    assert "schema_valid" not in prompt
    assert "The final result is <answer>." not in prompt
    assert "Do not copy placeholder text" in prompt
    assert "The final result is X." in prompt
    assert prompt.endswith(
        "Replace X with the concrete final answer, such as A, B, C, D, Yes, No, or the numeric result."
    )


def test_simple_actor_prompt_applies_style_and_options():
    prompt = build_simple_actor_prompt(
        {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "task_type": "multiple_choice",
        },
        "mmlu",
        style=ReasoningStyle.EVIDENCE,
    )

    assert prompt.startswith("/no_think\n")
    assert "Actor-evidence" in prompt
    assert "Key evidence:" in prompt
    assert "Application:" in prompt
    assert "Options:" in prompt
    assert "The final result is <answer>." in prompt


def test_guided_actor_prompt_keeps_natural_protocol():
    prompt = build_guided_actor_prompt(
        {"question": "Q?", "task_type": "yes_no"},
        "boolq",
        target_answer="Yes",
        style=ReasoningStyle.DIRECT,
    )

    assert prompt.startswith("/no_think\n")
    assert "reason toward the answer Yes" in prompt
    assert "Direct reason:" in prompt
    assert "The final result is <answer>." in prompt
    assert prompt.count("The final result is <answer>.") == 1
    assert prompt.endswith("The final result is <answer>.")
    assert "FINAL_ANSWER" not in prompt


def test_guided_critic_prompt_keeps_judgement_protocol_at_end():
    prompt = build_guided_critic_prompt(
        {"question": "Q?", "task_type": "yes_no"},
        "boolq",
        actor_response="The final result is No.",
        target_answer="Yes",
        skill=CriticSkill.VERIFICATION,
    )

    assert prompt.startswith("/no_think\n")
    assert "evaluate whether the answer should be Yes" in prompt
    assert "Judgement:" in prompt
    assert prompt.endswith("Confidence: 0.0-1.0")


def test_critic_prompt_uses_natural_critique_and_judgement_block():
    prompt = build_critic_prompt(
        CriticSkill.GROUNDING,
        "Question:\nIs the claim supported?",
        "The final result is Yes.",
    )

    assert "You are Critic-grounding" in prompt
    assert "Give natural-language critique first." in prompt
    assert "Judgement:" in prompt
    assert "Answer correct: yes/no/uncertain" in prompt
    assert "Suggested answer: A/B/C/D/Yes/No, a math result, or unknown" in prompt
    assert "[Answer_Correct" not in prompt


def test_simple_critic_prompt_includes_actor_response():
    prompt = build_simple_critic_prompt(
        {"question": "Is it true?", "passage": "The passage says no.", "task_type": "yes_no"},
        "boolq",
        "The final result is Yes.",
        skill=CriticSkill.VERIFICATION,
    )

    assert prompt.startswith("/no_think\n")
    assert "Actor response:" in prompt
    assert "The final result is Yes." in prompt
    assert "Critic-verification" in prompt


def test_control_token_helper_is_idempotent():
    prompt = ensure_no_think("/no_think\n/no_think\nQuestion?")

    assert prompt == "/no_think\nQuestion?"
