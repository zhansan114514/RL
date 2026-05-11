"""Tests for natural prompt construction."""

from src.prompts.actor_prompts import build_initial_actor_prompt, build_revision_actor_prompt
from src.prompts.critic_prompts import build_critic_prompt
from src.prompts.prompt_builder import (
    build_guided_actor_prompt,
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


def test_initial_actor_prompt_uses_light_final_result_anchor():
    prompt = build_initial_actor_prompt(
        ReasoningStyle.DIRECT,
        "Question:\nWhat is 2+2?",
    )

    assert "You are Actor-direct" in prompt
    assert "shortest sufficient reasoning" in prompt
    assert "The final result is <answer>." in prompt
    assert "FINAL_ANSWER" not in prompt
    assert "RATIONALE" not in prompt


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
    assert "weight=" not in prompt
    assert "schema_valid" not in prompt
    assert prompt.endswith("The final result is <answer>.")


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

    assert "Actor-evidence" in prompt
    assert "Options:" in prompt
    assert "The final result is <answer>." in prompt


def test_guided_actor_prompt_keeps_natural_protocol():
    prompt = build_guided_actor_prompt(
        {"question": "Q?", "task_type": "yes_no"},
        "boolq",
        target_answer="Yes",
        style=ReasoningStyle.DIRECT,
    )

    assert "reason toward the answer Yes" in prompt
    assert "The final result is <answer>." in prompt
    assert "FINAL_ANSWER" not in prompt


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
    assert "Suggested answer: A/B/C/D/Yes/No/unknown" in prompt
    assert "[Answer_Correct" not in prompt


def test_simple_critic_prompt_includes_actor_response():
    prompt = build_simple_critic_prompt(
        {"question": "Is it true?", "passage": "The passage says no.", "task_type": "yes_no"},
        "boolq",
        "The final result is Yes.",
        skill=CriticSkill.VERIFICATION,
    )

    assert "Actor response:" in prompt
    assert "The final result is Yes." in prompt
    assert "Critic-verification" in prompt
