"""Prompt construction entry points for natural deliberation."""

from __future__ import annotations

from typing import Any

from src.prompts.actor_prompts import build_initial_actor_prompt, build_revision_actor_prompt
from src.prompts.critic_prompts import build_critic_prompt
from src.society.agent_registry import AgentConfig, ReasoningStyle, CriticSkill


def build_problem_text(sample: dict[str, Any], dataset_name: str = "") -> str:
    """Render the task context without imposing an answer format."""
    question = str(sample.get("question", "")).strip()
    passage = str(sample.get("passage", "")).strip()
    choices = sample.get("choices", []) or []
    task_type = sample.get("task_type", "")

    parts: list[str] = []
    if task_type == "math" or dataset_name in {"math", "gsm8k"}:
        parts.append(f"Problem:\n{question}")
    else:
        parts.append(f"Question:\n{question}")

    if passage:
        parts.append(f"Passage:\n{passage}")

    if choices:
        option_lines = []
        for label, choice in zip("ABCD", choices):
            option_lines.append(f"({label}) {choice}")
        parts.append("Options:\n" + "\n".join(option_lines))

    return "\n\n".join(parts).strip()


def build_actor_prompt(
    actor: AgentConfig,
    sample: dict[str, Any],
    dataset_name: str,
    round_num: int = 0,
    previous_actor_response: str = "",
    critic_feedback: str = "",
) -> str:
    """Build an Actor prompt for the current deliberation round."""
    problem_text = build_problem_text(sample, dataset_name)
    style = actor.reasoning_style if isinstance(actor.reasoning_style, ReasoningStyle) else None
    if round_num <= 0 or not previous_actor_response:
        return build_initial_actor_prompt(style, problem_text, actor_name=actor.name)
    return build_revision_actor_prompt(
        style,
        problem_text,
        previous_actor_response=previous_actor_response,
        critic_feedback=critic_feedback,
        actor_name=actor.name,
    )


def build_simple_actor_prompt(
    sample: dict[str, Any],
    dataset_name: str,
    round_num: int = 0,
    previous_actor_response: str = "",
    critic_feedback: str = "",
    style: ReasoningStyle | None = None,
) -> str:
    """Build an Actor prompt when no AgentConfig is available."""
    problem_text = build_problem_text(sample, dataset_name)
    if round_num <= 0 or not previous_actor_response:
        return build_initial_actor_prompt(style, problem_text)
    return build_revision_actor_prompt(
        style,
        problem_text,
        previous_actor_response=previous_actor_response,
        critic_feedback=critic_feedback,
    )


def build_critic_feedback_prompt(
    critic: AgentConfig,
    sample: dict[str, Any],
    dataset_name: str,
    actor_response: str,
) -> str:
    """Build a Critic prompt for one Actor response."""
    problem_text = build_problem_text(sample, dataset_name)
    skill = critic.error_specialty if isinstance(critic.error_specialty, CriticSkill) else None
    return build_critic_prompt(skill, problem_text, actor_response, critic_name=critic.name)


def build_simple_critic_prompt(
    sample: dict[str, Any],
    dataset_name: str,
    actor_response: str,
    skill: CriticSkill | None = None,
) -> str:
    """Build a Critic prompt when no AgentConfig is available."""
    return build_critic_prompt(
        skill,
        build_problem_text(sample, dataset_name),
        actor_response,
    )


def build_guided_actor_prompt(
    sample: dict[str, Any],
    dataset_name: str,
    target_answer: str,
    round_num: int = 0,
    previous_actor_response: str = "",
    critic_feedback: str = "",
    style: ReasoningStyle | None = None,
) -> str:
    """Build a natural guided Actor prompt for preference-pair generation."""
    problem_text = build_problem_text(sample, dataset_name)
    guide = (
        f"For this training rollout, reason toward the answer {target_answer} "
        "when it is defensible from the problem."
    )
    guided_problem_text = f"{problem_text}\n\n{guide}".strip()
    if round_num <= 0:
        return build_initial_actor_prompt(style, guided_problem_text)
    return build_revision_actor_prompt(
        style,
        guided_problem_text,
        previous_actor_response=previous_actor_response,
        critic_feedback=critic_feedback,
    )


def build_guided_critic_prompt(
    sample: dict[str, Any],
    dataset_name: str,
    actor_response: str,
    target_answer: str,
    skill: CriticSkill | None = None,
) -> str:
    """Build a guided Critic prompt for preference-pair generation."""
    problem_text = build_problem_text(sample, dataset_name)
    guide = (
        f"For this training rollout, evaluate whether the answer should be {target_answer}."
    )
    guided_problem_text = f"{problem_text}\n\n{guide}".strip()
    return build_critic_prompt(skill, guided_problem_text, actor_response)
