"""Natural Critic prompts with lightweight judgement blocks."""

from __future__ import annotations

from src.prompts.control_tokens import ensure_no_think
from src.society.agent_registry import CriticSkill


CRITIC_SKILL_INSTRUCTIONS = {
    CriticSkill.REASONING: (
        "Focus on invalid inference, missing reasoning steps, wrong causal or "
        "logical jumps, and misuse of rules."
    ),
    CriticSkill.KNOWLEDGE: (
        "Focus on factual knowledge, domain concepts, terminology, and whether "
        "the actor relies on correct background information."
    ),
    CriticSkill.GROUNDING: (
        "Focus on whether the actor's answer is supported by the question, "
        "passage, options, or given context."
    ),
    CriticSkill.VERIFICATION: (
        "Focus on final-answer consistency, option mapping, contradiction "
        "checking, and whether the actor's final answer follows from its own reasoning."
    ),
}


JUDGEMENT_INSTRUCTION = """At the end, write a short judgement block:

Judgement:
Answer correct: yes/no/uncertain
Suggested answer: A/B/C/D/Yes/No/unknown
Confidence: 0.0-1.0"""


def critic_role_header(skill: CriticSkill | None, critic_name: str = "") -> str:
    """Return the natural-language Critic role header."""
    skill_name = skill.value if skill else critic_name or "general"
    focus = CRITIC_SKILL_INSTRUCTIONS.get(skill)
    if focus:
        return f"You are Critic-{skill_name}.\n{focus}"
    return (
        f"You are Critic-{skill_name}.\n"
        "Focus on whether the actor's reasoning and final answer are correct."
    )


def build_critic_prompt(
    skill: CriticSkill | None,
    problem_text: str,
    actor_response: str,
    critic_name: str = "",
    no_think: bool = False,
) -> str:
    """Build a natural Critic prompt."""
    body = f"""{critic_role_header(skill, critic_name)}

Your role is to help the actor improve its answer.
Read the question, the options or context, and the actor's response.

Give natural-language critique first.
Focus on whether the actor's reasoning is correct, what it missed, and whether the final answer should be kept or changed.

Do not use a rigid template in the critique itself.

{problem_text.strip()}

Actor response:
{actor_response.strip()}

Now provide your critique.

{JUDGEMENT_INSTRUCTION}""".strip()
    return ensure_no_think(body, enabled=no_think)


def render_critic_judgement(
    answer_correct: str,
    suggested_answer: str,
    confidence: float,
    critique: str,
) -> str:
    """Render a training target in the new natural Critic style."""
    answer_correct_norm = str(answer_correct).strip().lower()
    if answer_correct_norm not in {"yes", "no", "uncertain"}:
        answer_correct_norm = "uncertain"
    suggested = str(suggested_answer or "unknown").strip() or "unknown"
    try:
        conf = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        conf = 0.5
    return (
        f"{critique.strip()}\n\n"
        "Judgement:\n"
        f"Answer correct: {answer_correct_norm}\n"
        f"Suggested answer: {suggested}\n"
        f"Confidence: {conf:.2f}"
    )
