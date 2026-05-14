"""Prompt and target builders for first-round Critic SFT."""

from __future__ import annotations

from typing import Any

from src.prompts.control_tokens import ensure_no_think
from src.prompts.critic_prompts import (
    JUDGEMENT_INSTRUCTION,
    critic_role_header,
    render_critic_judgement,
)
from src.prompts.prompt_builder import build_problem_text
from src.society.agent_registry import CriticSkill, resolve_critic_skill


_CORRECTION_FOCUS = {
    CriticSkill.REASONING: (
        "The reasoning chain does not justify the current conclusion. Point out "
        "the invalid inference or missing step, then guide the Actor to re-check "
        "the answer."
    ),
    CriticSkill.KNOWLEDGE: (
        "The response appears to rely on the wrong concept, fact, or domain "
        "knowledge. Name the knowledge issue and use the task context to guide "
        "the Actor."
    ),
    CriticSkill.GROUNDING: (
        "The current answer is not well supported by the question, passage, or "
        "options. Anchor the feedback in the provided context and relevant peer "
        "evidence."
    ),
    CriticSkill.VERIFICATION: (
        "The final answer check fails. Focus on option mapping, consistency, and "
        "whether the final answer follows from the Actor's own response."
    ),
}

_KEEP_FOCUS = {
    CriticSkill.REASONING: (
        "The Actor's reasoning reaches the right conclusion. Explain why the "
        "reasoning should be preserved and why the conflicting peer path should "
        "not change the answer."
    ),
    CriticSkill.KNOWLEDGE: (
        "The Actor is using the relevant concept or fact correctly. Explain why "
        "the conflicting peer claim reflects a weaker or incorrect understanding."
    ),
    CriticSkill.GROUNDING: (
        "The Actor's answer is supported by the task context. Point out why the "
        "conflicting peer answer is not grounded in the question, passage, or "
        "options."
    ),
    CriticSkill.VERIFICATION: (
        "The Actor's final answer is consistent with the response and option "
        "mapping. Explain why the Actor should keep it after checking the peer "
        "disagreement."
    ),
}


def build_critic_sft_prompt(
    *,
    sample: dict[str, Any],
    dataset_name: str,
    critic_skill: str | CriticSkill,
    actor_name: str,
    actor_response: str,
    actor_answer: str,
    other_actor_summary: str,
    no_think: bool = True,
) -> str:
    """Build the Critic SFT input prompt without future or gold-answer fields."""

    skill = (
        critic_skill
        if isinstance(critic_skill, CriticSkill)
        else resolve_critic_skill(str(critic_skill))
    )
    problem_text = build_problem_text(sample, dataset_name)
    body = f"""{critic_role_header(skill)}

Your role is to help the target Actor improve or preserve its answer.
Use the other Actors' summary as peer context, but judge the target Actor independently.

Task:
{problem_text.strip()}

Target Actor:
{actor_name.strip()}

Target Actor current response:
{actor_response.strip()}

Target Actor extracted answer:
{str(actor_answer).strip()}

Other Actors summary:
{other_actor_summary.strip() if other_actor_summary.strip() else "No other Actor responses were available."}

Now provide feedback to the target Actor.

{JUDGEMENT_INSTRUCTION}""".strip()
    return ensure_no_think(body, enabled=no_think)


def render_critic_sft_feedback(
    *,
    critic_skill: str | CriticSkill,
    case_type: str,
    actor_name: str,
    actor_answer: str,
    target_answer: str,
    other_actor_summary: str,
) -> str:
    """Render a natural Critic SFT target with a parseable judgement block."""

    skill = (
        critic_skill
        if isinstance(critic_skill, CriticSkill)
        else resolve_critic_skill(str(critic_skill))
    )
    case = str(case_type).strip().lower()
    actor_label = actor_name.strip() or "the target Actor"
    current = str(actor_answer or "unknown").strip() or "unknown"
    target = str(target_answer or "unknown").strip() or "unknown"
    peer_line = (
        "Use the peer summary as supporting context, but do not copy a peer "
        "answer without checking it."
    )
    if other_actor_summary.strip():
        peer_line = (
            "The peer summary gives useful comparison points; use it to identify "
            "which competing answer is better supported."
        )

    if case == "correction":
        critique = (
            f"{actor_label}'s current answer {current} is not correct. "
            f"{_CORRECTION_FOCUS[skill]} {peer_line} "
            f"The Actor should revise the final answer to {target}."
        )
        return render_critic_judgement(
            answer_correct="no",
            suggested_answer=target,
            confidence=0.86,
            critique=critique,
        )

    if case == "keep":
        critique = (
            f"{actor_label}'s current answer {current} is already correct. "
            f"{_KEEP_FOCUS[skill]} {peer_line} "
            f"The Actor should keep the final answer as {target}."
        )
        return render_critic_judgement(
            answer_correct="yes",
            suggested_answer=target,
            confidence=0.88,
            critique=critique,
        )

    raise ValueError(f"Unknown Critic SFT case_type: {case_type}")
