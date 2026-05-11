"""Natural Actor prompts for style-diverse deliberation."""

from __future__ import annotations

from src.society.agent_registry import ReasoningStyle


ACTOR_STYLE_INSTRUCTIONS = {
    ReasoningStyle.DIRECT: (
        "Your role is to solve the problem with the shortest sufficient reasoning.\n"
        "Avoid unnecessary discussion. Focus on the most direct route to the answer."
    ),
    ReasoningStyle.EVIDENCE: (
        "Your role is to solve the problem by identifying key facts, definitions, "
        "wording, or evidence from the question.\n"
        "Make your answer grounded in the provided information."
    ),
    ReasoningStyle.ELIMINATION: (
        "Your role is to solve the problem by comparing the options and eliminating "
        "weaker or incorrect choices.\n"
        "Explain why the selected option is better than the alternatives."
    ),
}


FINAL_RESULT_INSTRUCTION = (
    "Give your reasoning naturally using the assigned style.\n"
    "At the end, write one final answer sentence:\n"
    "The final result is <answer>."
)


REVISION_FINAL_RESULT_INSTRUCTION = (
    "Now revise your answer if needed.\n"
    "You may keep your previous answer or change it, but make a fresh final decision.\n\n"
    "Use your assigned reasoning style naturally.\n"
    "At the end, write one final answer sentence:\n"
    "The final result is <answer>."
)


def actor_role_header(style: ReasoningStyle | None, actor_name: str = "") -> str:
    """Return the style header for an Actor."""
    style_name = style.value if style else actor_name or "general"
    instruction = ACTOR_STYLE_INSTRUCTIONS.get(style)
    if instruction:
        return f"You are Actor-{style_name}.\n\n{instruction}"
    return (
        f"You are Actor-{style_name}.\n\n"
        "Your role is to solve the problem using clear, natural reasoning."
    )


def build_initial_actor_prompt(
    style: ReasoningStyle | None,
    problem_text: str,
    actor_name: str = "",
) -> str:
    """Build an initial natural Actor prompt."""
    return (
        f"{actor_role_header(style, actor_name)}\n\n"
        f"{problem_text.strip()}\n\n"
        f"{FINAL_RESULT_INSTRUCTION}"
    ).strip()


def build_revision_actor_prompt(
    style: ReasoningStyle | None,
    problem_text: str,
    previous_actor_response: str,
    critic_feedback: str,
    actor_name: str = "",
) -> str:
    """Build a revision prompt that exposes only natural Critic feedback."""
    return (
        f"{actor_role_header(style, actor_name)}\n\n"
        f"{problem_text.strip()}\n\n"
        "You previously gave this response:\n"
        f"{previous_actor_response.strip()}\n\n"
        "Critics provided the following feedback:\n"
        f"{critic_feedback.strip() if critic_feedback.strip() else 'No critic feedback was selected.'}\n\n"
        f"{REVISION_FINAL_RESULT_INSTRUCTION}"
    ).strip()
