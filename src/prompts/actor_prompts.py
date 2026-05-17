"""Natural Actor prompts for style-diverse deliberation."""

from __future__ import annotations

from src.prompts.control_tokens import ensure_no_think
from src.society.agent_registry import ReasoningStyle


ACTOR_PROMPT_VERSION = "actor_style_contract_v2"
FINAL_ANSWER_LINE = "The final result is <answer>."


ACTOR_STYLE_INSTRUCTIONS = {
    ReasoningStyle.DIRECT: (
        "Your role is to solve the problem in the direct style.\n"
        "Give the shortest sufficient answer-first justification. A short factual "
        "phrase is allowed inside the single direct reason sentence, but do not "
        "frame facts, definitions, wording clues, or background as evidence.\n"
        "Do not compare options or rule out alternatives."
    ),
    ReasoningStyle.EVIDENCE: (
        "Your role is to solve the problem in the evidence style.\n"
        "Identify the decisive fact, definition, concept, question wording, or "
        "domain knowledge, then apply it to the answer.\n"
        "Do not use option-by-option elimination as the main reasoning surface."
    ),
    ReasoningStyle.ELIMINATION: (
        "Your role is to solve the problem in the elimination style.\n"
        "Compare the available options, rule out weaker or incorrect alternatives, "
        "and explain why the selected option remains best."
    ),
}


ACTOR_STYLE_OUTPUT_CONTRACTS = {
    ReasoningStyle.DIRECT: (
        "Do not add extra headings, bullets, paragraphs, or alternative analyses.\n"
        "Use exactly this visible output format; replace the placeholder text:\n"
        "Direct reason: <one short answer-first sentence, 8-25 words; no option "
        "comparison and no evidence framework>\n"
        f"{FINAL_ANSWER_LINE}"
    ),
    ReasoningStyle.EVIDENCE: (
        "Do not add extra headings, bullets, paragraphs, or option-by-option elimination.\n"
        "Use exactly this visible output format; replace the placeholder text:\n"
        "Key evidence: <the decisive fact, definition, concept, or question clue>\n"
        "Application: <why that evidence supports the answer>\n"
        f"{FINAL_ANSWER_LINE}"
    ),
    ReasoningStyle.ELIMINATION: (
        "Do not add extra headings, bullets, paragraphs, or evidence-only analysis.\n"
        "Use exactly this visible output format; replace the placeholder text:\n"
        "Option analysis: <briefly compare options or rule out alternatives>\n"
        "Elimination: <why the selected option remains best>\n"
        f"{FINAL_ANSWER_LINE}"
    ),
}

ACTOR_STYLE_REVISION_OUTPUT_CONTRACTS = {
    ReasoningStyle.DIRECT: (
        "Do not add extra headings, bullets, paragraphs, or alternative analyses.\n"
        "Do not copy placeholder text or angle-bracket examples from the prompt.\n"
        "Use exactly this visible output format with your actual content:\n"
        "Direct reason: one short answer-first sentence.\n"
        "The final result is X.\n"
        "Replace X with the concrete final answer, such as A, B, C, D, Yes, No, "
        "or the numeric result."
    ),
    ReasoningStyle.EVIDENCE: (
        "Do not add extra headings, bullets, paragraphs, or option-by-option elimination.\n"
        "Do not copy placeholder text or angle-bracket examples from the prompt.\n"
        "Use exactly this visible output format with your actual content:\n"
        "Key evidence: the decisive fact, definition, concept, or question clue.\n"
        "Application: why that evidence supports the answer.\n"
        "The final result is X.\n"
        "Replace X with the concrete final answer, such as A, B, C, D, Yes, No, "
        "or the numeric result."
    ),
    ReasoningStyle.ELIMINATION: (
        "Do not add extra headings, bullets, paragraphs, or evidence-only analysis.\n"
        "Do not copy placeholder text or angle-bracket examples from the prompt.\n"
        "Use exactly this visible output format with your actual content:\n"
        "Option analysis: briefly compare options or rule out alternatives.\n"
        "Elimination: why the selected option remains best.\n"
        "The final result is X.\n"
        "Replace X with the concrete final answer, such as A, B, C, D, Yes, No, "
        "or the numeric result."
    ),
}


DEFAULT_FINAL_RESULT_INSTRUCTION = (
    "Give a concise natural-language answer.\n"
    "At the end, write one final answer sentence:\n"
    f"{FINAL_ANSWER_LINE}"
)


def actor_output_contract(style: ReasoningStyle | None, revision: bool = False) -> str:
    """Return the visible output contract for an Actor prompt."""
    parts: list[str] = []
    if revision:
        parts.append(
            "Now revise your answer if needed. You may keep your previous answer "
            "or change it, but make a fresh final decision."
        )
        parts.append(
            ACTOR_STYLE_REVISION_OUTPUT_CONTRACTS.get(
                style,
                DEFAULT_FINAL_RESULT_INSTRUCTION
                + "\nDo not copy placeholder text; write the concrete final answer.",
            )
        )
    else:
        parts.append(ACTOR_STYLE_OUTPUT_CONTRACTS.get(style, DEFAULT_FINAL_RESULT_INSTRUCTION))
    return "\n\n".join(parts)


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
    no_think: bool = False,
) -> str:
    """Build an initial natural Actor prompt."""
    body = (
        f"{actor_role_header(style, actor_name)}\n\n"
        f"{problem_text.strip()}\n\n"
        f"{actor_output_contract(style)}"
    ).strip()
    return ensure_no_think(body, enabled=no_think)


def build_revision_actor_prompt(
    style: ReasoningStyle | None,
    problem_text: str,
    previous_actor_response: str,
    critic_feedback: str,
    actor_name: str = "",
    no_think: bool = False,
) -> str:
    """Build a revision prompt that exposes only natural Critic feedback."""
    body = (
        f"{actor_role_header(style, actor_name)}\n\n"
        f"{problem_text.strip()}\n\n"
        "You previously gave this response:\n"
        f"{previous_actor_response.strip()}\n\n"
        "Critics provided the following feedback:\n"
        f"{critic_feedback.strip() if critic_feedback.strip() else 'No critic feedback was selected.'}\n\n"
        f"{actor_output_contract(style, revision=True)}"
    ).strip()
    return ensure_no_think(body, enabled=no_think)
