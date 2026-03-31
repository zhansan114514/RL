"""
Deliberation engine: natural and guided multi-round Actor-Critic deliberation.

Implements the core iterative discussion loop from the ACC-Collab paper.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from src.prompts.templates import PromptType
from src.prompts.formatter import format_prompt
from src.reward.accuracy import extract_answer

logger = logging.getLogger(__name__)


def deliberate(
    actor_model,
    critic_model,
    sample: dict,
    dataset_name: str,
    num_rounds: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> list[dict]:
    """
    Run T rounds of Actor-Critic deliberation on a task.

    Args:
        actor_model: VLLMInference for the actor.
        critic_model: VLLMInference for the critic.
        sample: Standardized sample dict (question, passage, answer, etc.).
        dataset_name: Dataset name for prompt selection.
        num_rounds: Number of rounds (default 5, t=0..4).
        max_tokens: Max tokens per generation.
        temperature: Sampling temperature.

    Returns:
        List of per-round dicts with keys:
            round, actor_response, critic_response,
            actor_answer, actor_prompt, critic_prompt
    """
    trajectory = []
    previous_responses = []

    for t in range(num_rounds):
        # --- Actor turn ---
        if t == 0:
            actor_prompt = format_prompt(
                dataset_name, PromptType.SINGLE_SHOT, sample,
            )
        else:
            actor_prompt = format_prompt(
                dataset_name, PromptType.DELIBERATION_ACTOR, sample,
                responses=previous_responses,
            )

        actor_response = actor_model.generate_single(
            actor_prompt, max_tokens=max_tokens, temperature=temperature,
        )
        actor_answer = extract_answer(actor_response, sample.get("task_type", "yes_no"))

        # --- Critic turn ---
        critic_prompt = format_prompt(
            dataset_name, PromptType.DELIBERATION_CRITIC, sample,
            actor_response=actor_response,
        )
        critic_response = critic_model.generate_single(
            critic_prompt, max_tokens=max_tokens, temperature=temperature,
        )

        # Record this round
        trajectory.append({
            "round": t,
            "actor_prompt": actor_prompt,
            "actor_response": actor_response,
            "actor_answer": actor_answer,
            "critic_prompt": critic_prompt,
            "critic_response": critic_response,
        })

        # Update response history for next round
        previous_responses.append(actor_response)

    return trajectory


def guided_deliberate_round(
    model,
    sample: dict,
    dataset_name: str,
    target_answer: str,
    previous_responses: list[str],
    agent: str = "actor",
    previous_actor_response: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Generate a guided response for one deliberation round.

    Used to create guided trajectories towards or away from the correct answer.

    Args:
        model: VLLMInference for the target agent.
        sample: Standardized sample.
        dataset_name: Dataset name.
        target_answer: The target answer to guide towards.
        previous_responses: List of prior actor responses.
        agent: "actor" or "critic".
        previous_actor_response: Actor's response (for critic guidance).
        max_tokens: Max tokens.
        temperature: Sampling temperature.

    Returns:
        Guided response text.
    """
    if agent == "actor":
        if not previous_responses:
            prompt_type = PromptType.GUIDED_SINGLE_SHOT
            prompt = format_prompt(
                dataset_name, prompt_type, sample,
                target_answer=target_answer,
            )
        else:
            prompt_type = PromptType.GUIDED_DELIBERATION_ACTOR
            prompt = format_prompt(
                dataset_name, prompt_type, sample,
                target_answer=target_answer,
                responses=previous_responses,
            )
    elif agent == "critic":
        prompt = format_prompt(
            dataset_name, PromptType.GUIDED_DELIBERATION_CRITIC, sample,
            target_answer=target_answer,
            actor_response=previous_actor_response or "",
        )
    else:
        raise ValueError(f"Unknown agent: {agent}. Use 'actor' or 'critic'.")

    return model.generate_single(prompt, max_tokens=max_tokens,
                                 temperature=temperature)
