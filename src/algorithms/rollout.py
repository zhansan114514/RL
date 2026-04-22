"""
One-step roll-out Monte Carlo for reward estimation.

Following the ACC-Collab paper (Section 4.2), this estimates the partial reward
r(z^(t), x, y) by simulating ONE additional deliberation round from the current
response, repeated multiple times for Monte Carlo estimation.

Optimized: uses batch inference for MC simulations instead of sequential calls.
Only generates actor responses (the final answer comes from the actor, not the critic).
"""

from __future__ import annotations

import logging

from src.prompts.templates import PromptType
from src.prompts.formatter import format_prompt
from src.algorithms.reward import extract_answer, normalize_answer

logger = logging.getLogger(__name__)


def estimate_final_accuracy(
    actor_model,
    critic_model,
    sample: dict,
    dataset_name: str,
    current_actor_response: str,
    current_critic_response: str,
    previous_responses: list[str],
    num_simulations: int = 5,
    remaining_rounds: int = 1,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> float:
    """
    Estimate reward r(z^(t), x, y) via one-step Monte Carlo roll-out.

    Only generates actor responses since the final answer comes from the actor.
    The critic_model parameter is kept for API compatibility but not used.

    Args:
        actor_model: Actor VLLMInference.
        critic_model: Critic VLLMInference (unused in one-step roll-out).
        sample: Task sample.
        dataset_name: Dataset name.
        current_actor_response: Actor's response at round t.
        current_critic_response: Critic's response at round t.
        previous_responses: Responses from prior rounds (actor + critic interleaved).
        num_simulations: Number of MC simulations (default: 5).
        remaining_rounds: Rounds to simulate (should be 1 for one-step).
        max_tokens: Max tokens per generation.
        temperature: Sampling temperature.

    Returns:
        Estimated accuracy (float in [0, 1]).
    """
    task_type = sample.get("task_type", "yes_no")
    correct_answer = normalize_answer(sample.get("answer", ""), task_type=task_type)
    if not correct_answer:
        return 0.0

    if num_simulations <= 0 or remaining_rounds <= 0:
        return 0.0

    # Build all simulation prompts upfront for batch inference
    actor_prompts = []
    for _ in range(num_simulations):
        sim_responses = list(previous_responses) + [current_actor_response, current_critic_response]
        actor_prompt = format_prompt(
            dataset_name, PromptType.DELIBERATION_ACTOR, sample,
            responses=sim_responses,
        )
        actor_prompts.append(actor_prompt)

    # Batch actor inference: all simulations at once
    sim_actor_resps = actor_model.generate(
        actor_prompts, max_tokens=max_tokens, temperature=temperature,
    )

    # Count correct answers from actor responses
    correct_count = 0
    for sim_actor_resp in sim_actor_resps:
        final_answer = extract_answer(sim_actor_resp, task_type)
        if normalize_answer(final_answer or "", task_type=task_type) == correct_answer:
            correct_count += 1

    return correct_count / num_simulations
