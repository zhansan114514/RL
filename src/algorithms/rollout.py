"""
One-step roll-out Monte Carlo for reward estimation.

Following the ACC-Collab paper (Section 4.2), this estimates the partial reward
r(z^(t), x, y) by simulating ONE additional deliberation round from the current
response, repeated multiple times for Monte Carlo estimation.
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

    Per the ACC-Collab paper, simulates ONE additional deliberation round
    (actor + critic exchange) from the current state, repeated num_simulations
    times, and returns the average accuracy of the final answers.

    Args:
        actor_model: Actor VLLMInference.
        critic_model: Critic VLLMInference.
        sample: Task sample.
        dataset_name: Dataset name.
        current_actor_response: Actor's response at round t.
        current_critic_response: Critic's response at round t.
        previous_responses: Responses from prior rounds.
        num_simulations: Number of MC simulations (default: 5).
        remaining_rounds: Rounds to simulate (should be 1 for one-step).
        max_tokens: Max tokens per generation.
        temperature: Sampling temperature.

    Returns:
        Estimated accuracy (float in [0, 1]).
    """
    correct_answer = normalize_answer(sample.get("answer", ""))
    if not correct_answer:
        return 0.0

    if num_simulations <= 0:
        return 0.0

    correct_count = 0

    for _ in range(num_simulations):
        sim_responses = list(previous_responses) + [current_actor_response]
        sim_actor_resp = current_actor_response

        for r in range(remaining_rounds):
            actor_prompt = format_prompt(
                dataset_name, PromptType.DELIBERATION_ACTOR, sample,
                responses=sim_responses,
            )
            sim_actor_resp = actor_model.generate_single(
                actor_prompt, max_tokens=max_tokens, temperature=temperature,
            )

            critic_prompt = format_prompt(
                dataset_name, PromptType.DELIBERATION_CRITIC, sample,
                actor_response=sim_actor_resp,
            )
            sim_critic_resp = critic_model.generate_single(
                critic_prompt, max_tokens=max_tokens, temperature=temperature,
            )

            sim_responses.append(sim_actor_resp)

        task_type = sample.get("task_type", "yes_no")
        final_answer = extract_answer(sim_actor_resp, task_type)
        if normalize_answer(final_answer or "") == correct_answer:
            correct_count += 1

    return correct_count / num_simulations
