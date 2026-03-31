"""
One-step roll-out Monte Carlo for reward estimation.

Following the ACC-Collab paper (Section 4.2), this estimates the partial reward
r(z^(t), x, y) by simulating ONE additional deliberation round from the current
response (not all remaining rounds), repeated multiple times for Monte Carlo
estimation.

Paper quote: "we use one-step roll-out heuristics, i.e. simulating an additional
deliberation round multiple times from response z(t)."
"""

from __future__ import annotations

import logging
from typing import Optional

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

    Per the ACC-Collab paper, this simulates ONE additional deliberation round
    (actor + critic exchange) from the current state, repeated num_simulations
    times, and returns the average accuracy of the final answers.

    The key insight is that we only need to look ONE step ahead to estimate
    the quality of the current response, not simulate all remaining rounds.

    Args:
        actor_model: Actor VLLMInference.
        critic_model: Critic VLLMInference.
        sample: Task sample.
        dataset_name: Dataset name.
        current_actor_response: Actor's response at round t.
        current_critic_response: Critic's response at round t.
        previous_responses: Responses from prior rounds (for context).
        num_simulations: Number of MC simulations (default: 5).
        remaining_rounds: Rounds to simulate (should be 1 for one-step roll-out).
        max_tokens: Max tokens per generation.
        temperature: Sampling temperature.

    Returns:
        Estimated accuracy (float in [0, 1]) - the proportion of simulations
        that led to the correct answer.
    """
    from src.reward.accuracy import extract_answer, normalize_answer

    correct_answer = normalize_answer(sample.get("answer", ""))
    if not correct_answer:
        return 0.0

    correct_count = 0

    for _ in range(num_simulations):
        # For one-step roll-out from current state, we include:
        # - previous responses (rounds 0 to t-1)
        # - current actor response (round t)
        # The current critic response is implicitly part of the state context
        sim_responses = list(previous_responses) + [current_actor_response]
        sim_actor_resp = current_actor_response

        for r in range(remaining_rounds):
            # Actor generates next response
            from src.prompts.templates import PromptType
            from src.prompts.formatter import format_prompt

            actor_prompt = format_prompt(
                dataset_name, PromptType.DELIBERATION_ACTOR, sample,
                responses=sim_responses,
            )
            sim_actor_resp = actor_model.generate_single(
                actor_prompt, max_tokens=max_tokens, temperature=temperature,
            )

            # Critic generates feedback
            critic_prompt = format_prompt(
                dataset_name, PromptType.DELIBERATION_CRITIC, sample,
                actor_response=sim_actor_resp,
            )
            sim_critic_resp = critic_model.generate_single(
                critic_prompt, max_tokens=max_tokens, temperature=temperature,
            )

            sim_responses.append(sim_actor_resp)

        # Check final answer
        task_type = sample.get("task_type", "yes_no")
        final_answer = extract_answer(sim_actor_resp, task_type)
        if normalize_answer(final_answer or "") == correct_answer:
            correct_count += 1

    return correct_count / num_simulations
