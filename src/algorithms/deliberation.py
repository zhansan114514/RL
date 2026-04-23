"""
Deliberation engine: natural and guided multi-round Actor-Critic deliberation.

Implements the core iterative discussion loop from the ACC-Collab paper.
Optimized: batches actor and critic calls within each round for vLLM throughput.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.prompts.templates import PromptType
from src.prompts.formatter import format_prompt
from src.algorithms.reward import extract_answer

logger = logging.getLogger(__name__)


def deliberate(
    actor_model,
    critic_model,
    sample: dict,
    dataset_name: str,
    num_rounds: int = 5,
    max_tokens: int = 512,
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

        # Update response history for next round (interleaved actor and critic responses)
        previous_responses.append(actor_response)
        previous_responses.append(critic_response)

    return trajectory


def deliberate_batch(
    actor_model,
    critic_model,
    samples: list[dict],
    dataset_name: str,
    num_rounds: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> list[list[dict]]:
    """
    Batched deliberation: process multiple samples in parallel for vLLM throughput.

    Instead of sample-by-sample sequential generation, batches all actor prompts
    across samples into a single vLLM call, then all critic prompts, for each round.
    This dramatically improves GPU utilization with vLLM's continuous batching.

    For N samples, this reduces from 2*N*T serial calls to 2*T batched calls.

    Args:
        actor_model: VLLMInference for the actor.
        critic_model: VLLMInference for the critic.
        samples: List of standardized sample dicts.
        dataset_name: Dataset name for prompt selection.
        num_rounds: Number of rounds per sample.
        max_tokens: Max tokens per generation.
        temperature: Sampling temperature.

    Returns:
        List of trajectories, one per sample (same as calling deliberate() N times).
    """
    n = len(samples)
    if n == 0:
        return []
    # Single sample: fall back to non-batched (no overhead)
    if n == 1:
        return [deliberate(actor_model, critic_model, samples[0], dataset_name,
                           num_rounds, max_tokens, temperature)]

    # Per-sample state
    all_trajectories: list[list[dict]] = [[] for _ in range(n)]
    all_previous: list[list[str]] = [[] for _ in range(n)]

    for t in range(num_rounds):
        # --- Build all actor prompts for this round ---
        actor_prompts = []
        for i, sample in enumerate(samples):
            if t == 0:
                prompt = format_prompt(dataset_name, PromptType.SINGLE_SHOT, sample)
            else:
                prompt = format_prompt(
                    dataset_name, PromptType.DELIBERATION_ACTOR, sample,
                    responses=all_previous[i],
                )
            actor_prompts.append(prompt)

        # Batch actor inference: all samples at once
        actor_responses = actor_model.generate(
            actor_prompts, max_tokens=max_tokens, temperature=temperature,
        )

        # --- Build all critic prompts for this round ---
        critic_prompts = []
        for i, sample in enumerate(samples):
            prompt = format_prompt(
                dataset_name, PromptType.DELIBERATION_CRITIC, sample,
                actor_response=actor_responses[i],
            )
            critic_prompts.append(prompt)

        # Batch critic inference: all samples at once
        critic_responses = critic_model.generate(
            critic_prompts, max_tokens=max_tokens, temperature=temperature,
        )

        # Record rounds
        for i, sample in enumerate(samples):
            task_type = sample.get("task_type", "yes_no")
            actor_answer = extract_answer(actor_responses[i], task_type)

            all_trajectories[i].append({
                "round": t,
                "actor_prompt": actor_prompts[i],
                "actor_response": actor_responses[i],
                "actor_answer": actor_answer,
                "critic_prompt": critic_prompts[i],
                "critic_response": critic_responses[i],
            })

            all_previous[i].append(actor_responses[i])
            all_previous[i].append(critic_responses[i])

    return all_trajectories
