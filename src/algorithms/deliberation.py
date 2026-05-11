"""Simple natural Actor-Critic deliberation loop."""

from __future__ import annotations

import logging

from src.algorithms.reward import extract_answer
from src.prompts.prompt_builder import build_simple_actor_prompt, build_simple_critic_prompt

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
    """Run a single Actor/Critic natural deliberation trajectory."""
    trajectory = []
    previous_actor_response = ""
    critic_feedback = ""

    for round_idx in range(num_rounds):
        actor_prompt = build_simple_actor_prompt(
            sample,
            dataset_name,
            round_num=round_idx,
            previous_actor_response=previous_actor_response,
            critic_feedback=critic_feedback,
        )
        actor_response = actor_model.generate_single(
            actor_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        actor_answer = extract_answer(actor_response, sample.get("task_type", "yes_no"))

        critic_prompt = build_simple_critic_prompt(
            sample,
            dataset_name,
            actor_response=actor_response,
        )
        critic_response = critic_model.generate_single(
            critic_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        trajectory.append({
            "round": round_idx,
            "actor_prompt": actor_prompt,
            "actor_response": actor_response,
            "actor_answer": actor_answer,
            "critic_prompt": critic_prompt,
            "critic_response": critic_response,
        })

        previous_actor_response = actor_response
        critic_feedback = critic_response

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
    """Batched natural Actor/Critic deliberation."""
    if not samples:
        return []
    if len(samples) == 1:
        return [
            deliberate(
                actor_model,
                critic_model,
                samples[0],
                dataset_name,
                num_rounds=num_rounds,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        ]

    trajectories: list[list[dict]] = [[] for _ in samples]
    previous_actor_responses = [""] * len(samples)
    critic_feedbacks = [""] * len(samples)

    for round_idx in range(num_rounds):
        actor_prompts = [
            build_simple_actor_prompt(
                sample,
                dataset_name,
                round_num=round_idx,
                previous_actor_response=previous_actor_responses[i],
                critic_feedback=critic_feedbacks[i],
            )
            for i, sample in enumerate(samples)
        ]
        actor_responses = actor_model.generate(
            actor_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        critic_prompts = [
            build_simple_critic_prompt(
                sample,
                dataset_name,
                actor_response=actor_responses[i],
            )
            for i, sample in enumerate(samples)
        ]
        critic_responses = critic_model.generate(
            critic_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        for i, sample in enumerate(samples):
            actor_answer = extract_answer(
                actor_responses[i],
                sample.get("task_type", "yes_no"),
            )
            trajectories[i].append({
                "round": round_idx,
                "actor_prompt": actor_prompts[i],
                "actor_response": actor_responses[i],
                "actor_answer": actor_answer,
                "critic_prompt": critic_prompts[i],
                "critic_response": critic_responses[i],
            })
            previous_actor_responses[i] = actor_responses[i]
            critic_feedbacks[i] = critic_responses[i]

    return trajectories
