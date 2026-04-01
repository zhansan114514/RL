"""
Trajectory generation: Algorithm 1 from the ACC-Collab paper.

Generates natural + guided deliberation trajectories and builds
preference pairs based on reward deltas.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from src.algorithms.deliberation import deliberate
from src.algorithms.rollout import estimate_final_accuracy
from src.algorithms.reward import compute_reward_delta
from src.prompts.templates import PromptType
from src.prompts.formatter import format_prompt
from src.data.preprocessor import generate_wrong_answer

logger = logging.getLogger(__name__)


def generate_trajectories(
    actor_model,
    critic_model,
    sample: dict,
    dataset_name: str,
    num_rounds: int = 5,
    reward_threshold: float = 0.0,
    num_simulations: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 42,
) -> list[dict]:
    """
    Implement Algorithm 1: generate and select deliberation trajectories.

    For each sample and each round:
    1. Generate natural deliberation trajectory
    2. Generate guided trajectory towards correct answer (z_y)
    3. Generate guided trajectory away from correct answer (z_not_y)
    4. Estimate rewards for each via Monte Carlo roll-out
    5. Build preference pairs where delta >= epsilon

    Args:
        actor_model: VLLMInference for actor.
        critic_model: VLLMInference for critic.
        sample: Standardized sample.
        dataset_name: Dataset name.
        num_rounds: Total deliberation rounds.
        reward_threshold: Minimum delta for preference pairs.
        num_simulations: MC roll-out simulations.
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.
        seed: Random seed for reproducibility.

    Returns:
        List of preference pair dicts with keys:
            positive, negative, round, delta, sample_idx
    """
    rng = random.Random(seed)

    correct_answer = sample.get("answer", "")
    wrong_answer = generate_wrong_answer(correct_answer, sample.get("choices"), rng=rng)

    preference_pairs = []

    # Run natural deliberation
    natural_trajectory = deliberate(
        actor_model, critic_model, sample, dataset_name,
        num_rounds=num_rounds, max_tokens=max_tokens, temperature=temperature,
    )

    for t, round_data in enumerate(natural_trajectory):
        actor_response = round_data["actor_response"]
        critic_response = round_data["critic_response"]
        actor_prompt = round_data["actor_prompt"]
        critic_prompt = round_data["critic_prompt"]
        previous_responses = [
            r["actor_response"] for r in natural_trajectory[:t]
        ]

        # Estimate natural reward via one-step roll-out
        v_natural = estimate_final_accuracy(
            actor_model, critic_model, sample, dataset_name,
            current_actor_response=actor_response,
            current_critic_response=critic_response,
            previous_responses=previous_responses,
            num_simulations=num_simulations,
            remaining_rounds=1,
            max_tokens=max_tokens, temperature=temperature,
        )

        # Guided towards correct answer
        z_y_actor_prompt = _make_guided_prompt(
            dataset_name, sample, correct_answer, t, previous_responses,
            actor_response, agent="actor",
        )
        z_y_actor = actor_model.generate_single(
            z_y_actor_prompt, max_tokens=max_tokens, temperature=temperature,
        )
        z_y_critic_prompt = _make_guided_prompt(
            dataset_name, sample, correct_answer, t, previous_responses,
            z_y_actor, agent="critic",
        )
        z_y_critic = critic_model.generate_single(
            z_y_critic_prompt, max_tokens=max_tokens, temperature=temperature,
        )

        v_guided_correct = estimate_final_accuracy(
            actor_model, critic_model, sample, dataset_name,
            current_actor_response=z_y_actor,
            current_critic_response=z_y_critic,
            previous_responses=previous_responses,
            num_simulations=num_simulations,
            remaining_rounds=1,
            max_tokens=max_tokens, temperature=temperature,
        )

        # Guided away from correct answer
        z_not_y_actor_prompt = _make_guided_prompt(
            dataset_name, sample, wrong_answer, t, previous_responses,
            actor_response, agent="actor",
        )
        z_not_y_actor = actor_model.generate_single(
            z_not_y_actor_prompt, max_tokens=max_tokens, temperature=temperature,
        )
        z_not_y_critic_prompt = _make_guided_prompt(
            dataset_name, sample, wrong_answer, t, previous_responses,
            z_not_y_actor, agent="critic",
        )
        z_not_y_critic = critic_model.generate_single(
            z_not_y_critic_prompt, max_tokens=max_tokens, temperature=temperature,
        )

        v_guided_wrong = estimate_final_accuracy(
            actor_model, critic_model, sample, dataset_name,
            current_actor_response=z_not_y_actor,
            current_critic_response=z_not_y_critic,
            previous_responses=previous_responses,
            num_simulations=num_simulations,
            remaining_rounds=1,
            max_tokens=max_tokens, temperature=temperature,
        )

        # Compute deltas
        delta_y = compute_reward_delta(v_guided_correct, v_natural)
        delta_not_y = compute_reward_delta(v_natural, v_guided_wrong)

        # Build preference pairs
        if delta_y >= reward_threshold:
            preference_pairs.append({
                "actor_prompt": actor_prompt,
                "critic_prompt": critic_prompt,
                "positive": z_y_actor,
                "negative": actor_response,
                "positive_critic": z_y_critic,
                "negative_critic": critic_response,
                "round": t,
                "delta": delta_y,
                "direction": "towards",
                "agent": "actor",
            })

        if delta_not_y >= reward_threshold:
            preference_pairs.append({
                "actor_prompt": actor_prompt,
                "critic_prompt": critic_prompt,
                "positive": actor_response,
                "negative": z_not_y_actor,
                "positive_critic": critic_response,
                "negative_critic": z_not_y_critic,
                "round": t,
                "delta": delta_not_y,
                "direction": "away",
                "agent": "actor",
            })

    return preference_pairs


def _make_guided_prompt(
    dataset_name: str,
    sample: dict,
    target_answer: str,
    round_idx: int,
    previous_responses: list[str],
    actor_response: str,
    agent: str = "actor",
) -> str:
    """Build a guided deliberation prompt."""
    if agent == "actor":
        if round_idx == 0:
            prompt_type = PromptType.GUIDED_SINGLE_SHOT
            return format_prompt(
                dataset_name, prompt_type, sample,
                target_answer=target_answer,
            )
        else:
            prompt_type = PromptType.GUIDED_DELIBERATION_ACTOR
            return format_prompt(
                dataset_name, prompt_type, sample,
                target_answer=target_answer,
                responses=previous_responses,
            )
    else:  # critic
        prompt_type = PromptType.GUIDED_DELIBERATION_CRITIC
        return format_prompt(
            dataset_name, prompt_type, sample,
            target_answer=target_answer,
            actor_response=actor_response,
        )
