"""
Trajectory generation: Algorithm 1 from the ACC-Collab paper.

Generates natural + guided deliberation trajectories and builds
preference pairs based on reward deltas.

Optimized: merges all MC roll-out prompts into a single batch call per round,
and supports cross-sample batched deliberation.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from src.algorithms.deliberation import deliberate, deliberate_batch
from src.algorithms.rollout import estimate_final_accuracy
from src.algorithms.reward import compute_reward_delta, extract_answer, normalize_answer
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
    task_type = sample.get("task_type", "yes_no")

    correct_answer = sample.get("answer", "")
    wrong_answer = generate_wrong_answer(correct_answer, sample.get("choices"), task_type=task_type, rng=rng)

    preference_pairs = []

    # Run natural deliberation
    natural_trajectory = deliberate(
        actor_model, critic_model, sample, dataset_name,
        num_rounds=num_rounds, max_tokens=max_tokens, temperature=temperature,
    )

    # Algorithm 1: guided trajectories start from t=1 (z(0) only runs natural deliberation)
    for t in range(1, len(natural_trajectory)):
        round_data = natural_trajectory[t]
        actor_response = round_data["actor_response"]
        critic_response = round_data["critic_response"]
        actor_prompt = round_data["actor_prompt"]
        critic_prompt = round_data["critic_prompt"]
        # Include both actor and critic responses interleaved,
        # matching the format produced by deliberate()
        previous_responses = []
        for r in natural_trajectory[:t]:
            previous_responses.append(r["actor_response"])
            previous_responses.append(r["critic_response"])

        # --- Merged guided generation: batch actor (2) + critic (2) in fewer calls ---

        # Step 1: Generate guided actor responses (batch of 2)
        z_y_actor_prompt = _make_guided_prompt(
            dataset_name, sample, correct_answer, t, previous_responses,
            actor_response, agent="actor",
        )
        z_not_y_actor_prompt = _make_guided_prompt(
            dataset_name, sample, wrong_answer, t, previous_responses,
            actor_response, agent="actor",
        )

        guided_actor_resps = actor_model.generate(
            [z_y_actor_prompt, z_not_y_actor_prompt],
            max_tokens=max_tokens, temperature=temperature,
        )
        z_y_actor = guided_actor_resps[0]
        z_not_y_actor = guided_actor_resps[1]

        # Step 2: Generate guided critic responses (batch of 2)
        z_y_critic_prompt = _make_guided_prompt(
            dataset_name, sample, correct_answer, t, previous_responses,
            z_y_actor, agent="critic",
        )
        z_not_y_critic_prompt = _make_guided_prompt(
            dataset_name, sample, wrong_answer, t, previous_responses,
            z_not_y_actor, agent="critic",
        )

        guided_critic_resps = critic_model.generate(
            [z_y_critic_prompt, z_not_y_critic_prompt],
            max_tokens=max_tokens, temperature=temperature,
        )
        z_y_critic = guided_critic_resps[0]
        z_not_y_critic = guided_critic_resps[1]

        # Step 3: Merged MC roll-out — batch all 3 reward estimations into 1 call
        # Instead of 3 separate calls (v_natural, v_guided_correct, v_guided_wrong),
        # we batch all simulation prompts into a single actor_model.generate()
        all_sim_prompts = []
        sim_groups = []  # [(start_idx, count), ...] to split results back

        for prefix_actor, prefix_critic in [
            (actor_response, critic_response),         # v_natural
            (z_y_actor, z_y_critic),                    # v_guided_correct
            (z_not_y_actor, z_not_y_critic),            # v_guided_wrong
        ]:
            start = len(all_sim_prompts)
            for _ in range(num_simulations):
                sim_responses = list(previous_responses) + [prefix_actor, prefix_critic]
                all_sim_prompts.append(format_prompt(
                    dataset_name, PromptType.DELIBERATION_ACTOR, sample,
                    responses=sim_responses,
                ))
            sim_groups.append((start, num_simulations))

        # Single batch call for ALL MC simulations (3 * num_simulations prompts)
        all_sim_results = actor_model.generate(
            all_sim_prompts, max_tokens=max_tokens, temperature=temperature,
        )

        # Parse results back into 3 groups
        correct_norm = normalize_answer(correct_answer, task_type=task_type)

        def _count_correct(responses_slice):
            count = 0
            for resp in responses_slice:
                ans = extract_answer(resp, task_type)
                if normalize_answer(ans or "", task_type=task_type) == correct_norm:
                    count += 1
            return count / len(responses_slice) if responses_slice else 0.0

        v_natural = _count_correct(all_sim_results[sim_groups[0][0]:sim_groups[0][0] + sim_groups[0][1]])
        v_guided_correct = _count_correct(all_sim_results[sim_groups[1][0]:sim_groups[1][0] + sim_groups[1][1]])
        v_guided_wrong = _count_correct(all_sim_results[sim_groups[2][0]:sim_groups[2][0] + sim_groups[2][1]])

        # Compute deltas
        delta_y = compute_reward_delta(v_guided_correct, v_natural)
        delta_not_y = compute_reward_delta(v_natural, v_guided_wrong)

        # Build preference pairs (Algorithm 1: only one pair per round via elif)
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
        elif delta_not_y >= reward_threshold:
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


def generate_trajectories_batch(
    actor_model,
    critic_model,
    samples: list[dict],
    dataset_name: str,
    num_rounds: int = 5,
    reward_threshold: float = 0.0,
    num_simulations: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 42,
    batch_size: int = 4,
) -> list[dict]:
    """
    Batched trajectory generation: process multiple samples at once.

    Uses deliberate_batch() for natural deliberation (merges all samples'
    actor+critic calls into 2 calls per round), then generates guided
    trajectories per-sample (since each sample has different guided targets).

    This reduces vLLM calls from 30*N to ~(10 + 20)*ceil(N/batch_size) for
    the natural deliberation phase.

    Args:
        actor_model: VLLMInference for actor.
        critic_model: VLLMInference for critic.
        samples: List of standardized samples.
        dataset_name: Dataset name.
        num_rounds: Total deliberation rounds.
        reward_threshold: Minimum delta for preference pairs.
        num_simulations: MC roll-out simulations.
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.
        seed: Random seed.
        batch_size: Number of samples to deliberate in parallel.

    Returns:
        List of all preference pairs across all samples.
    """
    all_pairs = []

    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start:batch_start + batch_size]
        logger.info(
            f"Batched trajectories: samples {batch_start+1}-{batch_start+len(batch)}"
            f"/{len(samples)}"
        )

        # Batch natural deliberation
        trajectories = deliberate_batch(
            actor_model, critic_model, batch, dataset_name,
            num_rounds=num_rounds, max_tokens=max_tokens,
            temperature=temperature,
        )

        # Generate guided trajectories per sample (within the batch)
        for i, (sample, natural_trajectory) in enumerate(zip(batch, trajectories)):
            rng = random.Random(seed + batch_start + i)
            correct_answer = sample.get("answer", "")
            task_type = sample.get("task_type", "yes_no")
            wrong_answer = generate_wrong_answer(
                correct_answer, sample.get("choices"), task_type=task_type, rng=rng,
            )

            for t in range(1, len(natural_trajectory)):
                round_data = natural_trajectory[t]
                actor_response = round_data["actor_response"]
                critic_response = round_data["critic_response"]
                actor_prompt = round_data["actor_prompt"]
                critic_prompt = round_data["critic_prompt"]

                previous_responses = []
                for r in natural_trajectory[:t]:
                    previous_responses.append(r["actor_response"])
                    previous_responses.append(r["critic_response"])

                # Guided actor (batch of 2)
                z_y_actor_prompt = _make_guided_prompt(
                    dataset_name, sample, correct_answer, t,
                    previous_responses, actor_response, agent="actor",
                )
                z_not_y_actor_prompt = _make_guided_prompt(
                    dataset_name, sample, wrong_answer, t,
                    previous_responses, actor_response, agent="actor",
                )
                guided_actor_resps = actor_model.generate(
                    [z_y_actor_prompt, z_not_y_actor_prompt],
                    max_tokens=max_tokens, temperature=temperature,
                )
                z_y_actor = guided_actor_resps[0]
                z_not_y_actor = guided_actor_resps[1]

                # Guided critic (batch of 2)
                z_y_critic_prompt = _make_guided_prompt(
                    dataset_name, sample, correct_answer, t,
                    previous_responses, z_y_actor, agent="critic",
                )
                z_not_y_critic_prompt = _make_guided_prompt(
                    dataset_name, sample, wrong_answer, t,
                    previous_responses, z_not_y_actor, agent="critic",
                )
                guided_critic_resps = critic_model.generate(
                    [z_y_critic_prompt, z_not_y_critic_prompt],
                    max_tokens=max_tokens, temperature=temperature,
                )
                z_y_critic = guided_critic_resps[0]
                z_not_y_critic = guided_critic_resps[1]

                # Merged MC roll-out (3 * num_sims in 1 batch call)
                all_sim_prompts = []
                sim_groups = []

                for prefix_actor, prefix_critic in [
                    (actor_response, critic_response),
                    (z_y_actor, z_y_critic),
                    (z_not_y_actor, z_not_y_critic),
                ]:
                    start = len(all_sim_prompts)
                    for _ in range(num_simulations):
                        sim_responses = list(previous_responses) + [prefix_actor, prefix_critic]
                        all_sim_prompts.append(format_prompt(
                            dataset_name, PromptType.DELIBERATION_ACTOR, sample,
                            responses=sim_responses,
                        ))
                    sim_groups.append((start, num_simulations))

                all_sim_results = actor_model.generate(
                    all_sim_prompts, max_tokens=max_tokens,
                    temperature=temperature,
                )

                correct_norm = normalize_answer(correct_answer, task_type=task_type)

                def _count_correct(responses_slice):
                    count = 0
                    for resp in responses_slice:
                        ans = extract_answer(resp, task_type)
                        if normalize_answer(ans or "", task_type=task_type) == correct_norm:
                            count += 1
                    return count / len(responses_slice) if responses_slice else 0.0

                v_natural = _count_correct(
                    all_sim_results[sim_groups[0][0]:sim_groups[0][0] + sim_groups[0][1]]
                )
                v_guided_correct = _count_correct(
                    all_sim_results[sim_groups[1][0]:sim_groups[1][0] + sim_groups[1][1]]
                )
                v_guided_wrong = _count_correct(
                    all_sim_results[sim_groups[2][0]:sim_groups[2][0] + sim_groups[2][1]]
                )

                delta_y = compute_reward_delta(v_guided_correct, v_natural)
                delta_not_y = compute_reward_delta(v_natural, v_guided_wrong)

                if delta_y >= reward_threshold:
                    all_pairs.append({
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
                elif delta_not_y >= reward_threshold:
                    all_pairs.append({
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

    return all_pairs


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
