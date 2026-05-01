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

from src.algorithms.deliberation import deliberate, deliberate_batch
from src.algorithms.reward import compute_reward_delta, extract_answer, normalize_answer
from src.prompts.templates import PromptType
from src.prompts.formatter import format_prompt
from src.data.preprocessor import generate_wrong_answer

logger = logging.getLogger(__name__)


# ============================================================
# Core: per-sample guided trajectories + MC roll-out + pairs
# ============================================================

def _generate_guided_pairs_for_sample(
    actor_model,
    critic_model,
    sample: dict,
    natural_trajectory: list[dict],
    dataset_name: str,
    correct_answer: str,
    wrong_answer: str,
    reward_threshold: float = 0.0,
    num_simulations: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> list[dict]:
    """Run guided trajectory generation + MC roll-out for a single sample.

    Given the natural deliberation trajectory already produced by
    ``deliberate()`` or ``deliberate_batch()``, this function implements the
    core of Algorithm 1: for each available round, generate guided trajectories
    (toward correct / toward wrong), run MC roll-out reward estimation, and
    build preference pairs where the reward delta exceeds *reward_threshold*.

    Returns a list of preference-pair dicts (may be empty).
    """
    task_type = sample.get("task_type", "yes_no")
    preference_pairs: list[dict] = []

    for t in range(len(natural_trajectory)):
        round_data = natural_trajectory[t]
        actor_response = round_data["actor_response"]
        critic_response = round_data["critic_response"]
        actor_prompt = round_data["actor_prompt"]
        critic_prompt = round_data["critic_prompt"]

        # Interleave actor + critic responses from prior rounds
        previous_responses = []
        for r in natural_trajectory[:t]:
            previous_responses.append(r["actor_response"])
            previous_responses.append(r["critic_response"])

        # --- Guided actor responses (batch of 2) ---
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

        # --- Guided critic responses (batch of 2) ---
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

        # --- Merged MC roll-out with actor-critic simulation ---
        # Each simulation runs:
        #   Phase A: actor generates response from prefix context
        #   Phase B: critic generates feedback for actor's response
        #   Phase C: actor generates refined response with critic feedback
        # Final correctness is checked on Phase C responses, matching the
        # actual deliberation process where the actor sees critic feedback.
        prefix_pairs = [
            (actor_response, critic_response),      # v_natural
            (z_y_actor, z_y_critic),                 # v_guided_correct
            (z_not_y_actor, z_not_y_critic),         # v_guided_wrong
        ]
        n_sims = num_simulations

        # Phase A: actor generates initial responses
        phase_a_prompts = []
        sim_groups: list[tuple[int, int]] = []
        for prefix_actor, prefix_critic in prefix_pairs:
            start = len(phase_a_prompts)
            for _ in range(n_sims):
                sim_ctx = list(previous_responses) + [prefix_actor, prefix_critic]
                phase_a_prompts.append(format_prompt(
                    dataset_name, PromptType.DELIBERATION_ACTOR, sample,
                    responses=sim_ctx,
                ))
            sim_groups.append((start, n_sims))

        phase_a_results = actor_model.generate(
            phase_a_prompts, max_tokens=max_tokens, temperature=temperature,
        )

        # Phase B: critic generates feedback for each simulated actor response
        phase_b_prompts = [
            format_prompt(
                dataset_name, PromptType.DELIBERATION_CRITIC, sample,
                actor_response=resp,
            )
            for resp in phase_a_results
        ]
        phase_b_results = critic_model.generate(
            phase_b_prompts, max_tokens=max_tokens, temperature=temperature,
        )

        # Phase C: actor generates refined responses with critic feedback
        phase_c_prompts = []
        sim_idx = 0
        for prefix_actor, prefix_critic in prefix_pairs:
            for _ in range(n_sims):
                sim_ctx = (
                    list(previous_responses)
                    + [prefix_actor, prefix_critic]
                    + [phase_a_results[sim_idx], phase_b_results[sim_idx]]
                )
                phase_c_prompts.append(format_prompt(
                    dataset_name, PromptType.DELIBERATION_ACTOR, sample,
                    responses=sim_ctx,
                ))
                sim_idx += 1

        phase_c_results = actor_model.generate(
            phase_c_prompts, max_tokens=max_tokens, temperature=temperature,
        )

        # --- Reward estimation ---
        correct_norm = normalize_answer(correct_answer, task_type=task_type)

        def _count_correct(responses_slice: list[str]) -> float:
            count = 0
            for resp in responses_slice:
                ans = extract_answer(resp, task_type)
                if normalize_answer(ans or "", task_type=task_type) == correct_norm:
                    count += 1
            return count / len(responses_slice) if responses_slice else 0.0

        v_natural = _count_correct(
            phase_c_results[sim_groups[0][0]:sim_groups[0][0] + sim_groups[0][1]]
        )
        v_guided_correct = _count_correct(
            phase_c_results[sim_groups[1][0]:sim_groups[1][0] + sim_groups[1][1]]
        )
        v_guided_wrong = _count_correct(
            phase_c_results[sim_groups[2][0]:sim_groups[2][0] + sim_groups[2][1]]
        )

        # Compute deltas
        delta_y = compute_reward_delta(v_guided_correct, v_natural)
        delta_not_y = compute_reward_delta(v_natural, v_guided_wrong)

        # Build preference pairs (Algorithm 1: delta >= epsilon)
        if delta_y >= reward_threshold:
            preference_pairs.append({
                "sample": sample,
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
                "sample": sample,
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


def _generate_guided_pairs_for_batch(
    actor_model,
    critic_model,
    samples: list[dict],
    natural_trajectories: list[list[dict]],
    dataset_name: str,
    correct_answers: list[str],
    wrong_answers: list[str],
    reward_threshold: float = 0.0,
    num_simulations: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.7,
    sample_offset: int = 0,
) -> list[dict]:
    """Run guided trajectory generation + MC roll-out for a batch of samples.

    This is the throughput-oriented Algorithm 1 path. For each round, it
    batches all samples' guided actor prompts, guided critic prompts, and the
    three MC rollout phases into large vLLM calls instead of serializing by
    sample.
    """
    if not samples:
        return []

    if not (
        len(samples)
        == len(natural_trajectories)
        == len(correct_answers)
        == len(wrong_answers)
    ):
        raise ValueError("samples, trajectories, and answers must have equal length")

    preference_pairs: list[dict] = []
    max_rounds = max((len(traj) for traj in natural_trajectories), default=0)

    for t in range(max_rounds):
        active = [
            i for i, traj in enumerate(natural_trajectories)
            if t < len(traj)
        ]
        if not active:
            continue

        previous_by_sample: dict[int, list[str]] = {}
        for i in active:
            previous: list[str] = []
            for r in natural_trajectories[i][:t]:
                previous.append(r["actor_response"])
                previous.append(r["critic_response"])
            previous_by_sample[i] = previous

        # --- Guided actor responses: 2 * B prompts in one actor call ---
        guided_actor_prompts: list[str] = []
        guided_actor_meta: list[tuple[int, str]] = []
        for i in active:
            sample = samples[i]
            round_data = natural_trajectories[i][t]
            guided_actor_prompts.append(_make_guided_prompt(
                dataset_name, sample, correct_answers[i], t,
                previous_by_sample[i], round_data["actor_response"],
                agent="actor",
            ))
            guided_actor_meta.append((i, "correct"))
            guided_actor_prompts.append(_make_guided_prompt(
                dataset_name, sample, wrong_answers[i], t,
                previous_by_sample[i], round_data["actor_response"],
                agent="actor",
            ))
            guided_actor_meta.append((i, "wrong"))

        guided_actor_results = actor_model.generate(
            guided_actor_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        guided_actor: dict[int, dict[str, str]] = {
            i: {} for i in active
        }
        for (i, direction), response in zip(guided_actor_meta, guided_actor_results):
            guided_actor[i][direction] = response

        # --- Guided critic responses: 2 * B prompts in one critic call ---
        guided_critic_prompts: list[str] = []
        guided_critic_meta: list[tuple[int, str]] = []
        for i in active:
            sample = samples[i]
            guided_critic_prompts.append(_make_guided_prompt(
                dataset_name, sample, correct_answers[i], t,
                previous_by_sample[i], guided_actor[i]["correct"],
                agent="critic",
            ))
            guided_critic_meta.append((i, "correct"))
            guided_critic_prompts.append(_make_guided_prompt(
                dataset_name, sample, wrong_answers[i], t,
                previous_by_sample[i], guided_actor[i]["wrong"],
                agent="critic",
            ))
            guided_critic_meta.append((i, "wrong"))

        guided_critic_results = critic_model.generate(
            guided_critic_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        guided_critic: dict[int, dict[str, str]] = {
            i: {} for i in active
        }
        for (i, direction), response in zip(guided_critic_meta, guided_critic_results):
            guided_critic[i][direction] = response

        # --- MC rollout Phase A: actor responses from three prefixes ---
        phase_a_prompts: list[str] = []
        phase_a_meta: list[tuple[int, int]] = []
        for i in active:
            sample = samples[i]
            round_data = natural_trajectories[i][t]
            prefix_pairs = [
                (round_data["actor_response"], round_data["critic_response"]),
                (guided_actor[i]["correct"], guided_critic[i]["correct"]),
                (guided_actor[i]["wrong"], guided_critic[i]["wrong"]),
            ]
            for prefix_idx, (prefix_actor, prefix_critic) in enumerate(prefix_pairs):
                for _ in range(num_simulations):
                    sim_ctx = (
                        list(previous_by_sample[i])
                        + [prefix_actor, prefix_critic]
                    )
                    phase_a_prompts.append(format_prompt(
                        dataset_name, PromptType.DELIBERATION_ACTOR, sample,
                        responses=sim_ctx,
                    ))
                    phase_a_meta.append((i, prefix_idx))

        phase_a_results = actor_model.generate(
            phase_a_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        ) if phase_a_prompts else []

        # --- MC rollout Phase B: critic feedback for each simulated response ---
        phase_b_prompts = [
            format_prompt(
                dataset_name, PromptType.DELIBERATION_CRITIC, samples[i],
                actor_response=response,
            )
            for (i, _), response in zip(phase_a_meta, phase_a_results)
        ]
        phase_b_results = critic_model.generate(
            phase_b_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        ) if phase_b_prompts else []

        # --- MC rollout Phase C: actor refined response with critic feedback ---
        phase_c_prompts: list[str] = []
        phase_c_meta: list[tuple[int, int]] = []
        for (i, prefix_idx), actor_resp, critic_resp in zip(
            phase_a_meta, phase_a_results, phase_b_results,
        ):
            sample = samples[i]
            round_data = natural_trajectories[i][t]
            if prefix_idx == 0:
                prefix_actor = round_data["actor_response"]
                prefix_critic = round_data["critic_response"]
            elif prefix_idx == 1:
                prefix_actor = guided_actor[i]["correct"]
                prefix_critic = guided_critic[i]["correct"]
            else:
                prefix_actor = guided_actor[i]["wrong"]
                prefix_critic = guided_critic[i]["wrong"]

            sim_ctx = (
                list(previous_by_sample[i])
                + [prefix_actor, prefix_critic, actor_resp, critic_resp]
            )
            phase_c_prompts.append(format_prompt(
                dataset_name, PromptType.DELIBERATION_ACTOR, sample,
                responses=sim_ctx,
            ))
            phase_c_meta.append((i, prefix_idx))

        phase_c_results = actor_model.generate(
            phase_c_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        ) if phase_c_prompts else []

        correct_counts: dict[int, list[int]] = {
            i: [0, 0, 0] for i in active
        }
        total_counts: dict[int, list[int]] = {
            i: [0, 0, 0] for i in active
        }
        correct_norms = {
            i: normalize_answer(
                correct_answers[i],
                task_type=samples[i].get("task_type", "yes_no"),
            )
            for i in active
        }

        for (i, prefix_idx), response in zip(phase_c_meta, phase_c_results):
            task_type = samples[i].get("task_type", "yes_no")
            answer = extract_answer(response, task_type)
            if normalize_answer(answer or "", task_type=task_type) == correct_norms[i]:
                correct_counts[i][prefix_idx] += 1
            total_counts[i][prefix_idx] += 1

        for i in active:
            round_data = natural_trajectories[i][t]

            def value(prefix_idx: int) -> float:
                total = total_counts[i][prefix_idx]
                return correct_counts[i][prefix_idx] / total if total else 0.0

            v_natural = value(0)
            v_guided_correct = value(1)
            v_guided_wrong = value(2)
            delta_y = compute_reward_delta(v_guided_correct, v_natural)
            delta_not_y = compute_reward_delta(v_natural, v_guided_wrong)

            if delta_y >= reward_threshold:
                preference_pairs.append({
                    "sample": samples[i],
                    "sample_idx": sample_offset + i,
                    "actor_prompt": round_data["actor_prompt"],
                    "critic_prompt": round_data["critic_prompt"],
                    "positive": guided_actor[i]["correct"],
                    "negative": round_data["actor_response"],
                    "positive_critic": guided_critic[i]["correct"],
                    "negative_critic": round_data["critic_response"],
                    "round": t,
                    "delta": delta_y,
                    "direction": "towards",
                    "agent": "actor",
                })
            if delta_not_y >= reward_threshold:
                preference_pairs.append({
                    "sample": samples[i],
                    "sample_idx": sample_offset + i,
                    "actor_prompt": round_data["actor_prompt"],
                    "critic_prompt": round_data["critic_prompt"],
                    "positive": round_data["actor_response"],
                    "negative": guided_actor[i]["wrong"],
                    "positive_critic": round_data["critic_response"],
                    "negative_critic": guided_critic[i]["wrong"],
                    "round": t,
                    "delta": delta_not_y,
                    "direction": "away",
                    "agent": "actor",
                })

    return preference_pairs


# ============================================================
# Public API
# ============================================================

def generate_trajectories(
    actor_model,
    critic_model,
    sample: dict,
    dataset_name: str,
    num_rounds: int = 5,
    reward_threshold: float = 0.0,
    num_simulations: int = 5,
    max_tokens: int = 512,
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
    wrong_answer = generate_wrong_answer(
        correct_answer, sample.get("choices"), task_type=task_type, rng=rng,
    )

    # Run natural deliberation
    natural_trajectory = deliberate(
        actor_model, critic_model, sample, dataset_name,
        num_rounds=num_rounds, max_tokens=max_tokens, temperature=temperature,
    )

    return _generate_guided_pairs_for_sample(
        actor_model, critic_model, sample, natural_trajectory,
        dataset_name, correct_answer, wrong_answer,
        reward_threshold=reward_threshold,
        num_simulations=num_simulations,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def generate_trajectories_batch(
    actor_model,
    critic_model,
    samples: list[dict],
    dataset_name: str,
    num_rounds: int = 5,
    reward_threshold: float = 0.0,
    num_simulations: int = 5,
    max_tokens: int = 512,
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
    all_pairs: list[dict] = []

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

        correct_answers: list[str] = []
        wrong_answers: list[str] = []
        for i, sample in enumerate(batch):
            rng = random.Random(seed + batch_start + i)
            correct_answer = sample.get("answer", "")
            task_type = sample.get("task_type", "yes_no")
            correct_answers.append(correct_answer)
            wrong_answers.append(generate_wrong_answer(
                correct_answer, sample.get("choices"),
                task_type=task_type, rng=rng,
            ))

        pairs = _generate_guided_pairs_for_batch(
            actor_model, critic_model, batch, trajectories,
            dataset_name, correct_answers, wrong_answers,
            reward_threshold=reward_threshold,
            num_simulations=num_simulations,
            max_tokens=max_tokens,
            temperature=temperature,
            sample_offset=batch_start,
        )
        all_pairs.extend(pairs)

    return all_pairs


# ============================================================
# Prompt construction helper
# ============================================================

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
            responses=previous_responses,
        )
