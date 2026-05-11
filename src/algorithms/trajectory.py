"""Natural trajectory generation and preference-pair construction."""

from __future__ import annotations

import logging
import random

from src.algorithms.deliberation import deliberate, deliberate_batch
from src.algorithms.reward import compute_reward_delta, extract_answer, normalize_answer
from src.data.preprocessor import generate_wrong_answer
from src.prompts.prompt_builder import (
    build_guided_actor_prompt,
    build_guided_critic_prompt,
    build_simple_actor_prompt,
    build_simple_critic_prompt,
)

logger = logging.getLogger(__name__)


def _revision_prompt_from_pair(
    sample: dict,
    dataset_name: str,
    actor_response: str,
    critic_response: str,
) -> str:
    return build_simple_actor_prompt(
        sample,
        dataset_name,
        round_num=1,
        previous_actor_response=actor_response,
        critic_feedback=critic_response,
    )


def _make_guided_prompt(
    dataset_name: str,
    sample: dict,
    target_answer: str,
    round_idx: int,
    previous_responses: list[str],
    actor_response: str,
    agent: str = "actor",
) -> str:
    """Build a guided natural prompt for trajectory rollouts."""
    previous_actor = actor_response or (previous_responses[-2] if len(previous_responses) >= 2 else "")
    previous_feedback = previous_responses[-1] if previous_responses else ""
    if agent == "actor":
        return build_guided_actor_prompt(
            sample,
            dataset_name,
            target_answer=target_answer,
            round_num=round_idx,
            previous_actor_response=previous_actor,
            critic_feedback=previous_feedback,
        )
    return build_guided_critic_prompt(
        sample,
        dataset_name,
        actor_response=actor_response,
        target_answer=target_answer,
    )


def _count_correct(
    responses: list[str],
    correct_answer: str,
    task_type: str,
) -> float:
    if not responses:
        return 0.0
    correct_norm = normalize_answer(correct_answer, task_type=task_type)
    count = 0
    for response in responses:
        answer = extract_answer(response, task_type)
        if normalize_answer(answer or "", task_type=task_type) == correct_norm:
            count += 1
    return count / len(responses)


def _rollout_values(
    actor_model,
    critic_model,
    sample: dict,
    dataset_name: str,
    prefix_pairs: list[tuple[str, str]],
    correct_answer: str,
    num_simulations: int,
    max_tokens: int,
    temperature: float,
) -> list[float]:
    """Estimate value for each actor/critic prefix pair using natural rollouts."""
    phase_a_prompts = []
    groups: list[tuple[int, int]] = []
    for actor_response, critic_response in prefix_pairs:
        start = len(phase_a_prompts)
        for _ in range(num_simulations):
            phase_a_prompts.append(
                _revision_prompt_from_pair(
                    sample,
                    dataset_name,
                    actor_response,
                    critic_response,
                )
            )
        groups.append((start, num_simulations))

    phase_a_results = actor_model.generate(
        phase_a_prompts,
        max_tokens=max_tokens,
        temperature=temperature,
    ) if phase_a_prompts else []

    phase_b_prompts = [
        build_simple_critic_prompt(sample, dataset_name, actor_response=response)
        for response in phase_a_results
    ]
    phase_b_results = critic_model.generate(
        phase_b_prompts,
        max_tokens=max_tokens,
        temperature=temperature,
    ) if phase_b_prompts else []

    phase_c_prompts = [
        _revision_prompt_from_pair(sample, dataset_name, actor_response, critic_response)
        for actor_response, critic_response in zip(phase_a_results, phase_b_results)
    ]
    phase_c_results = actor_model.generate(
        phase_c_prompts,
        max_tokens=max_tokens,
        temperature=temperature,
    ) if phase_c_prompts else []

    task_type = sample.get("task_type", "yes_no")
    values = []
    for start, count in groups:
        values.append(
            _count_correct(
                phase_c_results[start:start + count],
                correct_answer,
                task_type,
            )
        )
    return values


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
    """Generate guided preference pairs for a single natural trajectory."""
    preference_pairs: list[dict] = []
    for round_idx in range(1, len(natural_trajectory)):
        round_data = natural_trajectory[round_idx]
        actor_response = round_data["actor_response"]
        critic_response = round_data["critic_response"]

        previous_responses: list[str] = []
        for previous in natural_trajectory[:round_idx]:
            previous_responses.extend([previous["actor_response"], previous["critic_response"]])

        z_y_actor_prompt = _make_guided_prompt(
            dataset_name,
            sample,
            correct_answer,
            round_idx,
            previous_responses,
            actor_response,
            agent="actor",
        )
        z_not_y_actor_prompt = _make_guided_prompt(
            dataset_name,
            sample,
            wrong_answer,
            round_idx,
            previous_responses,
            actor_response,
            agent="actor",
        )
        z_y_actor, z_not_y_actor = actor_model.generate(
            [z_y_actor_prompt, z_not_y_actor_prompt],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        z_y_critic_prompt = _make_guided_prompt(
            dataset_name,
            sample,
            correct_answer,
            round_idx,
            previous_responses,
            z_y_actor,
            agent="critic",
        )
        z_not_y_critic_prompt = _make_guided_prompt(
            dataset_name,
            sample,
            wrong_answer,
            round_idx,
            previous_responses,
            z_not_y_actor,
            agent="critic",
        )
        z_y_critic, z_not_y_critic = critic_model.generate(
            [z_y_critic_prompt, z_not_y_critic_prompt],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        values = _rollout_values(
            actor_model,
            critic_model,
            sample,
            dataset_name,
            [
                (actor_response, critic_response),
                (z_y_actor, z_y_critic),
                (z_not_y_actor, z_not_y_critic),
            ],
            correct_answer,
            num_simulations,
            max_tokens,
            temperature,
        )
        if len(values) != 3:
            continue
        v_natural, v_guided_correct, v_guided_wrong = values
        delta_y = compute_reward_delta(v_guided_correct, v_natural)
        delta_not_y = compute_reward_delta(v_natural, v_guided_wrong)

        if delta_y >= reward_threshold:
            preference_pairs.append({
                "sample": sample,
                "actor_prompt": round_data["actor_prompt"],
                "critic_prompt": round_data["critic_prompt"],
                "positive": z_y_actor,
                "negative": actor_response,
                "positive_critic": z_y_critic,
                "negative_critic": critic_response,
                "round": round_idx,
                "delta": delta_y,
                "direction": "towards",
                "agent": "actor",
            })
        if delta_not_y >= reward_threshold:
            preference_pairs.append({
                "sample": sample,
                "actor_prompt": round_data["actor_prompt"],
                "critic_prompt": round_data["critic_prompt"],
                "positive": actor_response,
                "negative": z_not_y_actor,
                "positive_critic": critic_response,
                "negative_critic": z_not_y_critic,
                "round": round_idx,
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
    """Generate guided pairs for a batch using natural prompts."""
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

    for round_idx in range(1, max_rounds):
        active = [
            idx for idx, trajectory in enumerate(natural_trajectories)
            if round_idx < len(trajectory)
        ]
        if not active:
            continue

        previous_by_sample: dict[int, list[str]] = {}
        for idx in active:
            previous: list[str] = []
            for item in natural_trajectories[idx][:round_idx]:
                previous.extend([item["actor_response"], item["critic_response"]])
            previous_by_sample[idx] = previous

        guided_actor_prompts: list[str] = []
        guided_actor_meta: list[tuple[int, str]] = []
        for idx in active:
            sample = samples[idx]
            round_data = natural_trajectories[idx][round_idx]
            guided_actor_prompts.append(
                _make_guided_prompt(
                    dataset_name,
                    sample,
                    correct_answers[idx],
                    round_idx,
                    previous_by_sample[idx],
                    round_data["actor_response"],
                    agent="actor",
                )
            )
            guided_actor_meta.append((idx, "correct"))
            guided_actor_prompts.append(
                _make_guided_prompt(
                    dataset_name,
                    sample,
                    wrong_answers[idx],
                    round_idx,
                    previous_by_sample[idx],
                    round_data["actor_response"],
                    agent="actor",
                )
            )
            guided_actor_meta.append((idx, "wrong"))

        guided_actor_results = actor_model.generate(
            guided_actor_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        guided_actor: dict[int, dict[str, str]] = {idx: {} for idx in active}
        for (idx, direction), response in zip(guided_actor_meta, guided_actor_results):
            guided_actor[idx][direction] = response

        guided_critic_prompts: list[str] = []
        guided_critic_meta: list[tuple[int, str]] = []
        for idx in active:
            sample = samples[idx]
            guided_critic_prompts.append(
                _make_guided_prompt(
                    dataset_name,
                    sample,
                    correct_answers[idx],
                    round_idx,
                    previous_by_sample[idx],
                    guided_actor[idx]["correct"],
                    agent="critic",
                )
            )
            guided_critic_meta.append((idx, "correct"))
            guided_critic_prompts.append(
                _make_guided_prompt(
                    dataset_name,
                    sample,
                    wrong_answers[idx],
                    round_idx,
                    previous_by_sample[idx],
                    guided_actor[idx]["wrong"],
                    agent="critic",
                )
            )
            guided_critic_meta.append((idx, "wrong"))

        guided_critic_results = critic_model.generate(
            guided_critic_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        guided_critic: dict[int, dict[str, str]] = {idx: {} for idx in active}
        for (idx, direction), response in zip(guided_critic_meta, guided_critic_results):
            guided_critic[idx][direction] = response

        phase_a_prompts: list[str] = []
        phase_a_meta: list[tuple[int, int]] = []
        for idx in active:
            sample = samples[idx]
            round_data = natural_trajectories[idx][round_idx]
            prefix_pairs = [
                (round_data["actor_response"], round_data["critic_response"]),
                (guided_actor[idx]["correct"], guided_critic[idx]["correct"]),
                (guided_actor[idx]["wrong"], guided_critic[idx]["wrong"]),
            ]
            for prefix_idx, (actor_response, critic_response) in enumerate(prefix_pairs):
                for _ in range(num_simulations):
                    phase_a_prompts.append(
                        _revision_prompt_from_pair(
                            sample,
                            dataset_name,
                            actor_response,
                            critic_response,
                        )
                    )
                    phase_a_meta.append((idx, prefix_idx))

        phase_a_results = actor_model.generate(
            phase_a_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        ) if phase_a_prompts else []

        phase_b_prompts = [
            build_simple_critic_prompt(samples[idx], dataset_name, actor_response=response)
            for (idx, _), response in zip(phase_a_meta, phase_a_results)
        ]
        phase_b_results = critic_model.generate(
            phase_b_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        ) if phase_b_prompts else []

        phase_c_prompts = [
            _revision_prompt_from_pair(samples[idx], dataset_name, actor_response, critic_response)
            for (idx, _), actor_response, critic_response
            in zip(phase_a_meta, phase_a_results, phase_b_results)
        ]
        phase_c_meta = phase_a_meta[:len(phase_c_prompts)]
        phase_c_results = actor_model.generate(
            phase_c_prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        ) if phase_c_prompts else []

        correct_counts: dict[int, list[int]] = {idx: [0, 0, 0] for idx in active}
        total_counts: dict[int, list[int]] = {idx: [0, 0, 0] for idx in active}
        for (idx, prefix_idx), response in zip(phase_c_meta, phase_c_results):
            task_type = samples[idx].get("task_type", "yes_no")
            answer = extract_answer(response, task_type)
            if (
                normalize_answer(answer or "", task_type=task_type)
                == normalize_answer(correct_answers[idx], task_type=task_type)
            ):
                correct_counts[idx][prefix_idx] += 1
            total_counts[idx][prefix_idx] += 1

        for idx in active:
            round_data = natural_trajectories[idx][round_idx]

            def value(prefix_idx: int) -> float:
                total = total_counts[idx][prefix_idx]
                return correct_counts[idx][prefix_idx] / total if total else 0.0

            v_natural = value(0)
            v_guided_correct = value(1)
            v_guided_wrong = value(2)
            delta_y = compute_reward_delta(v_guided_correct, v_natural)
            delta_not_y = compute_reward_delta(v_natural, v_guided_wrong)

            if delta_y >= reward_threshold:
                preference_pairs.append({
                    "sample": samples[idx],
                    "sample_idx": sample_offset + idx,
                    "actor_prompt": round_data["actor_prompt"],
                    "critic_prompt": round_data["critic_prompt"],
                    "positive": guided_actor[idx]["correct"],
                    "negative": round_data["actor_response"],
                    "positive_critic": guided_critic[idx]["correct"],
                    "negative_critic": round_data["critic_response"],
                    "round": round_idx,
                    "delta": delta_y,
                    "direction": "towards",
                    "agent": "actor",
                })
            if delta_not_y >= reward_threshold:
                preference_pairs.append({
                    "sample": samples[idx],
                    "sample_idx": sample_offset + idx,
                    "actor_prompt": round_data["actor_prompt"],
                    "critic_prompt": round_data["critic_prompt"],
                    "positive": round_data["actor_response"],
                    "negative": guided_actor[idx]["wrong"],
                    "positive_critic": round_data["critic_response"],
                    "negative_critic": guided_critic[idx]["wrong"],
                    "round": round_idx,
                    "delta": delta_not_y,
                    "direction": "away",
                    "agent": "actor",
                })

    return preference_pairs


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
    """Generate preference pairs for one sample."""
    rng = random.Random(seed)
    task_type = sample.get("task_type", "yes_no")
    correct_answer = sample.get("answer", "")
    wrong_answer = generate_wrong_answer(
        correct_answer,
        sample.get("choices"),
        task_type=task_type,
        rng=rng,
    )
    natural_trajectory = deliberate(
        actor_model,
        critic_model,
        sample,
        dataset_name,
        num_rounds=num_rounds,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _generate_guided_pairs_for_sample(
        actor_model,
        critic_model,
        sample,
        natural_trajectory,
        dataset_name,
        correct_answer,
        wrong_answer,
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
    """Generate preference pairs for a dataset batch."""
    all_pairs: list[dict] = []
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start:batch_start + batch_size]
        logger.info(
            "Batched trajectories: samples %s-%s/%s",
            batch_start + 1,
            batch_start + len(batch),
            len(samples),
        )
        trajectories = deliberate_batch(
            actor_model,
            critic_model,
            batch,
            dataset_name,
            num_rounds=num_rounds,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        correct_answers: list[str] = []
        wrong_answers: list[str] = []
        for idx, sample in enumerate(batch):
            rng = random.Random(seed + batch_start + idx)
            correct_answer = sample.get("answer", "")
            task_type = sample.get("task_type", "yes_no")
            correct_answers.append(correct_answer)
            wrong_answers.append(
                generate_wrong_answer(
                    correct_answer,
                    sample.get("choices"),
                    task_type=task_type,
                    rng=rng,
                )
            )

        all_pairs.extend(
            _generate_guided_pairs_for_batch(
                actor_model,
                critic_model,
                batch,
                trajectories,
                dataset_name,
                correct_answers,
                wrong_answers,
                reward_threshold=reward_threshold,
                num_simulations=num_simulations,
                max_tokens=max_tokens,
                temperature=temperature,
                sample_offset=batch_start,
            )
        )
    return all_pairs
