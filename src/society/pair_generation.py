"""Society-native pairwise deliberation and guided rollout data generation."""

from __future__ import annotations

import logging
import random
from typing import Any

from src.algorithms.reward import compute_reward_delta, extract_answer, normalize_answer
from src.data.preprocessor import generate_wrong_answer
from src.prompts.prompt_builder import (
    build_guided_actor_prompt,
    build_guided_critic_prompt,
    build_simple_actor_prompt,
    build_simple_critic_prompt,
)

logger = logging.getLogger(__name__)


def run_pairwise_deliberation(
    actor_model: Any,
    critic_model: Any,
    sample: dict,
    dataset_name: str,
    num_rounds: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> list[dict]:
    """Run one natural Actor/Critic dialogue used as a pairwise training seed."""
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


def run_pairwise_deliberation_batch(
    actor_model: Any,
    critic_model: Any,
    samples: list[dict],
    dataset_name: str,
    num_rounds: int = 5,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> list[list[dict]]:
    """Run batched natural Actor/Critic dialogues for Society DPO data."""
    if not samples:
        return []
    if len(samples) == 1:
        return [
            run_pairwise_deliberation(
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


def _guided_prompt(
    dataset_name: str,
    sample: dict,
    target_answer: str,
    round_idx: int,
    previous_responses: list[str],
    actor_response: str,
    agent: str,
) -> str:
    previous_actor = actor_response or (
        previous_responses[-2] if len(previous_responses) >= 2 else ""
    )
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
    if agent == "critic":
        return build_guided_critic_prompt(
            sample,
            dataset_name,
            actor_response=actor_response,
            target_answer=target_answer,
        )
    raise ValueError(f"Unknown guided prompt agent: {agent}")


def build_guided_rollout_pairs(
    actor_model: Any,
    critic_model: Any,
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
    """Build chosen/rejected pairs from natural seeds and guided rollout scoring."""
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
            for direction, target in (
                ("correct", correct_answers[idx]),
                ("wrong", wrong_answers[idx]),
            ):
                guided_actor_prompts.append(
                    _guided_prompt(
                        dataset_name,
                        sample,
                        target,
                        round_idx,
                        previous_by_sample[idx],
                        round_data["actor_response"],
                        agent="actor",
                    )
                )
                guided_actor_meta.append((idx, direction))

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
            for direction, target in (
                ("correct", correct_answers[idx]),
                ("wrong", wrong_answers[idx]),
            ):
                guided_critic_prompts.append(
                    _guided_prompt(
                        dataset_name,
                        sample,
                        target,
                        round_idx,
                        previous_by_sample[idx],
                        guided_actor[idx][direction],
                        agent="critic",
                    )
                )
                guided_critic_meta.append((idx, direction))

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
                    "sample_index": sample_offset + idx,
                    "round": round_idx,
                    "natural_seed": {
                        "actor_prompt": round_data["actor_prompt"],
                        "actor_response": round_data["actor_response"],
                        "critic_prompt": round_data["critic_prompt"],
                        "critic_response": round_data["critic_response"],
                    },
                    "guided": {
                        "correct": {
                            "target_answer": correct_answers[idx],
                            "actor_response": guided_actor[idx]["correct"],
                            "critic_response": guided_critic[idx]["correct"],
                        },
                        "wrong": {
                            "target_answer": wrong_answers[idx],
                            "actor_response": guided_actor[idx]["wrong"],
                            "critic_response": guided_critic[idx]["wrong"],
                        },
                    },
                    "rollout_scores": {
                        "natural": v_natural,
                        "guided_correct": v_guided_correct,
                        "guided_wrong": v_guided_wrong,
                    },
                    "comparison": {
                        "mode": "towards_correct",
                        "delta": delta_y,
                        "reward_threshold": reward_threshold,
                    },
                    "actor_candidate": {
                        "chosen": guided_actor[idx]["correct"],
                        "rejected": round_data["actor_response"],
                    },
                    "critic_candidate": {
                        "chosen": guided_critic[idx]["correct"],
                        "rejected": round_data["critic_response"],
                    },
                })
            if delta_not_y >= reward_threshold:
                preference_pairs.append({
                    "sample": samples[idx],
                    "sample_index": sample_offset + idx,
                    "round": round_idx,
                    "natural_seed": {
                        "actor_prompt": round_data["actor_prompt"],
                        "actor_response": round_data["actor_response"],
                        "critic_prompt": round_data["critic_prompt"],
                        "critic_response": round_data["critic_response"],
                    },
                    "guided": {
                        "correct": {
                            "target_answer": correct_answers[idx],
                            "actor_response": guided_actor[idx]["correct"],
                            "critic_response": guided_critic[idx]["correct"],
                        },
                        "wrong": {
                            "target_answer": wrong_answers[idx],
                            "actor_response": guided_actor[idx]["wrong"],
                            "critic_response": guided_critic[idx]["wrong"],
                        },
                    },
                    "rollout_scores": {
                        "natural": v_natural,
                        "guided_correct": v_guided_correct,
                        "guided_wrong": v_guided_wrong,
                    },
                    "comparison": {
                        "mode": "away_from_wrong",
                        "delta": delta_not_y,
                        "reward_threshold": reward_threshold,
                    },
                    "actor_candidate": {
                        "chosen": round_data["actor_response"],
                        "rejected": guided_actor[idx]["wrong"],
                    },
                    "critic_candidate": {
                        "chosen": round_data["critic_response"],
                        "rejected": guided_critic[idx]["wrong"],
                    },
                })

    return preference_pairs


def build_pairwise_training_pairs_batch(
    actor_model: Any,
    critic_model: Any,
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
    """Build Society pairwise training pairs directly from samples."""
    all_pairs: list[dict] = []
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start:batch_start + batch_size]
        logger.info(
            "Pairwise training data: samples %s-%s/%s",
            batch_start + 1,
            batch_start + len(batch),
            len(samples),
        )
        trajectories = run_pairwise_deliberation_batch(
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
            build_guided_rollout_pairs(
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
