"""
Evaluation pipeline: benchmarks, per-round accuracy, metrics.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.deliberation.engine import deliberate
from src.reward.accuracy import (
    extract_answer,
    compute_accuracy,
    compute_accuracy_with_ci,
    compute_per_round_accuracy,
    compute_improvement_rate,
)

logger = logging.getLogger(__name__)


def evaluate_benchmark(
    actor_model,
    critic_model,
    test_samples: list[dict],
    dataset_name: str,
    num_rounds: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> dict:
    """
    Evaluate Actor-Critic team on a benchmark dataset.

    Args:
        actor_model: VLLMInference for actor.
        critic_model: VLLMInference for critic.
        test_samples: List of standardized test samples.
        dataset_name: Dataset name for prompts.
        num_rounds: Deliberation rounds.
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.

    Returns:
        Dict with accuracy, per_round_accuracy, improvement_rate, ci.
    """
    labels = []
    all_round_answers = [[] for _ in range(num_rounds)]

    for i, sample in enumerate(test_samples):
        logger.info(f"Evaluating: {i+1}/{len(test_samples)}")

        trajectory = deliberate(
            actor_model, critic_model, sample, dataset_name,
            num_rounds=num_rounds, max_tokens=max_tokens,
            temperature=temperature,
        )

        task_type = sample.get("task_type", "yes_no")
        labels.append(sample.get("answer", ""))

        for t, round_data in enumerate(trajectory):
            answer = extract_answer(round_data["actor_response"], task_type)
            all_round_answers[t].append(answer or "")

    # Compute metrics
    per_round_acc = compute_per_round_accuracy(all_round_answers, labels)
    final_acc, ci_margin = compute_accuracy_with_ci(
        all_round_answers[-1], labels,
    )
    initial_acc = per_round_acc[0] if per_round_acc else 0.0
    improvement = compute_improvement_rate(final_acc, initial_acc)

    results = {
        "dataset": dataset_name,
        "num_samples": len(test_samples),
        "num_rounds": num_rounds,
        "final_accuracy": final_acc,
        "ci_margin": ci_margin,
        "initial_accuracy": initial_acc,
        "per_round_accuracy": per_round_acc,
        "improvement_rate": improvement,
    }

    logger.info(
        f"[{dataset_name}] Final acc: {final_acc:.3f} +/- {ci_margin:.3f}, "
        f"Improvement: {improvement:.3f}"
    )
    return results
