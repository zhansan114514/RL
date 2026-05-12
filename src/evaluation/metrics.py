"""Evaluation metrics for natural deliberation results."""

from __future__ import annotations

from typing import Optional

from src.evaluation.answer_resolution import (
    answers_match,
    compute_accuracy_with_ci_mixed,
    source_rates,
)
from src.evaluation.style_metrics import answer_diversity_rate, compute_style_behavior
from src.society.multi_deliberation import MultiDeliberationResult


def summarize_deliberation_results(
    results: list[MultiDeliberationResult],
    samples: list[dict],
) -> dict:
    """Summarize accuracy, parsing diagnostics, style diversity, and Critic utility."""
    labels = [sample.get("answer", "") for sample in samples]
    task_types = [sample.get("task_type", "yes_no") for sample in samples]
    num_rounds = max((len(result.rounds) for result in results), default=0)

    consensus_by_round: list[list[Optional[str]]] = [[] for _ in range(num_rounds)]
    sources_by_round: list[list[str]] = [[] for _ in range(num_rounds)]
    actor_answers_by_round: list[list[dict[str, Optional[str]]]] = [[] for _ in range(num_rounds)]
    actor_responses_by_round: list[list[dict[str, str]]] = [[] for _ in range(num_rounds)]

    for result in results:
        for round_idx, round_data in enumerate(result.rounds):
            consensus_by_round[round_idx].append(round_data.consensus_answer)
            sources_by_round[round_idx].extend(round_data.actor_answer_sources.values())
            actor_answers_by_round[round_idx].append(round_data.actor_answers)
            actor_responses_by_round[round_idx].append(round_data.actor_responses)

    per_round_consensus_accuracy = [
        compute_accuracy_with_ci_mixed(preds, labels, task_types)[0]
        for preds in consensus_by_round
    ]
    final_predictions = consensus_by_round[-1] if consensus_by_round else []
    final_acc, ci_margin = compute_accuracy_with_ci_mixed(
        final_predictions,
        labels,
        task_types,
    )
    initial_acc = per_round_consensus_accuracy[0] if per_round_consensus_accuracy else 0.0

    return {
        "consensus_accuracy": final_acc,
        "initial_consensus_accuracy": initial_acc,
        "absolute_improvement": final_acc - initial_acc,
        "per_round_consensus_accuracy": per_round_consensus_accuracy,
        "ci_margin": ci_margin,
        "ci_95": [max(0.0, final_acc - ci_margin), min(1.0, final_acc + ci_margin)],
        "actor_metrics": compute_actor_metrics(results, samples),
        "critic_metrics": compute_critic_metrics(results, samples),
        "style_diversity": {
            "answer_diversity_rate_by_round": [
                answer_diversity_rate(round_answers)
                for round_answers in actor_answers_by_round
            ],
            "style_behavior_by_round": [
                compute_style_behavior(round_responses)
                for round_responses in actor_responses_by_round
            ],
        },
        "parsing_diagnostics": {
            "actor_answer_sources_by_round": [
                source_rates(sources) for sources in sources_by_round
            ],
        },
    }


def compute_group_metrics(
    sample_details: list[dict],
    group_key: str = "subject",
) -> dict[str, dict]:
    """Compute per-group accuracy metrics from sample-level evaluation details."""
    groups: dict[str, list[dict]] = {}
    for item in sample_details:
        groups.setdefault(item.get(group_key) or "unknown", []).append(item)

    result: dict[str, dict] = {}
    for group, items in sorted(groups.items()):
        n = len(items)
        initial_correct = sum(1 for item in items if item["initially_correct"])
        final_correct = sum(1 for item in items if item["finally_correct"])
        result[group] = {
            "num_samples": n,
            "initial_accuracy": initial_correct / n if n else 0.0,
            "final_accuracy": final_correct / n if n else 0.0,
            "absolute_improvement": (
                final_correct / n - initial_correct / n
                if n else 0.0
            ),
            "flipped_to_correct": sum(1 for item in items if item["flipped_to_correct"]),
            "flipped_to_wrong": sum(1 for item in items if item["flipped_to_wrong"]),
        }
    return result


def compute_actor_metrics(
    results: list[MultiDeliberationResult],
    samples: list[dict],
) -> dict:
    """Compute per-Actor accuracy and parse success on final round."""
    by_actor_answers: dict[str, list[Optional[str]]] = {}
    by_actor_sources: dict[str, list[str]] = {}
    by_actor_labels: dict[str, list[str]] = {}
    by_actor_task_types: dict[str, list[str]] = {}
    for result, sample in zip(results, samples):
        if not result.rounds:
            continue
        final_round = result.rounds[-1]
        for actor_name, answer in final_round.actor_answers.items():
            by_actor_answers.setdefault(actor_name, []).append(answer)
            by_actor_sources.setdefault(actor_name, []).append(
                final_round.actor_answer_sources.get(actor_name, "none")
            )
            by_actor_labels.setdefault(actor_name, []).append(sample.get("answer", ""))
            by_actor_task_types.setdefault(actor_name, []).append(
                sample.get("task_type", "yes_no")
            )

    per_actor_accuracy = {}
    per_actor_parse_success = {}
    for actor_name, answers in by_actor_answers.items():
        per_actor_accuracy[actor_name] = compute_accuracy_with_ci_mixed(
            answers,
            by_actor_labels.get(actor_name, []),
            by_actor_task_types.get(actor_name, []),
        )[0]
        sources = by_actor_sources.get(actor_name, [])
        per_actor_parse_success[actor_name] = (
            sum(1 for source in sources if source != "none") / len(sources)
            if sources else 0.0
        )
    return {
        "per_actor_accuracy": per_actor_accuracy,
        "per_actor_parse_success": per_actor_parse_success,
    }


def compute_critic_metrics(
    results: list[MultiDeliberationResult],
    samples: list[dict],
) -> dict:
    """Compute Critic parse and usefulness diagnostics."""
    total = 0
    confidence = 0
    suggested = 0
    answer_correct = 0
    by_critic: dict[str, dict[str, float]] = {}

    for result_idx, result in enumerate(results):
        sample = samples[result_idx] if result_idx < len(samples) else {}
        for round_data in result.rounds:
            selected = {
                fb.critic_name
                for selected_list in round_data.routed_feedbacks.values()
                for fb in selected_list
            }
            for actor_feedbacks in round_data.critic_feedbacks.values():
                for fb in actor_feedbacks.values():
                    total += 1
                    confidence += int(fb.confidence is not None)
                    suggested += int(fb.suggested_answer is not None)
                    answer_correct += int(fb.answer_correct != "unknown")
                    stats = by_critic.setdefault(
                        fb.critic_name,
                        {
                            "total": 0.0,
                            "selected": 0.0,
                            "suggested_answer_count": 0.0,
                            "suggested_answer_correct": 0.0,
                            "confidence_count": 0.0,
                        },
                    )
                    stats["total"] += 1
                    stats["selected"] += float(fb.critic_name in selected)
                    stats["suggested_answer_count"] += float(fb.suggested_answer is not None)
                    if fb.suggested_answer is not None:
                        stats["suggested_answer_correct"] += float(
                            answers_match(
                                fb.suggested_answer,
                                sample.get("answer", ""),
                                sample.get("task_type", "yes_no"),
                            )
                        )
                    stats["confidence_count"] += float(fb.confidence is not None)

    helpfulness = {}
    for critic_name, stats in by_critic.items():
        denom = stats["total"] or 1.0
        helpfulness[critic_name] = {
            "selected_rate": stats["selected"] / denom,
            "suggested_answer_parse_rate": stats["suggested_answer_count"] / denom,
            "suggested_answer_accuracy": (
                stats["suggested_answer_correct"] / stats["suggested_answer_count"]
                if stats["suggested_answer_count"] else 0.0
            ),
            "confidence_parse_rate": stats["confidence_count"] / denom,
        }

    denom = total or 1
    return {
        "critic_parse_success": {
            "confidence_parse_rate": confidence / denom,
            "suggested_answer_parse_rate": suggested / denom,
            "answer_correct_parse_rate": answer_correct / denom,
        },
        "critic_helpfulness": helpfulness,
    }
