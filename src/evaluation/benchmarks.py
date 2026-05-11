"""Evaluation pipeline for natural deliberation."""

from __future__ import annotations

import logging
import time

from src.algorithms.deliberation import deliberate_batch
from src.evaluation.answer_resolution import (
    answers_match,
    compute_accuracy_with_ci_mixed,
    resolve_answer_for_round,
    source_rates,
)
from src.evaluation.style_metrics import compute_style_behavior

logger = logging.getLogger(__name__)


def evaluate_benchmark(
    actor_model,
    critic_model,
    test_samples: list[dict],
    dataset_name: str,
    num_rounds: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> dict:
    """Evaluate a single Actor/Critic team on a benchmark dataset."""
    return evaluate_benchmark_batch(
        actor_model,
        critic_model,
        test_samples,
        dataset_name,
        num_rounds=num_rounds,
        max_tokens=max_tokens,
        temperature=temperature,
        batch_size=batch_size,
    )


def evaluate_benchmark_batch(
    actor_model,
    critic_model,
    test_samples: list[dict],
    dataset_name: str,
    num_rounds: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
    batch_size: int = 8,
) -> dict:
    """Evaluate natural Actor/Critic deliberation with batched inference."""
    labels: list[str] = []
    task_types: list[str] = []
    all_round_answers = [[] for _ in range(num_rounds)]
    all_round_sources = [[] for _ in range(num_rounds)]
    all_round_parse_confidence = [[] for _ in range(num_rounds)]
    all_round_responses = [[] for _ in range(num_rounds)]
    sample_details = []
    eval_start = time.time()

    for batch_start in range(0, len(test_samples), batch_size):
        batch = test_samples[batch_start:batch_start + batch_size]
        logger.info(
            "Evaluating batch: %s-%s/%s",
            batch_start + 1,
            batch_start + len(batch),
            len(test_samples),
        )
        t0 = time.time()
        trajectories = deliberate_batch(
            actor_model,
            critic_model,
            batch,
            dataset_name,
            num_rounds=num_rounds,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        per_sample_elapsed = (time.time() - t0) / max(len(batch), 1)

        for local_i, (sample, trajectory) in enumerate(zip(batch, trajectories)):
            sample_idx = batch_start + local_i
            task_type = sample.get("task_type", "yes_no")
            label = sample.get("answer", "")
            labels.append(label)
            task_types.append(task_type)

            round_answers = []
            round_sources = []
            round_confidences = []
            for round_idx, round_data in enumerate(trajectory):
                response = round_data["actor_response"]
                resolved = resolve_answer_for_round(
                    response=response,
                    task_type=task_type,
                )
                all_round_answers[round_idx].append(resolved.resolved_answer)
                all_round_sources[round_idx].append(resolved.extract_source)
                all_round_parse_confidence[round_idx].append(resolved.parse_confidence)
                all_round_responses[round_idx].append({"actor": response})
                round_answers.append(resolved.resolved_answer)
                round_sources.append(resolved.extract_source)
                round_confidences.append(resolved.parse_confidence)

            initial_pred = round_answers[0] if round_answers else None
            final_pred = round_answers[-1] if round_answers else None
            initially_correct = answers_match(initial_pred, label, task_type)
            finally_correct = answers_match(final_pred, label, task_type)

            sample_details.append({
                "index": sample_idx,
                "dataset": sample.get("dataset", dataset_name),
                "source_split": sample.get("source_split", ""),
                "source_index": sample.get("source_index", None),
                "subject": sample.get("subject", "unknown"),
                "category": sample.get("category", "unknown"),
                "initial_answer": initial_pred,
                "final_answer": final_pred,
                "initial_extract_source": round_sources[0] if round_sources else "none",
                "final_extract_source": round_sources[-1] if round_sources else "none",
                "round_answers": round_answers,
                "round_answer_sources": round_sources,
                "round_parse_confidence": round_confidences,
                "label": label,
                "initially_correct": initially_correct,
                "finally_correct": finally_correct,
                "flipped_to_correct": not initially_correct and finally_correct,
                "flipped_to_wrong": initially_correct and not finally_correct,
                "elapsed_seconds": round(per_sample_elapsed, 2),
                "task_type": task_type,
            })

    eval_elapsed = time.time() - eval_start

    per_round_sources = [source_rates(sources) for sources in all_round_sources]
    per_round_acc = [
        compute_accuracy_with_ci_mixed(round_answers, labels, task_types)[0]
        for round_answers in all_round_answers
    ]
    final_acc, ci_margin = compute_accuracy_with_ci_mixed(
        all_round_answers[-1] if all_round_answers else [],
        labels,
        task_types,
    )
    initial_acc = per_round_acc[0] if per_round_acc else 0.0
    round_deltas = [
        per_round_acc[idx] - per_round_acc[idx - 1]
        for idx in range(1, len(per_round_acc))
    ]

    n_flipped_to_correct = sum(1 for item in sample_details if item["flipped_to_correct"])
    n_flipped_to_wrong = sum(1 for item in sample_details if item["flipped_to_wrong"])
    n_stayed_correct = sum(
        1 for item in sample_details
        if item["initially_correct"] and item["finally_correct"]
    )
    n_stayed_wrong = sum(
        1 for item in sample_details
        if not item["initially_correct"] and not item["finally_correct"]
    )

    final_source_rates = per_round_sources[-1] if per_round_sources else source_rates([])
    results = {
        "dataset": dataset_name,
        "num_samples": len(test_samples),
        "num_rounds": num_rounds,
        "initial_consensus_accuracy": initial_acc,
        "consensus_accuracy": final_acc,
        "initial_accuracy": initial_acc,
        "final_accuracy": final_acc,
        "ci_margin": ci_margin,
        "ci_95": [max(0, final_acc - ci_margin), min(1, final_acc + ci_margin)],
        "absolute_improvement": final_acc - initial_acc,
        "improvement_rate": (final_acc - initial_acc) / initial_acc if initial_acc else 0.0,
        "per_round_consensus_accuracy": per_round_acc,
        "per_round_accuracy": per_round_acc,
        "per_round_delta": round_deltas,
        "parse_success_rate": final_source_rates["parse_success_rate"],
        "actor_unresolved_rate": final_source_rates["none_rate"],
        "actor_answer_sources": final_source_rates,
        "parsing_diagnostics": {
            "actor_answer_sources_by_round": {
                str(idx): per_round_sources[idx]
                for idx in range(len(per_round_sources))
            },
            "avg_parse_confidence_by_round": [
                sum(values) / len(values) if values else 0.0
                for values in all_round_parse_confidence
            ],
        },
        "style_diversity": {
            "style_behavior_by_round": [
                compute_style_behavior(responses)
                for responses in all_round_responses
            ],
        },
        "flip_statistics": {
            "flipped_to_correct": n_flipped_to_correct,
            "flipped_to_wrong": n_flipped_to_wrong,
            "stayed_correct": n_stayed_correct,
            "stayed_wrong": n_stayed_wrong,
        },
        "correct_at_each_round": [
            sum(
                1 for idx, detail in enumerate(sample_details)
                if answers_match(
                    all_round_answers[round_idx][idx],
                    labels[idx],
                    detail.get("task_type", "yes_no"),
                )
            )
            for round_idx in range(num_rounds)
        ],
        "eval_time_seconds": round(eval_elapsed, 1),
        "avg_time_per_sample": round(eval_elapsed / max(len(test_samples), 1), 2),
        "sample_details": sample_details,
        "per_subject_metrics": compute_group_metrics(sample_details, "subject"),
    }
    _print_results_table(results)
    return results


def compute_group_metrics(
    sample_details: list[dict],
    group_key: str = "subject",
) -> dict[str, dict]:
    """Compute per-group accuracy metrics from sample details."""
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


def _print_results_table(results: dict) -> None:
    """Print a concise benchmark table."""
    sep = "=" * 65
    thin = "-" * 65
    logger.info(sep)
    logger.info("  BENCHMARK EVALUATION RESULTS")
    logger.info(sep)
    logger.info("  Dataset:             %s", results["dataset"])
    logger.info("  Test samples:        %s", results["num_samples"])
    logger.info("  Deliberation rounds: %s", results["num_rounds"])
    logger.info(
        "  Eval time:           %ss (%ss/sample)",
        results["eval_time_seconds"],
        results["avg_time_per_sample"],
    )
    logger.info(thin)
    logger.info("  ACCURACY")
    logger.info("    Initial consensus: %.4f", results["initial_consensus_accuracy"])
    logger.info("    Final consensus:   %.4f", results["consensus_accuracy"])
    logger.info("    Absolute gain:     %+0.4f", results["absolute_improvement"])
    logger.info("    95%% CI:            [%.4f, %.4f]", *results["ci_95"])
    logger.info("  PARSING")
    logger.info("    Actor parse success: %.4f", results["parse_success_rate"])
    logger.info("    Actor unresolved:    %.4f", results["actor_unresolved_rate"])
    logger.info(thin)
    logger.info("  PER-ROUND CONSENSUS ACCURACY")
    for idx, acc in enumerate(results["per_round_consensus_accuracy"]):
        delta = results["per_round_delta"][idx - 1] if idx > 0 else None
        logger.info(
            "    Round %s: %.4f%s",
            idx + 1,
            acc,
            "" if delta is None else f" ({delta:+.4f})",
        )
    logger.info(sep)
