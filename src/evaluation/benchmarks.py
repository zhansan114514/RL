"""
Evaluation pipeline: benchmarks, per-round accuracy, metrics.
"""

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
        Dict with stateful accuracy, per-round accuracy, format metrics, and CI.
    """
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
    """Evaluate Actor-Critic team with batched deliberation."""
    labels = []
    task_types = []
    all_round_answers = [[] for _ in range(num_rounds)]
    all_round_raw_answers = [[] for _ in range(num_rounds)]
    all_round_sources = [[] for _ in range(num_rounds)]
    all_round_format_valid = [[] for _ in range(num_rounds)]
    all_round_responses = [[] for _ in range(num_rounds)]
    sample_details = []
    eval_start = time.time()

    for batch_start in range(0, len(test_samples), batch_size):
        batch = test_samples[batch_start:batch_start + batch_size]
        t0 = time.time()
        logger.info(
            f"Evaluating batch: {batch_start + 1}-"
            f"{batch_start + len(batch)}/{len(test_samples)}"
        )

        trajectories = deliberate_batch(
            actor_model, critic_model, batch, dataset_name,
            num_rounds=num_rounds,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        batch_elapsed = time.time() - t0
        per_sample_elapsed = batch_elapsed / max(len(batch), 1)

        for local_i, (sample, trajectory) in enumerate(zip(batch, trajectories)):
            i = batch_start + local_i
            task_type = sample.get("task_type", "yes_no")
            label = sample.get("answer", "")
            labels.append(label)
            task_types.append(task_type)

            round_answers = []
            raw_round_answers = []
            round_answer_sources = []
            strict_format_ok = []
            previous_answer = None
            for t, round_data in enumerate(trajectory):
                response = round_data["actor_response"]
                resolved = resolve_answer_for_round(
                    response=response,
                    task_type=task_type,
                    previous_answer=previous_answer,
                )
                answer = resolved.resolved_answer
                previous_answer = answer
                all_round_answers[t].append(answer)
                all_round_raw_answers[t].append(resolved.raw_extracted_answer)
                all_round_sources[t].append(resolved.extract_source)
                all_round_format_valid[t].append(resolved.format_valid)
                all_round_responses[t].append(response)
                round_answers.append(answer)
                raw_round_answers.append(resolved.raw_extracted_answer)
                round_answer_sources.append(resolved.extract_source)
                strict_format_ok.append(resolved.format_valid)

            initial_pred = round_answers[0] if round_answers else None
            final_pred = round_answers[-1] if round_answers else None
            initially_correct = answers_match(initial_pred, label, task_type)
            finally_correct = answers_match(final_pred, label, task_type)

            sample_details.append({
                "index": i,
                "dataset": sample.get("dataset", dataset_name),
                "source_split": sample.get("source_split", ""),
                "source_index": sample.get("source_index", None),
                "subject": sample.get("subject", "unknown"),
                "category": sample.get("category", "unknown"),
                "initial_answer": initial_pred,
                "final_answer": final_pred,
                "initial_extract_source": (
                    round_answer_sources[0] if round_answer_sources else "none"
                ),
                "final_extract_source": (
                    round_answer_sources[-1] if round_answer_sources else "none"
                ),
                "round_answers": round_answers,
                "raw_round_answers": raw_round_answers,
                "round_answer_sources": round_answer_sources,
                "strict_format_ok": strict_format_ok,
                "label": label,
                "initially_correct": initially_correct,
                "finally_correct": finally_correct,
                "flipped_to_correct": not initially_correct and finally_correct,
                "flipped_to_wrong": initially_correct and not finally_correct,
                "elapsed_seconds": round(per_sample_elapsed, 2),
                "task_type": task_type,
            })

    eval_elapsed = time.time() - eval_start

    # Compute metrics
    per_round_answer_sources = [
        source_rates(round_sources)
        for round_sources in all_round_sources
    ]
    final_answer_sources = (
        per_round_answer_sources[-1]
        if per_round_answer_sources
        else source_rates([])
    )
    per_round_acc = [
        compute_accuracy_with_ci_mixed(round_answers, labels, task_types)[0]
        for round_answers in all_round_answers
    ]
    final_acc, ci_margin = compute_accuracy_with_ci_mixed(
        all_round_answers[-1] if all_round_answers else [],
        labels,
        task_types,
    )
    final_round_answers = all_round_answers[-1] if all_round_answers else []
    final_round_sources = all_round_sources[-1] if all_round_sources else []
    strict_final_answers = [
        answer if source == "strict" else None
        for answer, source in zip(final_round_answers, final_round_sources)
    ]
    strict_format_acc, _ = compute_accuracy_with_ci_mixed(
        strict_final_answers,
        labels,
        task_types,
    )
    initial_acc = per_round_acc[0] if per_round_acc else 0.0
    improvement = (
        (final_acc - initial_acc) / initial_acc
        if initial_acc > 0
        else 0.0
    )

    # Compute flip statistics
    n_flipped_to_correct = sum(1 for d in sample_details if d["flipped_to_correct"])
    n_flipped_to_wrong = sum(1 for d in sample_details if d["flipped_to_wrong"])
    n_stayed_correct = sum(1 for d in sample_details if d["initially_correct"] and d["finally_correct"])
    n_stayed_wrong = sum(1 for d in sample_details if not d["initially_correct"] and not d["finally_correct"])

    # Per-round accuracy delta (improvement from one round to the next)
    round_deltas = []
    for i in range(1, len(per_round_acc)):
        round_deltas.append(per_round_acc[i] - per_round_acc[i-1])

    results = {
        "dataset": dataset_name,
        "num_samples": len(test_samples),
        "num_rounds": num_rounds,
        "initial_accuracy": initial_acc,
        "final_accuracy": final_acc,
        "ci_margin": ci_margin,
        "ci_95": [max(0, final_acc - ci_margin), min(1, final_acc + ci_margin)],
        "improvement_rate": improvement,
        "absolute_improvement": final_acc - initial_acc,
        "strict_format_accuracy": strict_format_acc,
        "stateful_answer_accuracy": final_acc,
        "strict_extract_success_rate": final_answer_sources["strict_rate"],
        "fallback_extract_success_rate": final_answer_sources["fallback_rate"],
        "extract_success_rate": (
            final_answer_sources["strict_rate"]
            + final_answer_sources["fallback_rate"]
        ),
        "actor_carried_forward_rate": final_answer_sources["carried_forward_rate"],
        "actor_unresolved_rate": final_answer_sources["none_rate"],
        "carried_forward_count": final_answer_sources["carried_forward_count"],
        "extract_failure_count": final_answer_sources["none_count"],
        "per_round_extraction_rates": per_round_answer_sources,
        "per_round_answer_sources": per_round_answer_sources,
        "format_metrics": {
            "actor_answer_sources_by_round": {
                str(t): per_round_answer_sources[t]
                for t in range(len(per_round_answer_sources))
            },
            "actor_strict_format_rate_by_round": [
                rates["strict_rate"] for rates in per_round_answer_sources
            ],
            "actor_fallback_rate_by_round": [
                rates["fallback_rate"] for rates in per_round_answer_sources
            ],
            "actor_carried_forward_rate_by_round": [
                rates["carried_forward_rate"] for rates in per_round_answer_sources
            ],
            "actor_unresolved_rate_by_round": [
                rates["none_rate"] for rates in per_round_answer_sources
            ],
        },
        "per_round_accuracy": per_round_acc,
        "per_round_delta": round_deltas,
        "flip_statistics": {
            "flipped_to_correct": n_flipped_to_correct,
            "flipped_to_wrong": n_flipped_to_wrong,
            "stayed_correct": n_stayed_correct,
            "stayed_wrong": n_stayed_wrong,
        },
        "correct_at_each_round": [
            sum(1 for d in sample_details
                if answers_match(
                    all_round_answers[t][d["index"]],
                    labels[d["index"]],
                    d.get("task_type", "yes_no"),
                ))
            for t in range(num_rounds)
        ],
        "eval_time_seconds": round(eval_elapsed, 1),
        "avg_time_per_sample": round(eval_elapsed / max(len(test_samples), 1), 2),
        "sample_details": sample_details,
        "per_subject_metrics": compute_group_metrics(sample_details, "subject"),
    }

    # Print comprehensive results table
    _print_results_table(results)

    return results


def compute_group_metrics(
    sample_details: list[dict],
    group_key: str = "subject",
) -> dict[str, dict]:
    """Compute per-group accuracy metrics from sample details.

    Args:
        sample_details: List of per-sample evaluation dicts.
        group_key: Key to group by (e.g. "subject", "category").

    Returns:
        Dict mapping group name to metrics dict.
    """
    groups: dict[str, list[dict]] = {}
    for item in sample_details:
        group = item.get(group_key) or "unknown"
        groups.setdefault(group, []).append(item)

    result: dict[str, dict] = {}
    for group, items in sorted(groups.items()):
        n = len(items)
        initial_correct = sum(1 for x in items if x["initially_correct"])
        final_correct = sum(1 for x in items if x["finally_correct"])
        flipped_to_correct = sum(1 for x in items if x["flipped_to_correct"])
        flipped_to_wrong = sum(1 for x in items if x["flipped_to_wrong"])

        result[group] = {
            "num_samples": n,
            "initial_accuracy": initial_correct / n if n else 0.0,
            "final_accuracy": final_correct / n if n else 0.0,
            "absolute_improvement": (
                final_correct / n - initial_correct / n
                if n else 0.0
            ),
            "flipped_to_correct": flipped_to_correct,
            "flipped_to_wrong": flipped_to_wrong,
        }

    return result


def _print_results_table(results: dict) -> None:
    """Print a comprehensive, visually formatted results table."""
    sep = "=" * 65
    thin = "-" * 65

    logger.info(sep)
    logger.info("  BENCHMARK EVALUATION RESULTS")
    logger.info(sep)
    logger.info(f"  Dataset:            {results['dataset']}")
    logger.info(f"  Test samples:       {results['num_samples']}")
    logger.info(f"  Deliberation rounds:{results['num_rounds']}")
    logger.info(f"  Eval time:          {results['eval_time_seconds']}s "
                f"({results['avg_time_per_sample']}s/sample)")
    logger.info(thin)

    # Accuracy summary
    logger.info("  ACCURACY")
    logger.info(f"    Initial (round 1):  {results['initial_accuracy']:.4f}  "
                f"({results['initial_accuracy']*100:.2f}%)")
    logger.info(f"    Final   (round {results['num_rounds']}):  {results['final_accuracy']:.4f}  "
                f"({results['final_accuracy']*100:.2f}%)")
    ci = results['ci_95']
    logger.info(f"    95% CI:             [{ci[0]:.4f}, {ci[1]:.4f}]")
    logger.info(f"    Improvement rate:   {results['improvement_rate']:.4f}  "
                f"({results['improvement_rate']*100:.2f}%)")
    logger.info(f"    Absolute gain:      {results['absolute_improvement']:+.4f}  "
                f"({results['absolute_improvement']*100:+.2f}pp)")
    logger.info("  EXTRACTION")
    logger.info(f"    Strict FINAL_ANSWER:{results['strict_extract_success_rate']:.4f}  "
                f"({results['strict_extract_success_rate']*100:.2f}%)")
    logger.info(f"    Fallback parsed:    {results['fallback_extract_success_rate']:.4f}  "
                f"({results['fallback_extract_success_rate']*100:.2f}%)")
    logger.info(f"    Total extracted:    {results['extract_success_rate']:.4f}  "
                f"({results['extract_success_rate']*100:.2f}%)")
    logger.info(thin)

    # Per-round table
    logger.info("  PER-ROUND ACCURACY")
    header = "    Round  | Accuracy | Correct | Delta"
    logger.info(header)
    logger.info(f"    {'----':>4}   {'--------':>8}   {'-------':>7}   {'-----':>5}")
    for t, acc in enumerate(results['per_round_accuracy']):
        correct = results['correct_at_each_round'][t]
        delta = f"{results['per_round_delta'][t-1]:+.4f}" if t > 0 else "   ---"
        logger.info(f"    {t+1:>4}   {acc:>8.4f}   {correct:>7}   {delta:>5}")
    logger.info(thin)

    # Flip statistics
    flip = results['flip_statistics']
    total = results['num_samples']
    logger.info("  DELIBERATION EFFECT")
    logger.info(f"    Stayed correct:   {flip['stayed_correct']:>4}/{total}  "
                f"({flip['stayed_correct']/total*100:.1f}%)")
    logger.info(f"    Flipped correct:  {flip['flipped_to_correct']:>4}/{total}  "
                f"({flip['flipped_to_correct']/total*100:.1f}%)")
    logger.info(f"    Flipped wrong:    {flip['flipped_to_wrong']:>4}/{total}  "
                f"({flip['flipped_to_wrong']/total*100:.1f}%)")
    logger.info(f"    Stayed wrong:     {flip['stayed_wrong']:>4}/{total}  "
                f"({flip['stayed_wrong']/total*100:.1f}%)")
    logger.info(sep)
