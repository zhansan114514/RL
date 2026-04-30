"""
Evaluation pipeline: benchmarks, per-round accuracy, metrics.
"""

from __future__ import annotations

import logging
import time


from src.algorithms.deliberation import deliberate_batch
from src.algorithms.reward import (
    extract_answer_with_source,
    compute_accuracy_with_ci,
    compute_per_round_accuracy,
    compute_improvement_rate,
    compute_extraction_success_rates,
)

logger = logging.getLogger(__name__)


def _answers_match(pred: str, label: str, task_type: str) -> bool:
    """Check if predicted answer matches label, using task-appropriate comparison."""
    if task_type == "math":
        from src.algorithms.reward import math_answers_equal
        return math_answers_equal(pred, label)
    return pred.upper() == label.upper()


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
        Dict with accuracy, per_round_accuracy, improvement_rate, ci.
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
    all_round_answers = [[] for _ in range(num_rounds)]
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

            round_answers = []
            round_extraction_sources = []
            for t, round_data in enumerate(trajectory):
                response = round_data["actor_response"]
                extraction = extract_answer_with_source(response, task_type)
                answer = extraction.answer
                all_round_answers[t].append(answer or "")
                all_round_responses[t].append(response)
                round_answers.append(answer or "")
                round_extraction_sources.append(extraction.source or "none")

            initial_pred = round_answers[0] if round_answers else ""
            final_pred = round_answers[-1] if round_answers else ""

            # Use math-aware comparison for math tasks
            if task_type == "math":
                from src.algorithms.reward import math_answers_equal
                initially_correct = math_answers_equal(initial_pred, label)
                finally_correct = math_answers_equal(final_pred, label)
            else:
                correct_label = label.upper().strip()
                initially_correct = (initial_pred.upper() == correct_label)
                finally_correct = (final_pred.upper() == correct_label)

            sample_details.append({
                "index": i,
                "initial_answer": initial_pred,
                "final_answer": final_pred,
                "initial_extract_source": (
                    round_extraction_sources[0] if round_extraction_sources else "none"
                ),
                "final_extract_source": (
                    round_extraction_sources[-1] if round_extraction_sources else "none"
                ),
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
    task_type = test_samples[0].get("task_type", "yes_no") if test_samples else "yes_no"
    sample_task_types = [sample.get("task_type", "yes_no") for sample in test_samples]
    per_round_extraction_rates = [
        compute_extraction_success_rates(round_responses, sample_task_types)
        for round_responses in all_round_responses
    ]
    final_extraction_rates = (
        per_round_extraction_rates[-1]
        if per_round_extraction_rates
        else compute_extraction_success_rates([])
    )
    per_round_acc = compute_per_round_accuracy(all_round_answers, labels, task_type=task_type)
    final_acc, ci_margin = compute_accuracy_with_ci(
        all_round_answers[-1], labels, task_type=task_type,
    )
    initial_acc = per_round_acc[0] if per_round_acc else 0.0
    improvement = compute_improvement_rate(final_acc, initial_acc)

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
        "strict_extract_success_rate": final_extraction_rates["strict_extract_success_rate"],
        "fallback_extract_success_rate": final_extraction_rates["fallback_extract_success_rate"],
        "extract_success_rate": final_extraction_rates["extract_success_rate"],
        "per_round_extraction_rates": per_round_extraction_rates,
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
                if _answers_match(all_round_answers[t][d["index"]], labels[d["index"]], d.get("task_type", "yes_no")))
            for t in range(num_rounds)
        ],
        "eval_time_seconds": round(eval_elapsed, 1),
        "avg_time_per_sample": round(eval_elapsed / max(len(test_samples), 1), 2),
        "sample_details": sample_details,
    }

    # Print comprehensive results table
    _print_results_table(results)

    return results


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
