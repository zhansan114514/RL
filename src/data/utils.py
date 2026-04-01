"""Data utility functions: statistics, visualization helpers."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def compute_dataset_stats(samples: list[dict]) -> dict[str, Any]:
    """
    Compute statistics for a dataset split.

    Args:
        samples: List of standardized samples.

    Returns:
        Dictionary with count, answer distribution, etc.
    """
    if not samples:
        return {"count": 0}

    stats: dict[str, Any] = {"count": len(samples)}

    # Answer distribution
    answers = [s.get("answer", "") for s in samples if s.get("answer")]
    if answers:
        answer_counts = Counter(answers)
        stats["answer_distribution"] = dict(answer_counts)
        stats["answer_ratios"] = {
            k: round(v / len(answers), 4) for k, v in answer_counts.items()
        }

    # Task type
    task_types = set(s.get("task_type", "") for s in samples)
    stats["task_types"] = list(task_types)

    # Average question length
    questions = [s.get("question", "") for s in samples if s.get("question")]
    if questions:
        stats["avg_question_length"] = round(
            sum(len(q.split()) for q in questions) / len(questions), 1
        )

    return stats


def log_dataset_summary(data: dict[str, list[dict]], name: str) -> None:
    """Print a formatted summary of a loaded dataset."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {name}")
    logger.info(f"{'='*60}")

    for split_name, samples in data.items():
        stats = compute_dataset_stats(samples)
        logger.info(f"  {split_name}: {stats['count']} samples")
        if "answer_distribution" in stats:
            logger.info(f"    Answers: {stats['answer_distribution']}")
