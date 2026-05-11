"""Heuristic style-diversity metrics for Actor responses."""

from __future__ import annotations

from collections import Counter
from typing import Optional


ELIMINATION_MARKERS = [
    "eliminate",
    "rule out",
    "incorrect",
    "less likely",
    "option",
    "choice",
]

EVIDENCE_MARKERS = [
    "because",
    "according to",
    "the question states",
    "definition",
    "evidence",
    "given",
    "passage",
]


def answer_diversity_rate(round_actor_answers: list[dict[str, Optional[str]]]) -> float:
    """Fraction of samples where Actors disagreed in a round."""
    if not round_actor_answers:
        return 0.0
    disagree = 0
    for answers in round_actor_answers:
        unique = {answer for answer in answers.values() if answer}
        if len(unique) > 1:
            disagree += 1
    return disagree / len(round_actor_answers)


def compute_style_behavior(
    actor_responses: list[dict[str, str]],
) -> dict[str, dict[str, float]]:
    """Compute simple style behavior diagnostics by Actor name."""
    per_actor: dict[str, list[str]] = {}
    for response_map in actor_responses:
        for actor_name, response in response_map.items():
            per_actor.setdefault(actor_name, []).append(response or "")

    result: dict[str, dict[str, float]] = {}
    for actor_name, responses in per_actor.items():
        total = len(responses)
        lengths = [len(response.split()) for response in responses]
        text = "\n".join(responses).lower()
        result[actor_name] = {
            "avg_length": sum(lengths) / total if total else 0.0,
            "elimination_marker_count": float(_count_markers(text, ELIMINATION_MARKERS)),
            "evidence_marker_count": float(_count_markers(text, EVIDENCE_MARKERS)),
            "option_reference_count": float(_count_option_references(text)),
        }
    return result


def _count_markers(text: str, markers: list[str]) -> int:
    return sum(text.count(marker) for marker in markers)


def _count_option_references(text: str) -> int:
    counts = Counter()
    for label in ("a", "b", "c", "d"):
        counts[label] = text.count(f"option {label}") + text.count(f"({label})")
    return sum(counts.values())
