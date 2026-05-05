"""Diagnostics for classification, Critic schema, and Router behavior."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.society.critic_schema import parse_critic_judgement


def summarize_critic_schema(responses: list[str], actor_answers: list[str | None] | None = None) -> dict[str, Any]:
    total = len(responses)
    valid = 0
    confidence_ok = 0
    answer_correct_ok = 0
    error_counts: Counter[str] = Counter()

    for i, response in enumerate(responses):
        actor_answer = actor_answers[i] if actor_answers and i < len(actor_answers) else None
        judgement = parse_critic_judgement(response, actor_answer=actor_answer)
        if judgement.schema_valid:
            valid += 1
        if judgement.confidence is not None:
            confidence_ok += 1
        if judgement.answer_correct is not None:
            answer_correct_ok += 1
        error_counts.update(judgement.schema_errors)

    return {
        "total": total,
        "schema_valid_rate": valid / total if total else 0.0,
        "parse_confidence_success_rate": confidence_ok / total if total else 0.0,
        "parse_answer_correct_success_rate": answer_correct_ok / total if total else 0.0,
        "schema_error_counts": dict(error_counts),
    }


def summarize_router_rounds(rounds: list[Any]) -> dict[str, Any]:
    total_actor_routes = 0
    empty_routes = 0
    uniform_fallback_routes = 0
    selected_counter: Counter[str] = Counter()
    invalid_weight_violations = 0
    critic_feedbacks = 0
    schema_valid_feedbacks = 0

    for round_data in rounds:
        for routed in getattr(round_data, "routed_feedbacks", {}).values():
            total_actor_routes += 1
            if not getattr(routed, "selected_critics", []):
                empty_routes += 1
            if getattr(routed, "used_uniform_fallback", False):
                uniform_fallback_routes += 1
            selected_counter.update(getattr(routed, "selected_critics", []))
            for feedback in getattr(routed, "raw_feedbacks", []):
                critic_feedbacks += 1
                if feedback.schema_valid:
                    schema_valid_feedbacks += 1
                if not feedback.schema_valid and feedback.confidence > 0:
                    invalid_weight_violations += 1

    return {
        "total_actor_routes": total_actor_routes,
        "empty_route_rate": empty_routes / total_actor_routes if total_actor_routes else 0.0,
        "uniform_fallback_rate": (
            uniform_fallback_routes / total_actor_routes if total_actor_routes else 0.0
        ),
        "selected_critic_distribution": dict(selected_counter),
        "schema_invalid_critic_weight_violations": invalid_weight_violations,
        "critic_schema_valid_rate": (
            schema_valid_feedbacks / critic_feedbacks if critic_feedbacks else 0.0
        ),
    }


def classification_distribution_report(classified_results: list[dict[str, Any]]) -> dict[str, Any]:
    styles: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    for result in classified_results:
        for label in result.get("per_response_labels", []):
            if label.get("primary_style"):
                styles[label["primary_style"]] += 1
            if label.get("format_status"):
                formats[label["format_status"]] += 1
            profile = label.get("error_profile")
            if profile and profile.get("primary"):
                errors[profile["primary"]] += 1
    return {
        "style_distribution": dict(styles),
        "error_profile_distribution": dict(errors),
        "format_status_distribution": dict(formats),
    }
