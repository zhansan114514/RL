"""Quality gates for Critic DPO training pairs."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.algorithms.reward import extract_answer, math_answers_equal, normalize_answer
from src.parsing.critic_parser import parse_critic_response


def actor_error_response_is_usable(
    sample: dict[str, Any],
    response: str,
    actor_answer: str | None = None,
    require_extracted_answer: bool = True,
) -> tuple[bool, str, str | None]:
    """Return whether an Actor response is a usable wrong-answer case."""
    text = (response or "").strip()
    if not text:
        return False, "empty_actor_response", actor_answer

    correct_answer = sample.get("answer", "")
    if correct_answer in (None, ""):
        return False, "missing_correct_answer", actor_answer

    task_type = sample.get("task_type", "multiple_choice")
    answer = actor_answer if actor_answer not in (None, "") else extract_answer(text, task_type)
    if require_extracted_answer and answer in (None, ""):
        return False, "missing_actor_answer", answer

    if _answers_equal(answer or "", str(correct_answer), task_type):
        return False, "actor_response_correct", answer
    return True, "kept", answer


def filter_critic_raw_pairs(
    raw_pairs: list[dict[str, Any]],
    require_extracted_answer: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop invalid or duplicate raw Critic error cases."""
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    dropped: Counter[str] = Counter()

    for pair in raw_pairs:
        sample = pair.get("sample", {})
        response = pair.get("actor_response", "")
        ok, reason, actor_answer = actor_error_response_is_usable(
            sample,
            response,
            actor_answer=pair.get("actor_answer"),
            require_extracted_answer=require_extracted_answer,
        )
        if not ok:
            dropped[reason] += 1
            continue

        key = critic_raw_pair_key(pair)
        if key in seen:
            dropped["duplicate_actor_error_case"] += 1
            continue

        seen.add(key)
        normalized = dict(pair)
        normalized["actor_answer"] = actor_answer or ""
        kept.append(normalized)

    return kept, {
        "input_count": len(raw_pairs),
        "kept_count": len(kept),
        "dropped_count": len(raw_pairs) - len(kept),
        "dropped_counts": dict(dropped),
        "unique_error_cases": len(seen),
    }


def filter_routed_items_for_raw_pairs(
    routed_items: list[Any],
    raw_pairs: list[dict[str, Any]],
) -> tuple[list[Any], dict[str, Any]]:
    """Keep routed items whose underlying raw pair survived quality gates."""
    allowed_ids = {
        str(pair.get("raw_pair_id", ""))
        for pair in raw_pairs
        if pair.get("raw_pair_id")
    }
    allowed_key_to_base = {
        critic_raw_pair_key(pair): str(pair.get("raw_pair_id", ""))
        or _key_to_string(critic_raw_pair_key(pair))
        for pair in raw_pairs
    }
    kept: list[Any] = []
    seen_assignments: set[tuple[str, str | None]] = set()
    dropped: Counter[str] = Counter()

    for item in routed_items:
        response_id = str(getattr(item, "response_id", "") or "")
        item_key = _sample_response_key(
            getattr(item, "sample", {}) or {},
            getattr(item, "response", "") or "",
        )
        if response_id:
            if response_id in allowed_ids:
                base_key = response_id
                allowed = True
            elif item_key in allowed_key_to_base:
                base_key = allowed_key_to_base[item_key]
                allowed = True
            else:
                base_key = response_id
                allowed = False
        else:
            base_key = allowed_key_to_base.get(item_key, _key_to_string(item_key))
            allowed = item_key in allowed_key_to_base
        if not allowed:
            dropped["raw_pair_filtered"] += 1
            continue

        skill = getattr(getattr(item, "skill", None), "value", None)
        assignment_key = (base_key, skill)
        if assignment_key in seen_assignments:
            dropped["duplicate_routed_assignment"] += 1
            continue
        seen_assignments.add(assignment_key)
        kept.append(item)

    return kept, {
        "input_count": len(routed_items),
        "kept_count": len(kept),
        "dropped_count": len(routed_items) - len(kept),
        "dropped_counts": dict(dropped),
    }


def structured_critic_pair_is_usable(
    pair: dict[str, Any],
    require_correct_suggestion: bool = True,
) -> tuple[bool, str]:
    """Validate structured Critic chosen/rejected completions before DPO."""
    sample = pair.get("sample", {})
    task_type = sample.get("task_type", "multiple_choice")
    correct_answer = str(
        pair.get("metadata", {}).get("correct_answer")
        or sample.get("answer", "")
    )

    chosen = parse_critic_response(pair.get("chosen", ""), task_type)
    if not chosen.usable_for_feedback:
        return False, "chosen_unusable_feedback"
    if not chosen.has_answer_correct or chosen.answer_correct != "no":
        return False, "chosen_not_marked_incorrect"
    if not chosen.has_confidence:
        return False, "chosen_missing_confidence"
    if not chosen.has_suggested_answer:
        return False, "chosen_missing_suggested_answer"
    if require_correct_suggestion and not _answers_equal(
        chosen.suggested_answer or "",
        correct_answer,
        task_type,
    ):
        return False, "chosen_suggested_answer_mismatch"

    rejected = parse_critic_response(pair.get("rejected", ""), task_type)
    if not rejected.usable_for_feedback:
        return False, "rejected_unusable_feedback"
    if not rejected.has_answer_correct:
        return False, "rejected_missing_answer_correct"
    if not rejected.has_confidence:
        return False, "rejected_missing_confidence"

    return True, "kept"


def critic_raw_pair_key(pair: dict[str, Any]) -> tuple[str, str]:
    return _sample_response_key(
        pair.get("sample", {}) or {},
        pair.get("actor_response", "") or "",
    )


def routed_item_base_key(item: Any) -> str:
    response_id = str(getattr(item, "response_id", "") or "")
    if response_id:
        return response_id
    question, response = _sample_response_key(
        getattr(item, "sample", {}) or {},
        getattr(item, "response", "") or "",
    )
    return f"{question}||{response}"


def _sample_response_key(sample: dict[str, Any], response: str) -> tuple[str, str]:
    return (
        str(sample.get("question", "")).strip(),
        (response or "").strip(),
    )


def _key_to_string(key: tuple[str, str]) -> str:
    return f"{key[0]}||{key[1]}"


def _answers_equal(pred: str, label: str, task_type: str) -> bool:
    if task_type == "math":
        return math_answers_equal(pred, label)
    return normalize_answer(pred, task_type) == normalize_answer(label, task_type)
