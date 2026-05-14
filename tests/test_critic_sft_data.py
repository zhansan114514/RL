from __future__ import annotations

from src.parsing.critic_parser import parse_critic_response
from src.society.critic_sft_data import (
    attach_revision_result,
    build_critic_sft_candidates,
    examples_to_sft_rows,
    select_examples_per_skill,
)


def _sample():
    return {
        "sample_id": "mmlu_0",
        "question": "Which option is best supported?",
        "choices": ["Wrong A", "Right B", "Wrong C", "Wrong D"],
        "answer": "B",
        "task_type": "multiple_choice",
        "subject": "demo",
    }


def _actor_outputs():
    return [
        {
            "sample_id": "mmlu_0",
            "actor_name": "actor_direct",
            "actor_style": "direct",
            "actor_response": "I pick C.\nThe final result is C.",
            "actor_answer": "C",
            "actor_correct": False,
        },
        {
            "sample_id": "mmlu_0",
            "actor_name": "actor_evidence",
            "actor_style": "evidence",
            "actor_response": "The wording supports B.\nThe final result is B.",
            "actor_answer": "B",
            "actor_correct": True,
        },
        {
            "sample_id": "mmlu_0",
            "actor_name": "actor_elimination",
            "actor_style": "elimination",
            "actor_response": "I fail to eliminate C.\nThe final result is C.",
            "actor_answer": "C",
            "actor_correct": False,
        },
    ]


def test_builds_correction_and_keep_candidates_without_pair_fields():
    examples, metrics = build_critic_sft_candidates(
        samples=[_sample()],
        actor_outputs=_actor_outputs(),
        critic_skills=["reasoning", "verification"],
        dataset_name="mmlu",
    )

    correction = [
        ex for ex in examples
        if ex["case_type"] == "correction"
        and ex["metadata"]["actor_name"] == "actor_direct"
        and ex["critic_skill"] == "reasoning"
    ]
    keep = [
        ex for ex in examples
        if ex["case_type"] == "keep"
        and ex["metadata"]["actor_name"] == "actor_evidence"
        and ex["critic_skill"] == "reasoning"
    ]

    assert correction
    assert keep
    assert metrics["correction_count"] == 4
    assert metrics["keep_count"] == 2
    assert "chosen" not in correction[0]
    assert "rejected" not in correction[0]


def test_keep_requires_wrong_peer_conflict():
    sample = _sample()
    outputs = [
        {
            "sample_id": "mmlu_0",
            "actor_name": "actor_direct",
            "actor_style": "direct",
            "actor_response": "The final result is B.",
            "actor_answer": "B",
            "actor_correct": True,
        },
        {
            "sample_id": "mmlu_0",
            "actor_name": "actor_evidence",
            "actor_style": "evidence",
            "actor_response": "The final result is B.",
            "actor_answer": "B",
            "actor_correct": True,
        },
    ]

    examples, metrics = build_critic_sft_candidates(
        samples=[sample],
        actor_outputs=outputs,
        critic_skills=["reasoning"],
        dataset_name="mmlu",
    )

    assert examples == []
    assert metrics["skipped_counts"]["keep_without_wrong_peer"] == 2


def test_correction_and_keep_targets_have_expected_judgements():
    examples, _ = build_critic_sft_candidates(
        samples=[_sample()],
        actor_outputs=_actor_outputs(),
        critic_skills=["reasoning"],
        dataset_name="mmlu",
    )

    correction = next(
        ex for ex in examples
        if ex["case_type"] == "correction"
        and ex["metadata"]["actor_name"] == "actor_direct"
    )
    keep = next(
        ex for ex in examples
        if ex["case_type"] == "keep"
        and ex["metadata"]["actor_name"] == "actor_evidence"
    )

    parsed_correction = parse_critic_response(correction["response"], "multiple_choice")
    parsed_keep = parse_critic_response(keep["response"], "multiple_choice")

    assert parsed_correction.answer_correct == "no"
    assert parsed_correction.suggested_answer == "B"
    assert parsed_keep.answer_correct == "yes"
    assert parsed_keep.suggested_answer == "B"


def test_prompt_avoids_future_and_gold_metadata_fields():
    examples, _ = build_critic_sft_candidates(
        samples=[_sample()],
        actor_outputs=_actor_outputs(),
        critic_skills=["reasoning"],
        dataset_name="mmlu",
    )

    prompt = examples[0]["prompt"].lower()

    assert "gold answer" not in prompt
    assert "correct answer" not in prompt
    assert "target_answer" not in prompt
    assert "revised_answer" not in prompt
    assert "final_correct" not in prompt


def test_sft_row_selection_contains_prompt_response_only():
    examples, _ = build_critic_sft_candidates(
        samples=[_sample()],
        actor_outputs=_actor_outputs(),
        critic_skills=["reasoning"],
        dataset_name="mmlu",
    )
    selected, metrics = select_examples_per_skill(
        examples,
        critic_skills=["reasoning"],
        max_examples_per_critic=2,
        correction_ratio=0.5,
        keep_ratio=0.5,
        seed=1,
    )
    rows = examples_to_sft_rows(selected["reasoning"])

    assert len(rows) == 2
    assert set(rows[0]) == {"prompt", "response"}
    assert metrics["reasoning"]["num_examples"] == 2


def test_attach_revision_result_requires_final_correct():
    examples, _ = build_critic_sft_candidates(
        samples=[_sample()],
        actor_outputs=_actor_outputs(),
        critic_skills=["verification"],
        dataset_name="mmlu",
    )
    correction = next(ex for ex in examples if ex["case_type"] == "correction")

    updated, ok, reason = attach_revision_result(
        correction,
        revised_response="After feedback, answer B is best.\nThe final result is B.",
    )

    assert ok
    assert reason == "passed"
    assert updated["metadata"]["revised_answer"] == "B"
    assert updated["metadata"]["final_correct"] is True
