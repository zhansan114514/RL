"""Focused tests for first-round Actor SFT selection."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_train_sft_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    module_path = scripts_dir / "09_train_actors_sft.py"
    spec = importlib.util.spec_from_file_location("train_actors_sft", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _label(
    response_id: str,
    *,
    prompted_style: str = "evidence",
    primary_style: str = "evidence",
    is_correct: bool = True,
    confidence: float = 0.9,
    accepted: bool = True,
) -> dict:
    return {
        "response_id": response_id,
        "response": f"{response_id} has enough reasoning words for training.\nThe final result is A.",
        "answer": "A" if is_correct else "B",
        "is_correct": is_correct,
        "primary_style": primary_style,
        "reasoning_style_confidence": confidence,
        "prompted_style": prompted_style,
        "style_match": primary_style == prompted_style,
        "trainable_for_actor": is_correct,
        "style_verified": accepted,
        "accepted_for_actor_sft": accepted,
        "temperature": 0.7,
    }


def _args(**overrides):
    data = {
        "min_style_confidence": 0.55,
        "max_examples_per_question_style": 2,
        "balance_by_subject": False,
        "max_subject_imbalance_ratio": 3.0,
    }
    data.update(overrides)
    return type("Args", (), data)()


def test_correct_style_matched_responses_are_selected():
    train_sft = _load_train_sft_module()
    classified = [{
        "sample_id": "mmlu_0",
        "sample": {
            "question": "Q",
            "choices": ["A", "B", "C", "D"],
            "answer": "A",
            "task_type": "multiple_choice",
            "subject": "abstract_algebra",
        },
        "per_response_labels": [_label("ok")],
    }]

    examples, metrics = train_sft.select_sft_examples_for_style(
        train_sft.build_response_index(classified),
        "evidence",
        _args(),
    )

    assert [e["response_id"] for e in examples] == ["ok"]
    assert metrics["training_examples"] == 1
    assert metrics["subject_coverage"] == 1


def test_wrong_style_and_incorrect_responses_are_not_selected():
    train_sft = _load_train_sft_module()
    classified = [{
        "sample_id": "mmlu_0",
        "sample": {
            "question": "Q",
            "choices": ["A", "B", "C", "D"],
            "answer": "A",
            "task_type": "multiple_choice",
            "subject": "abstract_algebra",
        },
        "per_response_labels": [
            _label("wrong_style", primary_style="direct", accepted=False),
            _label("incorrect", is_correct=False, accepted=False),
            _label("low_conf", confidence=0.2, accepted=False),
        ],
    }]

    examples, metrics = train_sft.select_sft_examples_for_style(
        train_sft.build_response_index(classified),
        "evidence",
        _args(),
    )

    assert examples == []
    assert metrics["training_examples"] == 0


def test_deduplicates_by_question_and_response():
    train_sft = _load_train_sft_module()
    sample = {
        "question": "Q",
        "choices": ["A", "B", "C", "D"],
        "answer": "A",
        "task_type": "multiple_choice",
        "subject": "abstract_algebra",
    }
    first = _label("a")
    second = dict(first)
    second["response_id"] = "b"
    classified = [{
        "sample_id": "mmlu_0",
        "sample": sample,
        "per_response_labels": [first, second],
    }]

    examples, metrics = train_sft.select_sft_examples_for_style(
        train_sft.build_response_index(classified),
        "evidence",
        _args(),
    )

    assert len(examples) == 1
    assert metrics["candidate_examples"] == 2
    assert metrics["deduped_examples"] == 1


def test_subject_coverage_metrics_are_computed():
    train_sft = _load_train_sft_module()
    results = []
    for idx, subject in enumerate(["abstract_algebra", "anatomy", "astronomy"]):
        results.append({
            "sample_id": f"mmlu_{idx}",
            "sample": {
                "question": f"Q{idx}",
                "choices": ["A", "B", "C", "D"],
                "answer": "A",
                "task_type": "multiple_choice",
                "subject": subject,
            },
            "per_response_labels": [_label(f"ok_{idx}")],
        })

    examples, metrics = train_sft.select_sft_examples_for_style(
        train_sft.build_response_index(results),
        "evidence",
        _args(balance_by_subject=True),
    )

    assert len(examples) == 3
    assert metrics["subject_coverage"] == 3
    assert metrics["subject_counts"] == {
        "abstract_algebra": 1,
        "anatomy": 1,
        "astronomy": 1,
    }


def test_loose_subject_balance_does_not_cap_to_rare_subject_count():
    train_sft = _load_train_sft_module()
    results = []
    for idx in range(20):
        results.append({
            "sample_id": f"mmlu_a_{idx}",
            "sample": {
                "question": f"QA{idx}",
                "choices": ["A", "B", "C", "D"],
                "answer": "A",
                "task_type": "multiple_choice",
                "subject": "abundant",
            },
            "per_response_labels": [_label(f"abundant_{idx}")],
        })
    results.append({
        "sample_id": "mmlu_rare",
        "sample": {
            "question": "QR",
            "choices": ["A", "B", "C", "D"],
            "answer": "A",
            "task_type": "multiple_choice",
            "subject": "rare",
        },
        "per_response_labels": [_label("rare")],
    })

    examples, metrics = train_sft.select_sft_examples_for_style(
        train_sft.build_response_index(results),
        "evidence",
        _args(
            balance_by_subject=True,
            max_examples_per_question_style=10,
            max_subject_imbalance_ratio=3.0,
        ),
    )

    assert len(examples) > 4
    assert metrics["subject_coverage"] == 2
    assert metrics["subject_counts"]["rare"] == 1
