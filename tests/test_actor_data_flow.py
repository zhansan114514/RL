"""Tests for style-prompted classification and actor pair construction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script(name: str):
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    module_path = scripts_dir / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classification_checkpoint_metadata_tracks_input_fingerprint(tmp_path):
    classify = _load_script("08_classify_data.py")
    input_dir = tmp_path / "bootstrap"
    input_dir.mkdir()
    trajectory_file = input_dir / "trajectories.jsonl"
    trajectory_file.write_text('{"sample_id":"mmlu_0"}\n')
    args = type("Args", (), {
        "input_dir": str(input_dir),
        "min_style_confidence": 0.65,
    })()

    first = classify.checkpoint_metadata(args)
    trajectory_file.write_text('{"sample_id":"mmlu_0","changed":true}\n')
    second = classify.checkpoint_metadata(args)

    assert first["input_file"]["sha256"] != second["input_file"]["sha256"]


def test_classification_counts_include_resumed_failures():
    classify = _load_script("08_classify_data.py")
    results = [{
        "sample_id": "mmlu_0",
        "per_response_labels": [
            {
                "is_correct": True,
                "classification_source": "shared_classifier",
            },
            {
                "is_correct": True,
                "classification_source": None,
            },
            {
                "is_correct": False,
                "classification_source": "local_correctness_only",
            },
        ],
    }]

    assert classify.classification_counts(results) == (2, 1)


def test_actor_sft_acceptance_requires_style_verification():
    classify = _load_script("08_classify_data.py")
    label = {
        "prompted_style": "evidence",
        "primary_style": "direct",
        "task_type": "multiple_choice",
        "reasoning_style_confidence": 0.9,
        "is_correct": True,
        "response": "The answer follows from the question clue.\nThe final result is A.",
    }
    args = type("Args", (), {"min_style_confidence": 0.65})()

    classify.update_actor_acceptance(label, args)

    assert label["trainable_for_actor"] is True
    assert label["style_match"] is False
    assert label["style_verified"] is False
    assert label["accepted_for_actor_sft"] is False


def test_actor_sft_selection_uses_only_accepted_target_style_examples():
    train_sft = _load_script("09_train_actors_sft.py")
    sample = {
        "question": "Q",
        "choices": ["A", "B", "C", "D"],
        "answer": "A",
        "task_type": "multiple_choice",
        "subject": "abstract_algebra",
    }
    classified_results = [{
        "sample_id": "mmlu_0",
        "sample": sample,
        "per_response_labels": [
            {
                "response_id": "chosen",
                "response": "The question states the key fact, so A follows.\nThe final result is A.",
                "answer": "A",
                "is_correct": True,
                "primary_style": "evidence",
                "reasoning_style_confidence": 0.9,
                "format_status": "valid",
                "prompted_style": "evidence",
                "style_match": True,
                "trainable_for_actor": True,
                "style_verified": True,
                "accepted_for_actor_sft": True,
                "temperature": 0.7,
            },
            {
                "response_id": "wrong",
                "response": "This chooses the wrong option.\nThe final result is B.",
                "answer": "B",
                "is_correct": False,
                "format_status": "valid",
                "prompted_style": "evidence",
                "accepted_for_actor_sft": False,
            },
            {
                "response_id": "other_style",
                "response": "A.\nThe final result is A.",
                "answer": "A",
                "is_correct": True,
                "primary_style": "direct",
                "reasoning_style_confidence": 0.8,
                "format_status": "valid",
                "prompted_style": "direct",
                "style_match": True,
                "trainable_for_actor": True,
                "style_verified": True,
                "accepted_for_actor_sft": True,
            },
            {
                "response_id": "answer_only",
                "response": "A",
                "answer": "A",
                "is_correct": True,
                "primary_style": "evidence",
                "reasoning_style_confidence": 0.95,
                "format_status": "answer_only",
                "prompted_style": "evidence",
                "style_match": True,
                "trainable_for_actor": False,
                "style_verified": True,
                "accepted_for_actor_sft": False,
            },
        ],
    }]
    args = type("Args", (), {
        "min_style_confidence": 0.55,
        "max_examples_per_question_style": 2,
        "balance_by_subject": True,
        "max_subject_imbalance_ratio": 3.0,
    })()

    by_style = train_sft.build_response_index(classified_results)
    examples, metrics = train_sft.select_sft_examples_for_style(
        by_style,
        "evidence",
        args,
    )
    rows = train_sft.examples_to_sft_dataset(examples, "mmlu", "evidence")

    assert [example["response_id"] for example in examples] == ["chosen"]
    assert metrics["training_examples"] == 1
    assert metrics["subject_coverage"] == 1
    assert set(rows[0]) == {"prompt", "response", "metadata"}
    assert "chosen" not in rows[0]
    assert "rejected" not in rows[0]
    assert rows[0]["metadata"]["style"] == "evidence"
    assert rows[0]["metadata"]["temperature"] == 0.7


def test_actor_sft_selection_rejects_correct_wrong_style_and_incorrect():
    train_sft = _load_script("09_train_actors_sft.py")
    sample = {
        "question": "Q",
        "choices": ["A", "B", "C", "D"],
        "answer": "A",
        "task_type": "multiple_choice",
        "subject": "abstract_algebra",
    }
    labels = [
        {
            "response_id": "strict",
            "response": "Evidence.\nThe final result is A.",
            "answer": "A",
            "is_correct": True,
            "primary_style": "evidence",
            "reasoning_style_confidence": 0.9,
            "format_status": "valid",
            "prompted_style": "evidence",
            "style_match": True,
            "trainable_for_actor": True,
            "style_verified": True,
            "accepted_for_actor_sft": True,
        },
        {
            "response_id": "low_confidence",
            "response": "Low confidence but prompted evidence.\nThe final result is A.",
            "answer": "A",
            "is_correct": True,
            "primary_style": "evidence",
            "reasoning_style_confidence": 0.2,
            "format_status": "valid",
            "prompted_style": "evidence",
            "style_match": True,
            "trainable_for_actor": True,
            "style_verified": False,
            "accepted_for_actor_sft": False,
        },
        {
            "response_id": "wrong_style",
            "response": "Classifier missed the style.\nThe final result is A.",
            "answer": "A",
            "is_correct": True,
            "primary_style": "direct",
            "reasoning_style_confidence": 0.8,
            "format_status": "valid",
            "prompted_style": "evidence",
            "style_match": False,
            "trainable_for_actor": True,
            "style_verified": False,
            "accepted_for_actor_sft": False,
        },
        {
            "response_id": "wrong",
            "response": "Wrong.\nThe final result is B.",
            "answer": "B",
            "is_correct": False,
            "format_status": "valid",
            "prompted_style": "evidence",
            "accepted_for_actor_sft": False,
        },
    ]
    classified_results = [{
        "sample_id": "mmlu_0",
        "sample": sample,
        "per_response_labels": labels,
    }]
    args = type("Args", (), {
        "min_style_confidence": 0.55,
        "max_examples_per_question_style": 8,
        "balance_by_subject": False,
        "max_subject_imbalance_ratio": 3.0,
    })()

    examples, _ = train_sft.select_sft_examples_for_style(
        train_sft.build_response_index(classified_results),
        "evidence",
        args,
    )

    assert [example["response_id"] for example in examples] == ["strict"]


def test_actor_sft_selection_enforces_per_question_cap():
    train_sft = _load_script("09_train_actors_sft.py")
    sample = {
        "question": "Q",
        "choices": ["A", "B", "C", "D"],
        "answer": "A",
        "task_type": "multiple_choice",
        "subject": "abstract_algebra",
    }
    labels = []
    for i in range(3):
        labels.append({
            "response_id": f"accepted_{i}",
            "response": (
                f"Evidence response number {i} cites the question clue.\n"
                "The final result is A."
            ),
            "answer": "A",
            "is_correct": True,
            "primary_style": "evidence",
            "reasoning_style_confidence": 0.9,
            "prompted_style": "evidence",
            "style_match": True,
            "trainable_for_actor": True,
            "style_verified": True,
            "accepted_for_actor_sft": True,
        })
    classified_results = [{
        "sample_id": "mmlu_0",
        "sample": sample,
        "per_response_labels": labels,
    }]
    args = type("Args", (), {
        "min_style_confidence": 0.55,
        "max_examples_per_question_style": 2,
        "balance_by_subject": False,
        "max_subject_imbalance_ratio": 3.0,
    })()

    examples, metrics = train_sft.select_sft_examples_for_style(
        train_sft.build_response_index(classified_results),
        "evidence",
        args,
    )

    assert len(examples) == 2
    assert metrics["candidate_examples"] == 3
    assert metrics["after_question_cap"] == 2


def test_critic_structured_pairs_do_not_oversample_or_keep_invalid_judgements():
    diversify = _load_script("10_diversify_critics.py")
    from src.society.agent_registry import CriticSkill
    from src.society.diversity_split import RoutedTrainingItem

    sample_a = {
        "question": "Q1",
        "answer": "A",
        "task_type": "multiple_choice",
    }
    sample_b = {
        "question": "Q2",
        "answer": "B",
        "task_type": "multiple_choice",
    }
    raw_pairs = [
        {
            "raw_pair_id": "raw_1",
            "sample": sample_a,
            "actor_response": "Wrong reasoning.\nThe final result is C.",
            "actor_answer": "C",
            "correct_answer": "A",
            "actor_name": "actor_direct",
        },
        {
            "raw_pair_id": "raw_1_dup",
            "sample": sample_a,
            "actor_response": "Wrong reasoning.\nThe final result is C.",
            "actor_answer": "C",
            "correct_answer": "A",
            "actor_name": "actor_direct",
        },
        {
            "raw_pair_id": "raw_correct",
            "sample": sample_b,
            "actor_response": "Correct reasoning.\nThe final result is B.",
            "actor_answer": "B",
            "correct_answer": "B",
            "actor_name": "actor_direct",
        },
    ]
    routed_items = [
        RoutedTrainingItem(
            sample=sample_a,
            response="Wrong reasoning.\nThe final result is C.",
            skill=CriticSkill.REASONING,
            weight=1.0,
            profile={
                "primary": "reasoning",
                "confidence": 0.9,
                "evidence": "The actor uses the wrong inference.",
            },
            response_id="raw_1",
        ),
        RoutedTrainingItem(
            sample=sample_a,
            response="Wrong reasoning.\nThe final result is C.",
            skill=CriticSkill.REASONING,
            weight=1.0,
            profile={
                "primary": "reasoning",
                "confidence": 0.9,
                "evidence": "Duplicate route should not create another pair.",
            },
            response_id="raw_1_dup",
        ),
        RoutedTrainingItem(
            sample=sample_b,
            response="Correct reasoning.\nThe final result is B.",
            skill=CriticSkill.REASONING,
            weight=1.0,
            profile={
                "primary": "reasoning",
                "confidence": 0.9,
                "evidence": "Correct actor response must be filtered.",
            },
            response_id="raw_correct",
        ),
    ]

    pairs, metrics = diversify.build_structured_critic_pairs(
        raw_pairs=raw_pairs,
        routed_items=routed_items,
        critic_skill="reasoning",
        max_pairs=10,
        seed=42,
        min_real_specialty_items=1,
        min_unique_pairs=1,
        max_duplicate_rate=0.0,
        pair_mix={"specialty": 1.0, "general": 0.0},
    )

    assert len(pairs) == 1
    assert pairs[0]["metadata"]["raw_pair_id"] == "raw_1"
    assert metrics["unique_pair_count"] == 1
    assert metrics["duplicate_rate"] == 0.0
    assert metrics["raw_filter"]["dropped_counts"] == {
        "duplicate_actor_error_case": 1,
        "actor_response_correct": 1,
    }


def test_critic_structured_pairs_fail_when_unique_threshold_not_met():
    diversify = _load_script("10_diversify_critics.py")
    from src.society.agent_registry import CriticSkill
    from src.society.diversity_split import RoutedTrainingItem

    sample = {
        "question": "Q",
        "answer": "A",
        "task_type": "multiple_choice",
    }
    raw_pairs = [{
        "raw_pair_id": "raw_1",
        "sample": sample,
        "actor_response": "Wrong reasoning.\nThe final result is C.",
        "actor_answer": "C",
        "correct_answer": "A",
        "actor_name": "actor_direct",
    }]
    routed_items = [RoutedTrainingItem(
        sample=sample,
        response="Wrong reasoning.\nThe final result is C.",
        skill=CriticSkill.REASONING,
        weight=1.0,
        profile={
            "primary": "reasoning",
            "confidence": 0.9,
            "evidence": "The actor uses the wrong inference.",
        },
        response_id="raw_1",
    )]

    pairs, metrics = diversify.build_structured_critic_pairs(
        raw_pairs=raw_pairs,
        routed_items=routed_items,
        critic_skill="reasoning",
        max_pairs=10,
        seed=42,
        min_real_specialty_items=1,
        min_unique_pairs=2,
        pair_mix={"specialty": 1.0, "general": 0.0},
    )

    assert pairs == []
    assert metrics["status"] == "frozen_base"
    assert metrics["reason"] == "unique_pairs_below_threshold"


def test_critic_routed_filter_collapses_duplicate_content_with_different_ids():
    from src.society.agent_registry import CriticSkill
    from src.society.critic_pair_quality import filter_routed_items_for_raw_pairs
    from src.society.diversity_split import RoutedTrainingItem

    sample = {
        "question": "Q",
        "answer": "A",
        "task_type": "multiple_choice",
    }
    raw_pairs = [{
        "raw_pair_id": "raw_1",
        "sample": sample,
        "actor_response": "Wrong reasoning.\nThe final result is C.",
        "actor_answer": "C",
        "correct_answer": "A",
    }]
    routed_items = [
        RoutedTrainingItem(
            sample=sample,
            response="Wrong reasoning.\nThe final result is C.",
            skill=CriticSkill.REASONING,
            weight=1.0,
            profile={},
            response_id="raw_1",
        ),
        RoutedTrainingItem(
            sample=sample,
            response="Wrong reasoning.\nThe final result is C.",
            skill=CriticSkill.REASONING,
            weight=1.0,
            profile={},
            response_id="raw_1_dup",
        ),
    ]

    kept, metrics = filter_routed_items_for_raw_pairs(routed_items, raw_pairs)

    assert len(kept) == 1
    assert kept[0].response_id == "raw_1"
    assert metrics["dropped_counts"] == {"duplicate_routed_assignment": 1}


def test_critic_training_mix_fills_with_remaining_real_items_without_replacement():
    from src.society.agent_registry import CriticSkill
    from src.society.diversity_split import DiversitySplit, RoutedTrainingItem

    items = []
    for idx, skill in enumerate([
        CriticSkill.REASONING,
        CriticSkill.KNOWLEDGE,
        CriticSkill.GROUNDING,
        CriticSkill.VERIFICATION,
    ]):
        items.append(RoutedTrainingItem(
            sample={"question": f"Q{idx}"},
            response=f"Wrong {idx}",
            skill=skill,
            weight=1.0,
            profile={},
            response_id=f"raw_{idx}",
        ))

    selected = DiversitySplit(seed=42, use_api=False).build_critic_training_mix(
        all_items=items,
        target_skill=CriticSkill.REASONING,
        max_items=4,
        min_specialty_items=1,
        min_specialty_ratio=0.0,
        specialty_ratio=0.75,
        general_ratio=0.25,
        calibration_ratio=0.0,
    )

    assert len(selected) == 4
    assert len({item.response_id for item in selected}) == 4


def test_reasoning_style_parser_uses_evidence_when_primary_label_is_ambiguous():
    from src.society.agent_registry import ReasoningStyle
    from src.society.data_classifier import _parse_style_json_response

    parsed = _parse_style_json_response("""{
      "primary_style": "direct|evidence|elimination",
      "secondary_styles": [],
      "format_status": "valid",
      "confidence": 0.8,
      "evidence": "The response compares alternatives and eliminates option B."
    }""")

    assert parsed is not None
    assert parsed.primary_style == ReasoningStyle.ELIMINATION


def test_society_actor_prompt_builder_conditions_generation_once():
    from src.prompts.prompt_builder import build_simple_actor_prompt
    from src.society.agent_registry import ReasoningStyle

    prompt = build_simple_actor_prompt(
        {
            "question": "Q",
            "choices": ["A", "B", "C", "D"],
            "task_type": "multiple_choice",
        },
        "mmlu",
        style=ReasoningStyle.ELIMINATION,
    )

    assert prompt.count("You are Actor-elimination") == 1
    assert "comparing the options" in prompt
    assert prompt.count("The final result is <answer>.") == 1


def test_society_actor_dpo_prompt_is_style_conditioned(monkeypatch, tmp_path):
    from src.society import society_trainer
    from src.prompts.prompt_builder import build_actor_prompt
    from src.society.agent_registry import AgentConfig, AgentRole, ReasoningStyle

    captured = {}

    def fake_detect_model_type(model_name):
        return "qwen2.5"

    def fake_train_dpo(**kwargs):
        captured["dataset"] = kwargs["preference_dataset"]
        return str(tmp_path / "adapter")

    monkeypatch.setattr(
        "src.utils.model_utils.detect_model_type",
        fake_detect_model_type,
    )
    monkeypatch.setattr("src.training.dpo_trainer.train_dpo", fake_train_dpo)

    checkpoint = society_trainer._run_dpo_training(
        model_name="/models/base",
        preference_pairs=[{
            "sample": {
                "question": "Q",
                "choices": ["A", "B", "C", "D"],
                "answer": "A",
                "task_type": "multiple_choice",
            },
            "chosen": "Short reasoning.\nThe final result is A.",
            "rejected": "Wrong reasoning.\nThe final result is B.",
            "metadata": {"prompted_style": "direct"},
        }],
        output_dir=str(tmp_path),
        agent_type="actor",
        agent_name="actor_direct",
        dataset_name="mmlu",
        actor_style=ReasoningStyle.DIRECT,
    )

    prompt = captured["dataset"][0]["prompt"]
    expected_prompt = build_actor_prompt(
        AgentConfig(
            name="actor_direct",
            role=AgentRole.ACTOR,
            model_path="/models/base",
            reasoning_style=ReasoningStyle.DIRECT,
        ),
        {
            "question": "Q",
            "choices": ["A", "B", "C", "D"],
            "answer": "A",
            "task_type": "multiple_choice",
        },
        "mmlu",
    )
    assert checkpoint == str(tmp_path / "adapter")
    assert prompt == expected_prompt
    assert prompt.startswith("/no_think\n")
    assert prompt.count("/no_think") == 1
    assert prompt.count("You are Actor-direct") == 1
    assert "shortest sufficient reasoning" in prompt
    assert prompt.count("The final result is <answer>.") == 1
    assert "FINAL_ANSWER" not in prompt


def test_actor_sft_prompt_matches_simple_actor_prompt():
    train_sft = _load_script("09_train_actors_sft.py")
    from src.prompts.prompt_builder import build_simple_actor_prompt
    from src.society.agent_registry import ReasoningStyle

    sample = {
        "question": "Q",
        "choices": ["A", "B", "C", "D"],
        "answer": "A",
        "task_type": "multiple_choice",
    }

    prompt = train_sft._actor_training_prompt("mmlu", "evidence", sample)
    expected = build_simple_actor_prompt(
        sample,
        "mmlu",
        style=ReasoningStyle.EVIDENCE,
    )

    assert prompt == expected
    assert prompt.count("You are Actor-evidence") == 1
    assert prompt.count("The final result is <answer>.") == 1


def test_society_critic_dpo_ignores_stale_stored_prompt(monkeypatch, tmp_path):
    from src.society import society_trainer

    captured = {}

    def fake_detect_model_type(model_name):
        return "qwen2.5"

    def fake_train_dpo(**kwargs):
        captured["dataset"] = kwargs["preference_dataset"]
        return str(tmp_path / "adapter")

    monkeypatch.setattr(
        "src.utils.model_utils.detect_model_type",
        fake_detect_model_type,
    )
    monkeypatch.setattr("src.training.dpo_trainer.train_dpo", fake_train_dpo)

    society_trainer._run_dpo_training(
        model_name="/models/base",
        preference_pairs=[{
            "sample": {
                "question": "Q",
                "choices": ["A", "B", "C", "D"],
                "answer": "A",
                "task_type": "multiple_choice",
            },
            "prompt": "/no_think\nSTALE PROMPT",
            "actor_response": "Wrong path.\nThe final result is B.",
            "chosen": "Helpful critique.\n\nJudgement:\nAnswer correct: no\nSuggested answer: A\nConfidence: 0.9",
            "rejected": "Bad critique.\n\nJudgement:\nAnswer correct: yes\nSuggested answer: B\nConfidence: 0.9",
            "metadata": {"target_skill": "verification"},
        }],
        output_dir=str(tmp_path),
        agent_type="critic",
        agent_name="critic_verification",
        dataset_name="mmlu",
    )

    prompt = captured["dataset"][0]["prompt"]
    assert "STALE PROMPT" not in prompt
    assert "You are Critic-verification" in prompt
    assert "Actor response:" in prompt
    assert "The final result is B." in prompt


def test_society_training_fails_low_data_without_advancing_checkpoint(monkeypatch, tmp_path):
    from src.society import society_trainer
    from src.society.agent_registry import (
        AgentConfig,
        AgentRegistry,
        AgentRole,
        CriticSkill,
        ReasoningStyle,
    )

    monkeypatch.setattr(society_trainer, "_generate_critic_pairs_pairwise", lambda **kwargs: [])
    monkeypatch.setattr(society_trainer, "_cleanup_gpu", lambda: None)

    registry = AgentRegistry(base_model_path="/models/base")
    registry.register(AgentConfig(
        name="actor_direct",
        role=AgentRole.ACTOR,
        model_path="/models/base",
        reasoning_style=ReasoningStyle.DIRECT,
    ))
    registry.register(AgentConfig(
        name="critic_reasoning",
        role=AgentRole.CRITIC,
        model_path="/models/base",
        error_specialty=CriticSkill.REASONING,
    ))
    checkpoint_dir = tmp_path / "checkpoints"

    import pytest

    with pytest.raises(RuntimeError, match="Insufficient training pairs"):
        society_trainer.society_alternating_train(
            registry=registry,
            dataset=[{"question": "Q", "answer": "A", "task_type": "multiple_choice"}],
            dataset_name="mmlu",
            output_base_dir=str(tmp_path / "society"),
            checkpoint_dir=str(checkpoint_dir),
            num_iterations=1,
            min_pairs_per_critic=1,
            max_samples=1,
        )

    import json

    state = json.loads((checkpoint_dir / "society_training_state.json").read_text())
    assert state["iteration"] == 0
    assert state["iteration_completed"] is False
    assert state["failed_agents"] == {"critic_reasoning": "below_min_pairs_per_critic"}


def test_society_training_fails_actor_low_data_without_advancing_checkpoint(monkeypatch, tmp_path):
    from src.society import society_trainer
    from src.society.agent_registry import (
        AgentConfig,
        AgentRegistry,
        AgentRole,
        CriticSkill,
        ReasoningStyle,
    )

    critic_pair = {
        "sample": {"question": "Q", "answer": "A", "task_type": "multiple_choice"},
        "actor_response": "Wrong reasoning.\nThe final result is B.",
        "chosen": (
            "The actor chose the wrong option.\n\n"
            "Judgement:\n"
            "Answer correct: no\n"
            "Suggested answer: A\n"
            "Confidence: 0.9"
        ),
        "rejected": (
            "This critique misses the issue.\n\n"
            "Judgement:\n"
            "Answer correct: yes\n"
            "Suggested answer: B\n"
            "Confidence: 0.8"
        ),
        "metadata": {"target_skill": "reasoning", "assigned_skill": "reasoning"},
    }
    actor_pair = {
        "sample": {"question": "Q", "answer": "A", "task_type": "multiple_choice"},
        "chosen": "Some valid reasoning.\nThe final result is A.",
        "rejected": "Wrong reasoning.\nThe final result is B.",
        "metadata": {"style": "direct"},
    }

    monkeypatch.setattr(
        society_trainer,
        "_generate_critic_pairs_pairwise",
        lambda **kwargs: [critic_pair],
    )
    monkeypatch.setattr(
        society_trainer,
        "_generate_actor_pairs_pairwise",
        lambda **kwargs: [actor_pair],
    )
    monkeypatch.setattr(
        society_trainer,
        "_run_dpo_training",
        lambda **kwargs: str(tmp_path / kwargs["agent_name"]),
    )
    monkeypatch.setattr(society_trainer, "_cleanup_gpu", lambda: None)

    registry = AgentRegistry(base_model_path="/models/base")
    registry.register(AgentConfig(
        name="actor_direct",
        role=AgentRole.ACTOR,
        model_path="/models/base",
        reasoning_style=ReasoningStyle.DIRECT,
    ))
    registry.register(AgentConfig(
        name="critic_reasoning",
        role=AgentRole.CRITIC,
        model_path="/models/base",
        error_specialty=CriticSkill.REASONING,
    ))
    checkpoint_dir = tmp_path / "checkpoints"

    import pytest

    with pytest.raises(RuntimeError, match="Insufficient training pairs for actor_direct"):
        society_trainer.society_alternating_train(
            registry=registry,
            dataset=[{"question": "Q", "answer": "A", "task_type": "multiple_choice"}],
            dataset_name="mmlu",
            output_base_dir=str(tmp_path / "society"),
            checkpoint_dir=str(checkpoint_dir),
            num_iterations=1,
            min_pairs_per_critic=1,
            min_pairs_per_actor=2,
            max_samples=1,
        )

    import json

    state = json.loads((checkpoint_dir / "society_training_state.json").read_text())
    assert state["iteration"] == 0
    assert state["iteration_completed"] is False
    assert state["failed_agents"] == {"actor_direct": "below_min_pairs_per_actor"}
    assert state["phase_done"]["phase_A"] == ["critic_reasoning"]


def test_society_actor_acceptance_allows_prompted_fallback_but_requires_quality(monkeypatch):
    from src.society import society_trainer
    from src.society.agent_registry import (
        AgentConfig,
        AgentRegistry,
        AgentRole,
        CriticSkill,
        ReasoningStyle,
    )
    from src.society.data_classifier import ReasoningStyleResult

    sample = {
        "question": "Q",
        "choices": ["correct", "wrong", "wrong", "wrong"],
        "answer": "A",
        "task_type": "multiple_choice",
    }
    raw_pairs = [
        {
            "sample": sample,
            "actor_candidate": {
                "chosen": "The question states the key evidence.\nThe final result is A.",
                "rejected": "Wrong reasoning.\nThe final result is B.",
            },
            "critic_candidate": {"chosen": "critic chosen", "rejected": "critic rejected"},
            "comparison": {"delta": 1.0, "mode": "towards_correct"},
            "rollout_scores": {"natural": 0.0, "guided_correct": 1.0, "guided_wrong": 0.0},
        },
        {
            "sample": sample,
            "actor_candidate": {
                "chosen": (
                    "Low confidence evidence still names the relevant clue from the question.\n"
                    "The final result is A."
                ),
                "rejected": "Wrong reasoning.\nThe final result is B.",
            },
            "critic_candidate": {"chosen": "critic chosen", "rejected": "critic rejected"},
            "comparison": {"delta": 1.0, "mode": "towards_correct"},
            "rollout_scores": {"natural": 0.0, "guided_correct": 1.0, "guided_wrong": 0.0},
        },
        {
            "sample": sample,
            "actor_candidate": {
                "chosen": "A",
                "rejected": "Wrong reasoning.\nThe final result is B.",
            },
            "critic_candidate": {"chosen": "critic chosen", "rejected": "critic rejected"},
            "comparison": {"delta": 1.0, "mode": "towards_correct"},
            "rollout_scores": {"natural": 0.0, "guided_correct": 1.0, "guided_wrong": 0.0},
        },
        {
            "sample": sample,
            "actor_candidate": {
                "chosen": "Evidence but wrong.\nThe final result is B.",
                "rejected": "Wrong reasoning.\nThe final result is C.",
            },
            "critic_candidate": {"chosen": "critic chosen", "rejected": "critic rejected"},
            "comparison": {"delta": 1.0, "mode": "towards_correct"},
            "rollout_scores": {"natural": 0.0, "guided_correct": 1.0, "guided_wrong": 0.0},
        },
        {
            "sample": sample,
            "actor_candidate": {
                "chosen": "Direct.\nThe final result is A.",
                "rejected": "Wrong reasoning.\nThe final result is B.",
            },
            "critic_candidate": {"chosen": "critic chosen", "rejected": "critic rejected"},
            "comparison": {"delta": 1.0, "mode": "towards_correct"},
            "rollout_scores": {"natural": 0.0, "guided_correct": 1.0, "guided_wrong": 0.0},
        },
    ]

    class FakeActorAdapter:
        def generate(self, prompts, **kwargs):
            return ["actor"] * len(prompts)

    class FakeCriticAdapter:
        def generate(self, prompts, **kwargs):
            return ["critic"] * len(prompts)

    class FakeEngine:
        def cleanup(self):
            pass

    def fake_build_lora_adapters(engine, agents):
        return {
            "actor_evidence": FakeActorAdapter(),
            "critic_reasoning": FakeCriticAdapter(),
        }

    def fake_deliberate_batch(*args, **kwargs):
        return [[{
            "round": 0,
            "actor_prompt": "prompt",
            "actor_response": "Natural.\nThe final result is A.",
            "critic_prompt": "critic prompt",
            "critic_response": "critic",
        }, {
            "round": 1,
            "actor_prompt": "prompt",
            "actor_response": "Natural.\nThe final result is A.",
            "critic_prompt": "critic prompt",
            "critic_response": "critic",
        }]]

    def fake_guided_pairs(*args, **kwargs):
        return raw_pairs

    def fake_classify_reasoning_style(response, **kwargs):
        if "Direct" in response:
            return ReasoningStyleResult(
                primary_style=ReasoningStyle.DIRECT,
                secondary_styles=[],
                format_status="valid",
                confidence=0.9,
            )
        if response.strip() == "A":
            return ReasoningStyleResult(
                primary_style=ReasoningStyle.EVIDENCE,
                secondary_styles=[],
                format_status="answer_only",
                confidence=0.95,
            )
        if "Low confidence" in response:
            return ReasoningStyleResult(
                primary_style=ReasoningStyle.EVIDENCE,
                secondary_styles=[],
                format_status="valid",
                confidence=0.4,
            )
        return ReasoningStyleResult(
            primary_style=ReasoningStyle.EVIDENCE,
            secondary_styles=[],
            format_status="valid",
            confidence=0.9,
        )

    monkeypatch.setattr(
        "src.inference.vllm_server.VLLMInference",
        lambda *args, **kwargs: FakeEngine(),
    )
    monkeypatch.setattr(
        society_trainer,
        "_build_lora_adapters",
        fake_build_lora_adapters,
    )
    monkeypatch.setattr(
        society_trainer,
        "run_pairwise_deliberation_batch",
        fake_deliberate_batch,
    )
    monkeypatch.setattr(
        society_trainer,
        "build_guided_rollout_pairs",
        fake_guided_pairs,
    )
    monkeypatch.setattr(
        society_trainer,
        "classify_reasoning_style",
        fake_classify_reasoning_style,
    )
    monkeypatch.setattr(society_trainer, "_cleanup_gpu", lambda: None)

    registry = AgentRegistry(base_model_path="/models/base")
    actor = AgentConfig(
        name="actor_evidence",
        role=AgentRole.ACTOR,
        model_path="/models/base",
        reasoning_style=ReasoningStyle.EVIDENCE,
    )
    critic = AgentConfig(
        name="critic_reasoning",
        role=AgentRole.CRITIC,
        model_path="/models/base",
        error_specialty=CriticSkill.REASONING,
    )

    pairs = society_trainer._generate_actor_pairs_pairwise(
        actor=actor,
        critics=[critic],
        dataset=[sample],
        dataset_name="mmlu",
        registry=registry,
        num_rounds=2,
        num_simulations=1,
        max_tokens=64,
        reward_threshold=0.0,
        max_samples=1,
        seed=42,
        device=0,
        dtype="bfloat16",
        gpu_memory_utilization=0.1,
        max_model_len=512,
        strict_classification=True,
        min_style_confidence=0.65,
    )

    assert len(pairs) == 2
    chosen_texts = {pair["chosen"] for pair in pairs}
    assert "The question states the key evidence.\nThe final result is A." in chosen_texts
    assert (
        "Low confidence evidence still names the relevant clue from the question.\n"
        "The final result is A."
    ) in chosen_texts
    assert "A" not in chosen_texts
    assert all(pair["metadata"]["prompted_style"] == "evidence" for pair in pairs)
    assert all(pair["metadata"]["classified_style"] == "evidence" for pair in pairs)
    assert all("format_status" not in pair["metadata"] for pair in pairs)
    assert {pair["metadata"]["pair_source"] for pair in pairs} == {
        "strict",
        "prompted_fallback",
    }
    assert all(pair["metadata"]["is_correct"] is True for pair in pairs)


def test_society_actor_pair_generation_allows_fallback_without_strict_pair(monkeypatch):
    from src.society import society_trainer
    from src.society.agent_registry import (
        AgentConfig,
        AgentRegistry,
        AgentRole,
        CriticSkill,
        ReasoningStyle,
    )
    from src.society.data_classifier import ReasoningStyleResult

    sample = {
        "question": "Q",
        "choices": ["correct", "wrong", "wrong", "wrong"],
        "answer": "A",
        "task_type": "multiple_choice",
    }
    raw_pairs = [
        {
            "sample": sample,
            "actor_candidate": {
                "chosen": "Prompted evidence, but confidence is low.\nThe final result is A.",
                "rejected": "Wrong reasoning.\nThe final result is B.",
            },
            "critic_candidate": {"chosen": "critic chosen", "rejected": "critic rejected"},
            "comparison": {"delta": 1.0, "mode": "towards_correct"},
            "rollout_scores": {"natural": 0.0, "guided_correct": 1.0, "guided_wrong": 0.0},
        },
    ]

    class FakeActorAdapter:
        def generate(self, prompts, **kwargs):
            return ["actor"] * len(prompts)

    class FakeCriticAdapter:
        def generate(self, prompts, **kwargs):
            return ["critic"] * len(prompts)

    class FakeEngine:
        def cleanup(self):
            pass

    monkeypatch.setattr(
        "src.inference.vllm_server.VLLMInference",
        lambda *args, **kwargs: FakeEngine(),
    )
    monkeypatch.setattr(
        society_trainer,
        "_build_lora_adapters",
        lambda engine, agents: {
            "actor_evidence": FakeActorAdapter(),
            "critic_reasoning": FakeCriticAdapter(),
        },
    )
    monkeypatch.setattr(
        society_trainer,
        "run_pairwise_deliberation_batch",
        lambda *args, **kwargs: [[{
            "round": 0,
            "actor_response": "Natural.\nThe final result is A.",
            "critic_response": "critic",
        }]],
    )
    monkeypatch.setattr(
        society_trainer,
        "build_guided_rollout_pairs",
        lambda *args, **kwargs: raw_pairs,
    )
    monkeypatch.setattr(
        society_trainer,
        "classify_reasoning_style",
        lambda response, **kwargs: ReasoningStyleResult(
            primary_style=ReasoningStyle.EVIDENCE,
            secondary_styles=[],
            format_status="valid",
            confidence=0.4,
        ),
    )
    monkeypatch.setattr(society_trainer, "_cleanup_gpu", lambda: None)

    registry = AgentRegistry(base_model_path="/models/base")
    actor = AgentConfig(
        name="actor_evidence",
        role=AgentRole.ACTOR,
        model_path="/models/base",
        reasoning_style=ReasoningStyle.EVIDENCE,
    )
    critic = AgentConfig(
        name="critic_reasoning",
        role=AgentRole.CRITIC,
        model_path="/models/base",
        error_specialty=CriticSkill.REASONING,
    )

    pairs = society_trainer._generate_actor_pairs_pairwise(
        actor=actor,
        critics=[critic],
        dataset=[sample],
        dataset_name="mmlu",
        registry=registry,
        num_rounds=1,
        num_simulations=1,
        max_tokens=64,
        reward_threshold=0.0,
        max_samples=1,
        seed=42,
        device=0,
        dtype="bfloat16",
        gpu_memory_utilization=0.1,
        max_model_len=512,
        strict_classification=True,
        min_style_confidence=0.65,
        max_fallback_ratio=1.0,
    )

    assert len(pairs) == 1
    assert pairs[0]["metadata"]["pair_source"] == "prompted_fallback"


def test_actor_training_summary_reports_strict_and_fallback_ratios():
    from src.society.society_trainer import summarize_actor_training_pairs

    pairs = [
        {
            "sample": {"question": "Q1"},
            "chosen": "Reasoning.\nThe final result is A.",
            "rejected": "Wrong.\nThe final result is B.",
            "metadata": {
                "pair_source": "strict",
                "style_verified": True,
                "style_match": True,
                "classified_style": "evidence",
            },
        },
        {
            "sample": {"question": "Q2"},
            "chosen": "Prompted evidence.\nThe final result is A.",
            "rejected": "Wrong.\nThe final result is B.",
            "metadata": {
                "pair_source": "prompted_fallback",
                "style_verified": False,
                "style_match": True,
                "classified_style": "evidence",
            },
        },
    ]

    summary = summarize_actor_training_pairs(pairs)

    assert summary["sample_count"] == 2
    assert summary["pair_source_counts"] == {
        "strict": 1,
        "prompted_fallback": 1,
    }
    assert summary["pair_source_ratios"] == {
        "strict": 0.5,
        "prompted_fallback": 0.5,
    }
    assert summary["style_verified_count"] == 1
