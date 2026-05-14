"""Tests for Actor SFT candidate trajectory generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch


def _load_bootstrap_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    module_path = scripts_dir / "07_bootstrap_actors.py"
    spec = importlib.util.spec_from_file_location("bootstrap_actors", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        return [f"response-{len(self.calls)}-{i}" for i in range(len(prompts))]


def test_style_prompt_contains_style_guidance_and_contract():
    bootstrap = _load_bootstrap_module()
    sample = {
        "question": "What is 2+2?",
        "choices": ["3", "4", "5", "6"],
        "task_type": "multiple_choice",
    }

    prompt = bootstrap.build_style_prompt(
        sample,
        "mmlu",
        bootstrap.ReasoningStyle.EVIDENCE,
        temperature=1.0,
        generation_index=2,
    )

    assert prompt.count("Actor-evidence") == 1
    assert "key facts, definitions" in prompt
    assert "independent SFT candidate generation attempt 3" in prompt
    assert "temperature 1" in prompt
    assert prompt.count("The final result is <answer>.") == 1
    assert "FINAL_ANSWER" not in prompt


def test_style_temperature_generation_batches_by_temperature():
    bootstrap = _load_bootstrap_module()
    model = FakeModel()
    args = type("Args", (), {
        "dataset": "mmlu",
        "temperatures": [0.4, 0.7],
        "generations_per_temperature": 2,
        "seed": 42,
        "max_tokens": 32,
        "top_p": 0.9,
        "source_splits": ["dev", "validation"],
        "subject_balanced": True,
        "min_subjects": 1,
        "max_samples_per_subject": 20,
        "max_samples": None,
        "sampling": None,
        "mmlu_load_mode": "by_subject",
        "model_name": "model",
    })()
    styles = [
        bootstrap.ReasoningStyle.EVIDENCE,
        bootstrap.ReasoningStyle.DIRECT,
        bootstrap.ReasoningStyle.ELIMINATION,
    ]
    batch_entries = [
        (0, {
            "question": "Q0",
            "choices": ["A", "B", "C", "D"],
            "answer": "A",
            "task_type": "multiple_choice",
            "source_split": "dev",
            "subject": "abstract_algebra",
        }),
        (1, {
            "question": "Q1",
            "choices": ["A", "B", "C", "D"],
            "answer": "B",
            "task_type": "multiple_choice",
            "source_split": "validation",
            "subject": "anatomy",
        }),
    ]

    with patch(
        "src.algorithms.reward.extract_answer",
        side_effect=lambda response, task_type: response,
    ):
        records = bootstrap.generate_batch(model, batch_entries, args, styles)

    assert len(model.calls) == 2
    assert [call[1]["temperature"] for call in model.calls] == [0.4, 0.7]
    assert all(len(call[0]) == 12 for call in model.calls)
    assert model.calls[0][1]["seed"] == 42
    assert model.calls[1][1]["seed"] == 100042
    assert all("The final result is <answer>." in prompt for call in model.calls for prompt in call[0])

    assert len(records) == 2
    first = records[0]
    assert first["sample_id"] == "mmlu_0"
    assert first["source_split"] == "dev"
    assert first["subject"] == "abstract_algebra"
    assert first["debate_rounds"] == []
    assert first["metadata"]["schema_version"] == 4
    assert first["metadata"]["generation_mode"] == "actor_sft_candidates"
    assert first["metadata"]["temperatures"] == [0.4, 0.7]
    assert len(first["initial_responses"]) == 12

    response_ids = [r["response_id"] for r in first["initial_responses"]]
    assert "mmlu_0_evidence_t0p4_g0" in response_ids
    assert "mmlu_0_direct_t0p7_g1" in response_ids
    assert "mmlu_0_elimination_t0p7_g1" in response_ids
    assert {r["temperature"] for r in first["initial_responses"]} == {0.4, 0.7}
    assert {r["generation_index"] for r in first["initial_responses"]} == {0, 1}


def test_subject_balanced_selection_preserves_coverage_under_total_cap():
    bootstrap = _load_bootstrap_module()
    data = {
        "dev": [
            {
                "question": f"Q{s}_{i}",
                "choices": ["A", "B"],
                "answer": "A",
                "subject": f"subject_{s}",
                "source_split": "dev",
                "source_index": i,
            }
            for s in range(3)
            for i in range(3)
        ],
    }

    selected = bootstrap.select_subject_balanced_samples(
        data,
        ["dev"],
        subject_balanced=True,
        min_subjects=3,
        max_samples_per_subject=3,
        max_samples=6,
        seed=42,
    )

    subjects = {sample["subject"] for _, sample in selected}
    assert subjects == {"subject_0", "subject_1", "subject_2"}
    assert len(selected) == 6


def test_subject_coverage_check_fails_fast():
    bootstrap = _load_bootstrap_module()
    data = {
        "dev": [{
            "question": "Q",
            "choices": ["A", "B"],
            "answer": "A",
            "subject": "only_subject",
            "source_split": "dev",
            "source_index": 0,
        }],
    }

    try:
        bootstrap.select_subject_balanced_samples(
            data,
            ["dev"],
            subject_balanced=True,
            min_subjects=2,
            max_samples_per_subject=3,
            max_samples=None,
            seed=42,
        )
    except ValueError as exc:
        assert "Subject coverage 1 is below min_subjects=2" in str(exc)
    else:
        raise AssertionError("Expected insufficient subject coverage to fail")


def test_generation_result_count_mismatch_fails_fast():
    bootstrap = _load_bootstrap_module()

    try:
        bootstrap.coerce_generation_results(["one"], expected=2)
    except ValueError as exc:
        assert "Generated 1 responses for 2 prompts" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_existing_bootstrap_rejects_stale_generation(tmp_path):
    bootstrap = _load_bootstrap_module()
    output_file = tmp_path / "trajectories.jsonl"
    output_file.write_text(json.dumps({
        "sample_id": "mmlu_0",
        "metadata": {
            "schema_version": 3,
            "generation_mode": "style_prompted",
        },
    }) + "\n")
    args = type("Args", (), {
        "dataset": "mmlu",
        "temperatures": [0.4, 0.7],
        "generations_per_temperature": 1,
        "top_p": 0.9,
        "max_tokens": 256,
        "source_splits": ["dev"],
        "subject_balanced": True,
        "min_subjects": 1,
        "max_samples_per_subject": 20,
        "max_samples": None,
        "sampling": None,
        "mmlu_load_mode": "by_subject",
        "model_name": "model",
        "seed": 42,
    })()
    styles = [
        bootstrap.ReasoningStyle.EVIDENCE,
        bootstrap.ReasoningStyle.DIRECT,
        bootstrap.ReasoningStyle.ELIMINATION,
    ]

    try:
        bootstrap.existing_sample_ids(output_file, args, styles)
    except RuntimeError as exc:
        assert "actor-sft candidate configuration" in str(exc)
    else:
        raise AssertionError("Expected stale bootstrap output to fail fast")


def test_existing_bootstrap_resumes_matching_actor_sft_output(tmp_path):
    bootstrap = _load_bootstrap_module()
    args = type("Args", (), {
        "dataset": "mmlu",
        "temperatures": [0.4, 0.7],
        "generations_per_temperature": 1,
        "top_p": 0.9,
        "max_tokens": 256,
        "source_splits": ["dev"],
        "subject_balanced": True,
        "min_subjects": 1,
        "max_samples_per_subject": 20,
        "max_samples": None,
        "sampling": None,
        "mmlu_load_mode": "by_subject",
        "model_name": "model",
        "seed": 42,
    })()
    styles = [
        bootstrap.ReasoningStyle.EVIDENCE,
        bootstrap.ReasoningStyle.DIRECT,
        bootstrap.ReasoningStyle.ELIMINATION,
    ]
    output_file = tmp_path / "trajectories.jsonl"
    output_file.write_text(json.dumps({
        "sample_id": "mmlu_0",
        "metadata": bootstrap.expected_bootstrap_metadata(args, styles),
    }) + "\n")

    assert bootstrap.existing_sample_ids(output_file, args, styles) == {"mmlu_0"}
