"""Tests for style-prompted bootstrap trajectory generation."""

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
        return [f"response-{i}" for i in range(len(prompts))]


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
        bootstrap.ReasoningStyle.ALGEBRAIC,
        generation_index=2,
    )

    assert "Actor-algebraic" in prompt
    assert "variables, equations" in prompt
    assert "independent generation attempt 3" in prompt
    assert "FINAL_ANSWER: A or B or C or D" in prompt


def test_style_prompted_generation_batches_samples_styles_and_attempts():
    bootstrap = _load_bootstrap_module()
    model = FakeModel()
    args = type("Args", (), {
        "dataset": "mmlu",
        "generations_per_style": 2,
        "seed": 42,
        "max_tokens": 32,
        "temperature": 0.5,
        "top_p": 0.9,
    })()
    styles = [
        bootstrap.ReasoningStyle.ALGEBRAIC,
        bootstrap.ReasoningStyle.DIRECT,
        bootstrap.ReasoningStyle.BACKTRACKING,
    ]
    batch_entries = [
        (0, {"question": "Q0", "choices": ["A", "B", "C", "D"], "task_type": "multiple_choice"}),
        (1, {"question": "Q1", "choices": ["A", "B", "C", "D"], "task_type": "multiple_choice"}),
    ]

    with patch(
        "src.algorithms.reward.extract_answer",
        side_effect=lambda response, task_type: response,
    ):
        records = bootstrap.generate_batch(model, batch_entries, args, styles)

    assert len(model.calls) == 1
    prompts, kwargs = model.calls[0]
    assert len(prompts) == 12
    assert kwargs["seed"] == 42
    assert kwargs["max_tokens"] == 32
    assert kwargs["temperature"] == 0.5
    assert kwargs["top_p"] == 0.9
    assert all("FINAL_ANSWER:" in prompt for prompt in prompts)

    assert len(records) == 2
    first = records[0]
    assert first["sample_id"] == "mmlu_0"
    assert first["debate_rounds"] == []
    assert first["metadata"]["generation_mode"] == "style_prompted"
    assert first["metadata"]["reasoning_styles"] == [
        "algebraic",
        "direct",
        "backtracking",
    ]
    assert len(first["initial_responses"]) == 6

    response_ids = [r["response_id"] for r in first["initial_responses"]]
    assert response_ids == [
        "mmlu_0_algebraic_0",
        "mmlu_0_algebraic_1",
        "mmlu_0_direct_0",
        "mmlu_0_direct_1",
        "mmlu_0_backtracking_0",
        "mmlu_0_backtracking_1",
    ]
    assert [r["prompted_style"] for r in first["initial_responses"]] == [
        "algebraic",
        "algebraic",
        "direct",
        "direct",
        "backtracking",
        "backtracking",
    ]
    assert [r["agent_name"] for r in first["initial_responses"]] == [
        "actor_algebraic",
        "actor_algebraic",
        "actor_direct",
        "actor_direct",
        "actor_backtracking",
        "actor_backtracking",
    ]


def test_generation_result_count_mismatch_fails_fast():
    bootstrap = _load_bootstrap_module()

    try:
        bootstrap.coerce_generation_results(["one"], expected=2)
    except ValueError as exc:
        assert "Generated 1 responses for 2 prompts" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_existing_bootstrap_rejects_stale_natural_generation(tmp_path):
    bootstrap = _load_bootstrap_module()
    output_file = tmp_path / "trajectories.jsonl"
    output_file.write_text(json.dumps({
        "sample_id": "mmlu_0",
        "metadata": {
            "schema_version": 3,
            "generation_mode": "natural",
        },
    }) + "\n")
    args = type("Args", (), {
        "dataset": "mmlu",
        "generations_per_style": 4,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 256,
    })()
    styles = [
        bootstrap.ReasoningStyle.ALGEBRAIC,
        bootstrap.ReasoningStyle.DIRECT,
        bootstrap.ReasoningStyle.BACKTRACKING,
    ]

    try:
        bootstrap.existing_sample_ids(output_file, args, styles)
    except RuntimeError as exc:
        assert "does not match the current style-prompted" in str(exc)
    else:
        raise AssertionError("Expected stale bootstrap output to fail fast")


def test_existing_bootstrap_resumes_matching_style_prompted_output(tmp_path):
    bootstrap = _load_bootstrap_module()
    args = type("Args", (), {
        "dataset": "mmlu",
        "generations_per_style": 4,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 256,
    })()
    styles = [
        bootstrap.ReasoningStyle.ALGEBRAIC,
        bootstrap.ReasoningStyle.DIRECT,
        bootstrap.ReasoningStyle.BACKTRACKING,
    ]
    output_file = tmp_path / "trajectories.jsonl"
    output_file.write_text(json.dumps({
        "sample_id": "mmlu_0",
        "metadata": bootstrap.expected_bootstrap_metadata(args, styles),
    }) + "\n")

    assert bootstrap.existing_sample_ids(output_file, args, styles) == {"mmlu_0"}
