"""Tests for batched bootstrap trajectory generation."""

from __future__ import annotations

import importlib.util
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


def test_initial_responses_are_batched_across_samples_and_agents():
    bootstrap = _load_bootstrap_module()
    model = FakeModel()
    samples = [
        {"question": "Q0", "task_type": "math"},
        {"question": "Q1", "task_type": "math"},
    ]

    with (
        patch("src.prompts.formatter.format_prompt", side_effect=lambda *args, **kwargs: args[2]["question"]),
        patch("src.algorithms.reward.extract_answer", side_effect=lambda response, task_type: response),
    ):
        responses = bootstrap.generate_initial_responses_batch(
            model,
            samples,
            "math",
            num_agents=3,
            temperature=0.0,
            max_tokens=32,
            base_seed=123,
        )

    assert len(model.calls) == 1
    prompts, kwargs = model.calls[0]
    assert len(prompts) == 6
    assert kwargs["seed"] == 123
    assert all("FINAL_ANSWER:" in prompt for prompt in prompts)
    assert all(
        prompt.rfind("Produce an independent solution.")
        < prompt.rfind("Output format requirements:")
        for prompt in prompts
    )

    assert [r.agent_id for r in responses[0]] == [0, 1, 2]
    assert [r.response for r in responses[0]] == ["response-0", "response-1", "response-2"]
    assert [r.agent_id for r in responses[1]] == [0, 1, 2]
    assert [r.response for r in responses[1]] == ["response-3", "response-4", "response-5"]


def test_debate_round_batch_keeps_sample_contexts_separate():
    bootstrap = _load_bootstrap_module()
    model = FakeModel()
    samples = [
        {"question": "Q0", "task_type": "math"},
        {"question": "Q1", "task_type": "math"},
    ]
    previous = [
        [
            bootstrap.AgentResponse(agent_id=0, round=0, response="s0-a0", answer=""),
            bootstrap.AgentResponse(agent_id=1, round=0, response="s0-a1", answer=""),
        ],
        [
            bootstrap.AgentResponse(agent_id=0, round=0, response="s1-a0", answer=""),
            bootstrap.AgentResponse(agent_id=1, round=0, response="s1-a1", answer=""),
        ],
    ]

    def fake_format_prompt(dataset_name, prompt_type, sample, **kwargs):
        return f"{sample['question']}|{kwargs['responses']}"

    with (
        patch("src.prompts.formatter.format_prompt", side_effect=fake_format_prompt),
        patch("src.algorithms.reward.extract_answer", side_effect=lambda response, task_type: response),
    ):
        responses = bootstrap.simulate_debate_round_batch(
            model,
            samples,
            "math",
            previous,
            round_num=1,
            temperature=0.0,
            max_tokens=32,
            base_seed=123,
        )

    prompts, kwargs = model.calls[0]
    assert len(prompts) == 4
    assert kwargs["seed"] == 223
    assert all("s1-a" not in prompt for prompt in prompts[:2])
    assert all("s0-a" not in prompt for prompt in prompts[2:])
    assert all("FINAL_ANSWER:" in prompt for prompt in prompts)
    assert all(
        prompt.rfind("Revise independently after reading the debate.")
        < prompt.rfind("Output format requirements:")
        for prompt in prompts
    )

    assert [r.agent_id for r in responses[0]] == [0, 1]
    assert [r.round for r in responses[0]] == [1, 1]
    assert [r.response for r in responses[0]] == ["response-0", "response-1"]
    assert [r.agent_id for r in responses[1]] == [0, 1]
    assert [r.response for r in responses[1]] == ["response-2", "response-3"]
