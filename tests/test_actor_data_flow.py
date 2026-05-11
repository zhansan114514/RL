"""Tests for style-prompted classification and actor pair construction."""

from __future__ import annotations

import importlib.util
import json
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


def test_actor_pairs_use_only_accepted_target_style_chosen():
    diversify = _load_script("09_diversify_actors.py")
    sample = {
        "question": "Q",
        "answer": "A",
        "task_type": "multiple_choice",
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
                "accepted_for_actor": True,
            },
            {
                "response_id": "wrong",
                "response": "This chooses the wrong option.\nThe final result is B.",
                "answer": "B",
                "is_correct": False,
                "format_status": "valid",
                "prompted_style": "evidence",
                "accepted_for_actor": False,
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
                "accepted_for_actor": True,
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
                "accepted_for_actor": False,
            },
        ],
    }]
    args = type("Args", (), {
        "pair_mix": {"correctness": 0.34, "style": 0.33, "format": 0.33},
        "target_pairs_per_actor": 3,
        "max_pairs_per_actor": 10,
        "max_pairs_per_sample": 10,
    })()

    by_sample = diversify.build_response_index(classified_results)
    pairs, metrics = diversify.build_preference_pairs_for_style(
        by_sample,
        "evidence",
        args,
    )

    assert {p["metadata"]["pair_type"] for p in pairs} == {
        "correctness",
        "style",
        "format",
    }
    assert all(p["metadata"]["chosen_response_id"] == "chosen" for p in pairs)
    assert metrics["candidate_counts"] == {
        "correctness": 1,
        "style": 2,
        "format": 1,
    }


def test_society_actor_style_prompt_wrapper_conditions_generation():
    from src.society.agent_registry import ReasoningStyle
    from src.society.society_trainer import StyleConditionedActorAdapter

    class FakeModel:
        def __init__(self):
            self.prompts = []

        def generate(self, prompts, **kwargs):
            self.prompts.extend(prompts)
            return ["ok" for _ in prompts]

    model = FakeModel()
    adapter = StyleConditionedActorAdapter(model, ReasoningStyle.ELIMINATION)

    adapter.generate(["Solve Q"], max_tokens=8, temperature=0.3)

    assert "You are Actor-elimination" in model.prompts[0]
    assert "Compare options and rule out" in model.prompts[0]
    assert model.prompts[0].endswith("Solve Q")


def test_society_actor_dpo_prompt_is_style_conditioned(monkeypatch, tmp_path):
    from src.society import society_trainer
    from src.society.agent_registry import ReasoningStyle

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
    assert checkpoint == str(tmp_path / "adapter")
    assert "You are Actor-direct" in prompt
    assert "shortest sufficient reasoning" in prompt
    assert "The final result is <answer>." in prompt
    assert "FINAL_ANSWER" not in prompt


def test_society_actor_acceptance_requires_style_format_confidence_and_correctness(monkeypatch):
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
            "positive": "The question states the key evidence.\nThe final result is A.",
            "negative": "Wrong reasoning.\nThe final result is B.",
            "delta": 1.0,
            "direction": "towards",
        },
        {
            "sample": sample,
            "positive": "Low confidence evidence.\nThe final result is A.",
            "negative": "Wrong reasoning.\nThe final result is B.",
            "delta": 1.0,
            "direction": "towards",
        },
        {
            "sample": sample,
            "positive": "A",
            "negative": "Wrong reasoning.\nThe final result is B.",
            "delta": 1.0,
            "direction": "towards",
        },
        {
            "sample": sample,
            "positive": "Evidence but wrong.\nThe final result is B.",
            "negative": "Wrong reasoning.\nThe final result is C.",
            "delta": 1.0,
            "direction": "towards",
        },
        {
            "sample": sample,
            "positive": "Direct.\nThe final result is A.",
            "negative": "Wrong reasoning.\nThe final result is B.",
            "delta": 1.0,
            "direction": "towards",
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
            "critic_computation": FakeCriticAdapter(),
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
        "src.algorithms.deliberation.deliberate_batch",
        fake_deliberate_batch,
    )
    monkeypatch.setattr(
        "src.algorithms.trajectory._generate_guided_pairs_for_batch",
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
        name="critic_computation",
        role=AgentRole.CRITIC,
        model_path="/models/base",
        error_specialty=CriticSkill.COMPUTATION,
    )

    pairs = society_trainer._generate_actor_pairs_algorithm1(
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

    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "The question states the key evidence.\nThe final result is A."
    assert pairs[0]["metadata"]["prompted_style"] == "evidence"
    assert pairs[0]["metadata"]["classified_style"] == "evidence"
    assert pairs[0]["metadata"]["format_status"] == "valid"
    assert pairs[0]["metadata"]["style_confidence"] == 0.9
    assert pairs[0]["metadata"]["is_correct"] is True
