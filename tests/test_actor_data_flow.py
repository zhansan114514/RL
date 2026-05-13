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
                "trainable_for_actor": True,
                "style_verified": True,
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
                "trainable_for_actor": True,
                "style_verified": True,
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
                "trainable_for_actor": False,
                "style_verified": True,
                "accepted_for_actor": False,
            },
        ],
    }]
    args = type("Args", (), {
        "pair_mix": {"correctness": 0.7, "style": 0.3},
        "target_pairs_per_actor": 2,
        "max_pairs_per_actor": 10,
        "max_pairs_per_sample": 10,
    })()

    by_sample = diversify.build_response_index(classified_results)
    pairs, metrics = diversify.build_preference_pairs_for_style(
        by_sample,
        "evidence",
        args,
    )

    assert {p["metadata"]["pair_type"] for p in pairs} == {"correctness", "style"}
    assert all(p["metadata"]["chosen_response_id"] == "chosen" for p in pairs)
    assert metrics["candidate_counts"]["correctness"] == 1
    assert metrics["candidate_counts"]["style"] == 2
    assert metrics["candidate_counts"]["strict_style_correct"] == 1
    assert metrics["candidate_counts"]["prompted_style_correct_fallback"] == 0
    assert metrics["candidate_counts"]["selected_pairs"] == 2


def test_actor_pair_builder_caps_prompted_style_fallback():
    diversify = _load_script("09_diversify_actors.py")
    sample = {
        "question": "Q",
        "answer": "A",
        "task_type": "multiple_choice",
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
            "accepted_for_actor": True,
        },
        {
            "response_id": "fallback_1",
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
            "accepted_for_actor": True,
        },
        {
            "response_id": "fallback_2",
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
            "accepted_for_actor": True,
        },
        {
            "response_id": "wrong",
            "response": "Wrong.\nThe final result is B.",
            "answer": "B",
            "is_correct": False,
            "format_status": "valid",
            "prompted_style": "evidence",
            "accepted_for_actor": False,
        },
    ]
    classified_results = [{
        "sample_id": "mmlu_0",
        "sample": sample,
        "per_response_labels": labels,
    }]
    args = type("Args", (), {
        "pair_mix": {"correctness": 1.0, "style": 0.0},
        "target_pairs_per_actor": 8,
        "max_pairs_per_actor": 8,
        "max_pairs_per_sample": 8,
        "max_prompted_fallback_ratio": 0.5,
    })()

    pairs, metrics = diversify.build_preference_pairs_for_style(
        diversify.build_response_index(classified_results),
        "evidence",
        args,
    )

    assert metrics["fallback_selection"]["strict_selected"] == 1
    assert metrics["fallback_selection"]["prompted_fallback_candidates"] == 2
    assert metrics["fallback_selection"]["prompted_fallback_selected"] == 1
    assert metrics["fallback_selection"]["prompted_fallback_dropped"] == 1
    assert sum(
        1 for pair in pairs
        if pair["metadata"]["pair_source"] == "prompted_fallback"
    ) == 1


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


def test_actor_diversification_dpo_prompt_matches_simple_actor_prompt():
    diversify = _load_script("09_diversify_actors.py")
    from src.prompts.prompt_builder import build_simple_actor_prompt
    from src.society.agent_registry import ReasoningStyle

    sample = {
        "question": "Q",
        "choices": ["A", "B", "C", "D"],
        "answer": "A",
        "task_type": "multiple_choice",
    }

    prompt = diversify._actor_training_prompt("mmlu", "evidence", sample)
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


def test_society_actor_pair_generation_requires_strict_pair_for_fallback(monkeypatch):
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

    assert pairs == []
