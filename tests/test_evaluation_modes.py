"""Tests for config-driven evaluation GPU modes."""

from __future__ import annotations

import importlib.util
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest


def _load_society_evaluate_module():
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    module_path = root / "scripts" / "12_society_evaluate.py"
    spec = importlib.util.spec_from_file_location("society_evaluate", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_dual_gpu_evaluation_mode():
    evaluate = _load_society_evaluate_module()
    args = Namespace(
        evaluation_mode="dual_gpu",
        evaluation_modes={
            "single_gpu": {
                "devices": [4],
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.65,
            },
            "dual_gpu": {
                "devices": [4, 5],
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.80,
            },
        },
    )

    runtime = evaluate.resolve_evaluation_runtime(args)

    assert runtime.mode == "dual_gpu"
    assert runtime.devices == [4, 5]
    assert runtime.tensor_parallel_size == 2
    assert runtime.gpu_memory_utilization == 0.80


def test_evaluation_mode_rejects_mismatched_tensor_parallel_size():
    evaluate = _load_society_evaluate_module()
    args = Namespace(
        evaluation_mode="dual_gpu",
        evaluation_modes={
            "dual_gpu": {
                "devices": [4, 5],
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.80,
            },
        },
    )

    with pytest.raises(ValueError, match="tensor_parallel_size must match"):
        evaluate.resolve_evaluation_runtime(args)


def test_vllm_inference_accepts_cuda_device_list():
    from src.inference.vllm_server import VLLMInference

    engine = VLLMInference(
        "test-model",
        cuda_device=[4, 5],
        gpu_memory_utilization=0.8,
        max_model_len=4096,
    )

    assert engine._cuda_devices == (4, 5)
    assert engine._init_kwargs["tensor_parallel_size"] == 2
    assert engine._init_kwargs["gpu_memory_utilization"] == 0.8


def test_vllm_sets_cuda_visible_devices_before_loading(monkeypatch):
    from src.inference import vllm_server
    from src.inference.vllm_server import VLLMInference

    observed_cuda_visible_devices = {}

    class FakeLLM:
        def __init__(self, model_name, **kwargs):
            observed_cuda_visible_devices["value"] = (
                vllm_server.os.environ.get("CUDA_VISIBLE_DEVICES")
            )
            self.model_name = model_name
            self.kwargs = kwargs

        def get_tokenizer(self):
            return object()

    fake_vllm = types.SimpleNamespace(LLM=FakeLLM)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    engine = VLLMInference("test-model", cuda_device=[4, 5])
    engine._ensure_loaded()

    assert observed_cuda_visible_devices["value"] == "4,5"
    assert engine._init_kwargs["tensor_parallel_size"] == 2


def test_society_per_round_accuracy_does_not_use_future_final_answer(monkeypatch):
    evaluate = _load_society_evaluate_module()
    from src.society import multi_deliberation

    def fake_multi_agent_deliberate_single_gpu(**kwargs):
        round0 = types.SimpleNamespace(
            round_num=0,
            consensus_answer=None,
            consensus_confidence=0.0,
            actor_answer_sources={"actor_direct": "none"},
            routed_feedbacks={},
            raw_actor_answers={"actor_direct": None},
            actor_answers={"actor_direct": None},
            actor_format_valid={"actor_direct": False},
            actor_responses={"actor_direct": "No final decision."},
        )
        round1 = types.SimpleNamespace(
            round_num=1,
            consensus_answer="B",
            consensus_confidence=1.0,
            actor_answer_sources={"actor_direct": "strict"},
            routed_feedbacks={},
            raw_actor_answers={"actor_direct": "B"},
            actor_answers={"actor_direct": "B"},
            actor_format_valid={"actor_direct": True},
            actor_responses={"actor_direct": "FINAL_ANSWER: B"},
        )
        return types.SimpleNamespace(
            consensus_answer="B",
            consensus_confidence=1.0,
            rounds=[round0, round1],
            final_answers={"actor_direct": "B"},
        )

    monkeypatch.setattr(
        multi_deliberation,
        "multi_agent_deliberate_single_gpu",
        fake_multi_agent_deliberate_single_gpu,
    )

    result = evaluate._run_deliberation_on_samples(
        engine=object(),
        actor_configs=[],
        critic_configs=[],
        samples=[{
            "question": "Pick B.",
            "answer": "B",
            "task_type": "multiple_choice",
        }],
        dataset_name="mmlu",
        lora_paths={},
        num_rounds=2,
        max_tokens=16,
        temperature=0.0,
    )

    assert result.initial_accuracy == 0.0
    assert result.final_consensus_accuracy == 1.0
    assert result.per_round_accuracy == [0.0, 1.0]
