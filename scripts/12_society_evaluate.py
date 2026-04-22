"""
Society evaluation and ablation experiments.

Runs A1-A5 ablation experiments and computes:
- Per-round accuracy
- Improvement rate
- Consensus accuracy
- Diversity metrics
- Wilson 95% CI

Key design: loads the base model ONCE, all ablation experiments share
the same vLLM engine and switch LoRA adapters dynamically.

Usage:
    python scripts/12_society_evaluate.py \
        --config configs/society/experiment_h100.yaml \
        --run_ablations
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import resolve_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

ALLOWED_DATASETS = ("boolq", "mmlu", "bbh", "sciq", "arc", "math", "gsm8k")
COMMON_KEYS = ("model_name", "dataset", "cache_dir", "society_dir", "output_dir", "seed", "max_samples", "device", "dtype", "gpu_memory_utilization")

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "output/society",
    "society_dir": "output/society/society",
    "output_dir": "output/society/eval",
    "num_rounds": 2,
    "max_tokens": 512,
    "temperature": 0.7,
    "device": 0,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.65,
    "aggregation": "majority_vote",
    "run_ablations": True,
    "num_samples_for_qualitative": 5,
}


@dataclass
class EvalResult:
    """Result of society evaluation."""
    initial_accuracy: float
    final_accuracy: float
    per_round_accuracy: List[float]
    improvement_rate: float
    absolute_improvement: float
    consensus_accuracy: float
    diversity_score: float
    ci_95: tuple[float, float]
    num_samples: int
    eval_time_seconds: float
    sample_details: List[Dict[str, Any]]


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Society evaluation",
    )
    parser.add_argument(
        "--config", type=str, default="configs/society/experiment_h100.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--no_ablations", action="store_true",
        help="Skip ablation experiments.",
    )
    cli_args = parser.parse_args()
    args = resolve_config(
        cli_args.config, "step06_evaluate", STEP_DEFAULTS,
        common_keys=COMMON_KEYS,
        allowed_datasets=ALLOWED_DATASETS,
    )

    if cli_args.no_ablations:
        args.run_ablations = False

    return args


def load_society_registry(society_dir: str) -> Dict[str, Any]:
    """Load society agent registry."""
    registry_file = os.path.join(society_dir, "final_agent_registry.json")

    if not os.path.exists(society_dir):
        logger.warning(f"Society directory not found: {society_dir}")
        return {}

    if not os.path.exists(registry_file):
        logger.warning(f"Registry not found: {registry_file}")
        return {}

    with open(registry_file) as f:
        return json.load(f)


def load_dataset(dataset_name: str, seed: int, max_samples: Optional[int]) -> List[Dict]:
    """Load test dataset."""
    from src.data.loader import load_dataset

    data = load_dataset(dataset_name, seed=seed)
    test_data = data.get("test", []) or data.get("validation", [])

    if max_samples:
        test_data = test_data[:max_samples]

    return test_data


def compute_diversity(responses: List[str]) -> float:
    """Compute diversity score using unique response ratio."""
    if not responses:
        return 0.0
    return len(set(responses)) / len(responses) if responses else 0.0


def _compute_ci(
    predictions: List[str],
    labels: List[str],
    task_type: str,
) -> tuple:
    """Compute accuracy and Wilson confidence interval margin."""
    from src.algorithms.reward import math_answers_equal

    correct = 0
    for pred, label in zip(predictions, labels):
        if task_type == "math":
            if math_answers_equal(pred or "", label):
                correct += 1
        else:
            if (pred or "").upper() == label.upper():
                correct += 1

    n = len(labels)
    accuracy = correct / n if n > 0 else 0.0

    z = 1.96
    if n == 0:
        return accuracy, 0.0
    denom = 1 + z**2 / n
    margin = z * math.sqrt((accuracy * (1 - accuracy) + z**2 / (4 * n)) / n) / denom
    return accuracy, margin


# ============================================================
# Agent config helpers
# ============================================================

def _build_agent_configs(
    registry: Dict[str, Any],
    actor_names: Optional[List[str]] = None,
    critic_names: Optional[List[str]] = None,
):
    """Build AgentConfig lists and LoRA paths from registry.

    Returns (actor_configs, critic_configs, lora_paths).
    """
    from src.society.agent_registry import AgentConfig, AgentRole, ReasoningStyle, ErrorType

    actors_info = registry.get("actors", {})
    critics_info = registry.get("critics", {})
    base_model = registry.get("training_config", {}).get("base_model", "Qwen/Qwen2.5-7B-Instruct")

    # Filter by names if specified
    if actor_names is not None:
        actors_info = {k: v for k, v in actors_info.items() if k in actor_names}
    if critic_names is not None:
        critics_info = {k: v for k, v in critics_info.items() if k in critic_names}

    actor_configs = []
    for name, info in actors_info.items():
        style_str = name.replace("actor_", "")
        try:
            style = ReasoningStyle(style_str.upper())
        except ValueError:
            style = ReasoningStyle.ALGEBRAIC
        actor_configs.append(AgentConfig(
            name=name,
            role=AgentRole.ACTOR,
            reasoning_style=style,
            model_path=base_model,
            lora_path=info.get("model_path", ""),
            system_prompt="",
        ))

    critic_configs = []
    for name, info in critics_info.items():
        error_str = name.replace("critic_", "")
        try:
            error = ErrorType(error_str.upper())
        except ValueError:
            error = ErrorType.LOGIC
        critic_configs.append(AgentConfig(
            name=name,
            role=AgentRole.CRITIC,
            error_specialty=error,
            model_path=base_model,
            lora_path=info.get("model_path", ""),
            system_prompt="",
        ))

    lora_paths = {}
    for name, info in actors_info.items():
        path = info.get("model_path", "")
        if path:
            lora_paths[name] = path
    for name, info in critics_info.items():
        path = info.get("model_path", "")
        if path:
            lora_paths[name] = path

    return actor_configs, critic_configs, lora_paths


def _run_deliberation_on_samples(
    engine,
    actor_configs,
    critic_configs,
    samples: List[Dict],
    dataset_name: str,
    lora_paths: Dict[str, str],
    num_rounds: int,
    max_tokens: int,
    temperature: float,
    router_top_k: int = 2,
    router_uniform: bool = False,
) -> EvalResult:
    """Run deliberation on samples with a shared vLLM engine. No model loading."""
    from src.society.multi_deliberation import multi_agent_deliberate_single_gpu
    from src.society.router import CriticRouter

    # Create router with the specified configuration
    router = CriticRouter(top_k=router_top_k, uniform_weights=router_uniform)

    start_time = time.time()
    all_final_answers = []
    all_initial_answers = []
    all_responses = []
    details = []

    for si, sample in enumerate(samples):
        if (si + 1) % 5 == 0 or si == 0:
            logger.info(f"    Sample {si + 1}/{len(samples)}")

        result = multi_agent_deliberate_single_gpu(
            inference_engine=engine,
            actors=actor_configs,
            critics=critic_configs,
            sample=sample,
            dataset_name=dataset_name,
            lora_paths=lora_paths,
            num_rounds=num_rounds,
            max_tokens=max_tokens,
            temperature=temperature,
            router=router,
        )

        final_answer = result.consensus_answer or ""
        all_final_answers.append(final_answer)
        all_responses.append(final_answer)

        # Compute initial answer as majority vote across all actors (not just first)
        initial_answer = final_answer
        if result.rounds:
            first_round = result.rounds[0]
            init_answers = [a for a in first_round.actor_answers.values() if a is not None]
            if init_answers:
                counter = Counter(init_answers)
                initial_answer = counter.most_common(1)[0][0]
            else:
                initial_answer = final_answer
        all_initial_answers.append(initial_answer)

        details.append({
            "question": sample.get("question", ""),
            "final_answer": final_answer,
            "confidence": result.consensus_confidence,
            "ground_truth": sample.get("answer", ""),
        })

    labels = [s.get("answer", "") for s in samples]
    task_type = samples[0].get("task_type", "math") if samples else "math"

    initial_acc, _ = _compute_ci(all_initial_answers, labels, task_type)
    final_acc, ci_margin = _compute_ci(all_final_answers, labels, task_type)
    ci_95 = (max(0, final_acc - ci_margin), min(1, final_acc + ci_margin))

    return EvalResult(
        initial_accuracy=initial_acc,
        final_accuracy=final_acc,
        per_round_accuracy=[initial_acc, final_acc],
        improvement_rate=(final_acc - initial_acc) / initial_acc if initial_acc > 0 else 0.0,
        absolute_improvement=final_acc - initial_acc,
        consensus_accuracy=final_acc,
        diversity_score=compute_diversity(all_responses),
        ci_95=ci_95,
        num_samples=len(samples),
        eval_time_seconds=time.time() - start_time,
        sample_details=details,
    )


# ============================================================
# Main evaluation with shared engine
# ============================================================

def run_all_evaluations(
    registry: Dict[str, Any],
    samples: List[Dict],
    dataset_name: str,
    num_rounds: int,
    max_tokens: int,
    temperature: float,
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
    run_ablations: bool,
) -> Dict[str, EvalResult]:
    """Load base model ONCE, run all evaluations sharing the same engine."""
    from src.inference.vllm_server import VLLMInference

    base_model = registry.get("training_config", {}).get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    all_actor_names = list(registry.get("actors", {}).keys())
    all_critic_names = list(registry.get("critics", {}).keys())

    results: Dict[str, EvalResult] = {}

    logger.info(f"Loading base model ONCE: {base_model}")
    engine = VLLMInference(
        base_model,
        cuda_device=device,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
    )

    try:
        if run_ablations:
            # A1: 1 Actor + 1 Critic (baseline ACC-Collab)
            logger.info("[A1] 1 Actor + 1 Critic (baseline)...")
            a_configs, c_configs, lora = _build_agent_configs(
                registry,
                actor_names=all_actor_names[:1],
                critic_names=all_critic_names[:1],
            )
            results["A1_baseline"] = _run_deliberation_on_samples(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=1,
            )
            logger.info(f"  A1: initial={results['A1_baseline'].initial_accuracy:.3f} final={results['A1_baseline'].final_accuracy:.3f}")

            # A2: 3 Actors + 1 Critic (Actor diversity only)
            logger.info("[A2] 3 Actors + 1 Critic (Actor diversity)...")
            a_configs, c_configs, lora = _build_agent_configs(
                registry,
                actor_names=all_actor_names[:3],
                critic_names=all_critic_names[:1],
            )
            results["A2_actor_diversity"] = _run_deliberation_on_samples(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=1,
            )
            logger.info(f"  A2: initial={results['A2_actor_diversity'].initial_accuracy:.3f} final={results['A2_actor_diversity'].final_accuracy:.3f}")

            # A3: 1 Actor + 4 Critics + Router (Critic specialization only)
            logger.info("[A3] 1 Actor + 4 Critics (Critic specialization)...")
            a_configs, c_configs, lora = _build_agent_configs(
                registry,
                actor_names=all_actor_names[:1],
                critic_names=all_critic_names,
            )
            results["A3_critic_specialization"] = _run_deliberation_on_samples(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=2,
            )
            logger.info(f"  A3: initial={results['A3_critic_specialization'].initial_accuracy:.3f} final={results['A3_critic_specialization'].final_accuracy:.3f}")

            # A4: 3 Actors + 4 Critics, UNIFORM weights (no routing)
            # Key difference from A5: uniform_weights=True means all critics
            # contribute equally (no softmax confidence gating)
            logger.info("[A4] 3 Actors + 4 Critics (no routing, uniform weights)...")
            a_configs, c_configs, lora = _build_agent_configs(
                registry,
                actor_names=all_actor_names[:3],
                critic_names=all_critic_names,
            )
            results["A4_no_routing"] = _run_deliberation_on_samples(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=4,       # Use ALL critics
                router_uniform=True,  # Equal weights (no softmax)
            )
            logger.info(f"  A4: initial={results['A4_no_routing'].initial_accuracy:.3f} final={results['A4_no_routing'].final_accuracy:.3f}")

            # A5: 3 Actors + 4 Critics + Router (full system)
            logger.info("[A5] 3 Actors + 4 Critics + Router (full system)...")
            a_configs, c_configs, lora = _build_agent_configs(
                registry,
                actor_names=all_actor_names[:3],
                critic_names=all_critic_names,
            )
            results["A5_full_system"] = _run_deliberation_on_samples(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=2,        # Top-2 confident critics
                router_uniform=False,  # Softmax confidence weighting
            )
            logger.info(f"  A5: initial={results['A5_full_system'].initial_accuracy:.3f} final={results['A5_full_system'].final_accuracy:.3f}")
        else:
            # Single main evaluation with full system
            logger.info("[Main] Full system evaluation...")
            a_configs, c_configs, lora = _build_agent_configs(registry)
            results["main"] = _run_deliberation_on_samples(
                engine, a_configs, c_configs, samples, dataset_name,
                lora, num_rounds, max_tokens, temperature,
                router_top_k=2,
            )

    finally:
        del engine
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return results


def print_qualitative_examples(result: EvalResult, num_examples: int):
    """Print qualitative examples for analysis."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Qualitative Examples")
    logger.info("=" * 60)

    details = result.sample_details[:num_examples]

    for i, detail in enumerate(details, 1):
        logger.info(f"\n[Example {i}]")
        logger.info(f"  Question: {detail['question'][:100]}...")
        logger.info(f"  Predicted: {detail['final_answer']}")
        logger.info(f"  Ground Truth: {detail['ground_truth']}")
        logger.info(f"  Confidence: {detail.get('confidence', 0.0):.2f}")


def main():
    args = parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Society Evaluation")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Society dir: {args.society_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Run ablations: {args.run_ablations}")
    logger.info("=" * 60)

    # Load society registry
    logger.info("[Step 1] Loading society registry...")
    registry = load_society_registry(args.society_dir)

    # Load dataset
    logger.info("[Step 2] Loading dataset...")
    samples = load_dataset(args.dataset, args.seed, args.max_samples)
    logger.info(f"  Test samples: {len(samples)}")

    # Run all evaluations with shared engine
    logger.info("[Step 3] Running evaluation (shared base model)...")
    total_start = time.time()
    ablation_results = run_all_evaluations(
        registry=registry,
        samples=samples,
        dataset_name=args.dataset,
        num_rounds=args.num_rounds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        run_ablations=args.run_ablations,
    )
    total_time = time.time() - total_start
    logger.info(f"Total evaluation time: {total_time:.1f}s")

    # Save results
    logger.info("[Step 4] Saving results...")

    results_file = os.path.join(output_dir, "results.json")
    first_result = list(ablation_results.values())[0] if ablation_results else None
    with open(results_file, "w") as f:
        json.dump({
            "ablation_results": {
                name: {
                    "initial_accuracy": r.initial_accuracy,
                    "final_accuracy": r.final_accuracy,
                    "per_round_accuracy": r.per_round_accuracy,
                    "improvement_rate": r.improvement_rate,
                    "absolute_improvement": r.absolute_improvement,
                    "consensus_accuracy": r.consensus_accuracy,
                    "diversity_score": r.diversity_score,
                    "ci_95": list(r.ci_95),
                    "num_samples": r.num_samples,
                    "eval_time_seconds": r.eval_time_seconds,
                }
                for name, r in ablation_results.items()
            },
            "total_eval_time_seconds": total_time,
            "sample_details": first_result.sample_details if first_result else [],
        }, f, indent=2)

    logger.info(f"  Results saved: {results_file}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)

    for name, result in ablation_results.items():
        logger.info(f"\n[{name}]")
        logger.info(f"  Initial Accuracy:  {result.initial_accuracy:.4f}")
        logger.info(f"  Final Accuracy:    {result.final_accuracy:.4f}")
        logger.info(f"  Improvement Rate:  {result.improvement_rate:.4f}")
        logger.info(f"  Absolute Gain:     {result.absolute_improvement:+.4f}")
        logger.info(f"  95% CI:            ({result.ci_95[0]:.4f}, {result.ci_95[1]:.4f})")
        logger.info(f"  Diversity Score:   {result.diversity_score:.4f}")
        logger.info(f"  Eval Time:         {result.eval_time_seconds:.1f}s")

    # Print qualitative examples
    main_result = ablation_results.get("A5_full_system", first_result)
    if main_result:
        print_qualitative_examples(main_result, args.num_samples_for_qualitative)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
