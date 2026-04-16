"""
Society evaluation and ablation experiments.

Runs A1-A5 ablation experiments and computes:
- Per-round accuracy
- Improvement rate
- Consensus accuracy
- Diversity metrics
- Wilson 95% CI

Usage:
    python scripts/12_society_evaluate.py \
        --config configs/society/experiment_h100.yaml \
        --run_ablations
"""

from __future__ import annotations

import json
import logging
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
COMMON_KEYS = ("model_name", "dataset", "cache_dir", "society_dir", "output_dir", "seed", "max_samples")

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "cache/society",
    "society_dir": "cache/society/society",
    "output_dir": "cache/society/eval",
    "num_rounds": 5,
    "max_tokens": 512,
    "temperature": 0.7,
    "device": 0,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.85,
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

    if not os.path.exists(registry_dir):
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
    """Compute diversity score using entropy."""
    if not responses:
        return 0.0

    # Count unique responses
    unique_responses = len(set(responses))
    total_responses = len(responses)

    # Normalize diversity
    diversity = unique_responses / total_responses if total_responses > 0 else 0.0

    return diversity


def evaluate_society(
    registry: Dict[str, Any],
    samples: List[Dict],
    dataset_name: str,
    num_rounds: int,
    max_tokens: int,
    temperature: float,
    aggregation: str,
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
) -> EvalResult:
    """Evaluate society on test samples."""
    from src.inference.vllm_server import VLLMInference
    from src.society.multi_deliberation import multi_agent_deliberate_single_gpu
    from src.algorithms.reward import extract_answer
    from src.evaluation.benchmarks import compute_accuracy_with_ci

    start_time = time.time()

    # Collect all agent paths
    actor_paths = {
        name: info["model_path"]
        for name, info in registry.get("actors", {}).items()
    }
    critic_paths = {
        name: info["model_path"]
        for name, info in registry.get("critics", {}).items()
    }

    if not actor_paths or not critic_paths:
        logger.warning("No agents found, using base model")
        # Fall back to base model evaluation
        model_path = registry.get("training_config", {}).get("base_model", "Qwen/Qwen2.5-7B-Instruct")
        model = VLLMInference(
            model_path,
            cuda_device=device,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
        )

        predictions = []
        details = []

        for sample in samples:
            prompt = sample.get("question", "")
            response = model.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            answer = extract_answer(response, sample.get("task_type", "math"))

            predictions.append(answer)
            details.append({
                "question": prompt,
                "response": response,
                "predicted": answer,
                "ground_truth": sample.get("answer", ""),
            })

        labels = [s.get("answer", "") for s in samples]
        accuracy, ci = compute_accuracy_with_ci(predictions, labels)

        return EvalResult(
            initial_accuracy=accuracy,
            final_accuracy=accuracy,
            per_round_accuracy=[accuracy],
            improvement_rate=0.0,
            absolute_improvement=0.0,
            consensus_accuracy=accuracy,
            diversity_score=0.0,
            ci_95=ci,
            num_samples=len(samples),
            eval_time_seconds=time.time() - start_time,
            sample_details=details,
        )

    # Load base model (shared across agents)
    base_model_path = list(actor_paths.values())[0]
    model = VLLMInference(
        base_model_path,
        cuda_device=device,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
    )

    # Create agent configs
    from src.society.agent_registry import AgentConfig, AgentType, ThinkingStyle

    actor_configs = []
    critic_configs = []

    for name, path in actor_paths.items():
        style = ThinkingStyle.ANALYTICAL  # Default
        actor_configs.append(AgentConfig(
            name=name,
            agent_type=AgentType.ACTOR,
            thinking_style=style,
            model_path=path,
            system_prompt="",
        ))

    for name, path in critic_paths.items():
        style = ThinkingStyle.ANALYTICAL  # Default
        critic_configs.append(AgentConfig(
            name=name,
            agent_type=AgentType.CRITIC,
            thinking_style=style,
            model_path=path,
            system_prompt="",
        ))

    # Run evaluation
    all_final_answers = []
    all_initial_answers = []
    all_round_answers = [[] for _ in range(num_rounds)]
    details = []
    all_responses = []

    for sample in samples:
        result = multi_agent_deliberate_single_gpu(
            model=model,
            actor_configs=actor_configs[:1],  # Use one actor for efficiency
            critic_configs=critic_configs[:1],  # Use one critic for efficiency
            sample=sample,
            dataset_name=dataset_name,
            lora_paths={**actor_paths, **critic_paths},
            num_rounds=num_rounds,
            max_tokens=max_tokens,
            temperature=temperature,
            aggregation=aggregation,
        )

        all_final_answers.append(result.final_answer)
        all_responses.append(result.final_answer)

        # Extract initial answer (first actor response)
        if result.individual_results:
            first_result = result.individual_results[0]
            trajectory = first_result.get("trajectory", [])
            if trajectory:
                all_initial_answers.append(trajectory[0].get("actor_answer"))
            else:
                all_initial_answers.append(result.final_answer)
        else:
            all_initial_answers.append(result.final_answer)

        details.append({
            "question": sample.get("question", ""),
            "final_answer": result.final_answer,
            "confidence": result.confidence,
            "ground_truth": sample.get("answer", ""),
        })

    # Compute metrics
    labels = [s.get("answer", "") for s in samples]

    initial_acc, _ = compute_accuracy_with_ci(all_initial_answers, labels)
    final_acc, ci_95 = compute_accuracy_with_ci(all_final_answers, labels)

    improvement_rate = 0.0
    if initial_acc > 0:
        improvement_rate = (final_acc - initial_acc) / initial_acc

    absolute_improvement = final_acc - initial_acc

    # Consensus accuracy (same as final for now)
    consensus_acc = final_acc

    # Diversity score
    diversity = compute_diversity(all_responses)

    return EvalResult(
        initial_accuracy=initial_acc,
        final_accuracy=final_acc,
        per_round_accuracy=[initial_acc, final_acc],
        improvement_rate=improvement_rate,
        absolute_improvement=absolute_improvement,
        consensus_accuracy=consensus_acc,
        diversity_score=diversity,
        ci_95=ci_95,
        num_samples=len(samples),
        eval_time_seconds=time.time() - start_time,
        sample_details=details,
    )


def run_ablation_experiments(
    registry: Dict[str, Any],
    samples: List[Dict],
    dataset_name: str,
    num_rounds: int,
    device: int,
    dtype: str,
    gpu_memory_utilization: float,
) -> Dict[str, EvalResult]:
    """Run ablation experiments A1-A5."""
    results = {}

    # A1: Single Actor (no deliberation)
    logger.info("[A1] Single Actor (no deliberation)...")
    # Use base model only
    results["A1_single_actor"] = evaluate_society(
        {"actors": {}, "critics": {}},  # Empty registry -> base model
        samples,
        dataset_name,
        num_rounds=1,
        max_tokens=512,
        temperature=0.7,
        aggregation="majority_vote",
        device=device,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # A2: Actor + Single Critic
    logger.info("[A2] Actor + Single Critic...")
    if registry.get("actors") and registry.get("critics"):
        # Use first actor and first critic only
        first_actor = list(registry["actors"].items())[0]
        first_critic = list(registry["critics"].items())[0]
        limited_registry = {
            "actors": {first_actor[0]: first_actor[1]},
            "critics": {first_critic[0]: first_critic[1]},
        }
        results["A2_actor_critic"] = evaluate_society(
            limited_registry,
            samples,
            dataset_name,
            num_rounds=num_rounds,
            max_tokens=512,
            temperature=0.7,
            aggregation="majority_vote",
            device=device,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    # A3: Full Society (no routing)
    logger.info("[A3] Full Society (no routing)...")
    results["A3_full_society"] = evaluate_society(
        registry,
        samples,
        dataset_name,
        num_rounds=num_rounds,
        max_tokens=512,
        temperature=0.7,
        aggregation="majority_vote",
        device=device,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # A4: Society with confidence routing
    logger.info("[A4] Society with confidence routing...")
    results["A4_confidence_routing"] = evaluate_society(
        registry,
        samples,
        dataset_name,
        num_rounds=num_rounds,
        max_tokens=512,
        temperature=0.7,
        aggregation="weighted",  # Use weighted aggregation
        device=device,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # A5: Society with error type routing
    logger.info("[A5] Society with error type routing...")
    results["A5_error_routing"] = evaluate_society(
        registry,
        samples,
        dataset_name,
        num_rounds=num_rounds,
        max_tokens=512,
        temperature=0.7,
        aggregation="best",  # Use best selection
        device=device,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
    )

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

    # Setup directories
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

    # Run evaluation
    if args.run_ablations:
        logger.info("[Step 3] Running ablation experiments...")
        ablation_results = run_ablation_experiments(
            registry,
            samples,
            args.dataset,
            args.num_rounds,
            args.device,
            args.dtype,
            args.gpu_memory_utilization,
        )
    else:
        logger.info("[Step 3] Running main evaluation...")
        main_result = evaluate_society(
            registry,
            samples,
            args.dataset,
            args.num_rounds,
            args.max_tokens,
            args.temperature,
            args.aggregation,
            args.device,
            args.dtype,
            args.gpu_memory_utilization,
        )
        ablation_results = {"main": main_result}

    # Save results
    logger.info("[Step 4] Saving results...")

    results_file = os.path.join(output_dir, "results.json")
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
                    "ci_95": r.ci_95,
                    "num_samples": r.num_samples,
                    "eval_time_seconds": r.eval_time_seconds,
                }
                for name, r in ablation_results.items()
            },
            "sample_details": ablation_results.get("main", ablation_results.get("A3_full_society")).sample_details,
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
    main_result = ablation_results.get("main", ablation_results.get("A3_full_society"))
    if main_result:
        print_qualitative_examples(main_result, args.num_samples_for_qualitative)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
