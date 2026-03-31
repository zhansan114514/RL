"""
Step 5: Evaluate Actor-Critic team on benchmark datasets.

Loads trained actor and critic models, runs deliberation on test data,
and reports accuracy with confidence intervals.

Usage:
    python scripts/05_evaluate.py \
    --config configs/config.yaml
"""

from __future__ import annotations

import json
import logging
import os

from _utils import resolve_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

ALLOWED_DATASETS = ("boolq", "mmlu", "bbh", "sciq", "arc")
COMMON_KEYS = ("dataset", "max_samples", "seed")

STEP_DEFAULTS = {
    "actor_path": "experiments/gemma2_boolq/actor",
    "critic_path": "experiments/gemma2_boolq/critic",
    "dataset": "boolq",
    "output_dir": "experiments/gemma2_boolq/eval",
    "num_rounds": 5,
    "max_tokens": 256,
    "temperature": 0.7,
    "max_samples": None,
    "seed": 42,
    "actor_device": 0,
    "critic_device": 0,
    "dtype": "float32",
    "gpu_memory_utilization": 0.45,
}


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Evaluate Actor-Critic team",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    return resolve_config(
        cli_args.config, "step05", STEP_DEFAULTS,
        common_keys=COMMON_KEYS,
        allowed_datasets=ALLOWED_DATASETS,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if trained models exist
    actor_path = args.actor_path
    critic_path = args.critic_path
    if not os.path.exists(actor_path):
        logger.error(f"Actor model not found: {actor_path}")
        logger.error("Please run scripts/04_train_actor.py first")
        raise FileNotFoundError(f"Actor model not found: {actor_path}")
    if not os.path.exists(critic_path):
        logger.error(f"Critic model not found: {critic_path}")
        logger.error("Please run scripts/03_train_critic.py first")
        raise FileNotFoundError(f"Critic model not found: {critic_path}")

    logger.info(f"Loading dataset: {args.dataset}")
    from src.data.loader import load_dataset

    data = load_dataset(args.dataset, seed=args.seed)
    test_data = data.get("test", [])
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    logger.info(f"  Test samples: {len(test_data)}")

    logger.info(f"Loading actor: {args.actor_path}")
    logger.info(f"Loading critic: {args.critic_path}")
    logger.info(f"  Actor on GPU {args.actor_device}, Critic on GPU {args.critic_device}")
    from src.inference.vllm_server import VLLMInference

    actor = VLLMInference(
        args.actor_path,
        cuda_device=args.actor_device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    critic = VLLMInference(
        args.critic_path,
        cuda_device=args.critic_device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    logger.info("Evaluating...")
    from src.evaluation.benchmarks import evaluate_benchmark

    results = evaluate_benchmark(
        actor, critic, test_data, args.dataset,
        num_rounds=args.num_rounds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    logger.info("=" * 50)
    logger.info(f"Dataset: {results['dataset']}")
    logger.info(f"Samples: {results['num_samples']}")
    logger.info(f"Initial accuracy: {results['initial_accuracy']:.3f}")
    logger.info(f"Final accuracy:   {results['final_accuracy']:.3f} +/- {results['ci_margin']:.3f}")
    logger.info(f"Improvement rate:  {results['improvement_rate']:.3f}")
    logger.info(f"Per-round: {[f'{a:.3f}' for a in results['per_round_accuracy']]}")
    logger.info("=" * 50)

    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
