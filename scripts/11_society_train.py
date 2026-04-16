"""
Society alternating training for N Actors + M Critics.

Uses society_alternating_train() to execute alternating training:
- Each iteration: train all Critics (fixed Actors), then train all Actors (fixed Critics)

Usage:
    python scripts/11_society_train.py \
        --config configs/society/experiment_h100.yaml \
        --num_iterations 2 \
        --num_rounds 5
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import resolve_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

ALLOWED_DATASETS = ("boolq", "mmlu", "bbh", "sciq", "arc", "math", "gsm8k")
COMMON_KEYS = ("model_name", "dataset", "cache_dir", "output_dir", "seed", "max_samples")

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "cache/society",
    "actor_base_dir": "cache/society/actors",
    "critic_base_dir": "cache/society/critics",
    "output_dir": "cache/society/society",
    "num_iterations": 2,
    "num_rounds": 5,
    "num_simulations": 5,
    "reward_threshold": 0.0,
    "lora_r": 256,
    "lora_alpha": 512,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "num_epochs": 1,
    "max_length": 2048,
    "beta": 0.1,
    "seed": 42,
    "device": 0,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.85,
    "checkpoint_dir": "cache/society/checkpoints",
}


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Society alternating training",
    )
    parser.add_argument(
        "--config", type=str, default="configs/society/experiment_h100.yaml",
        help="YAML config path.",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=None,
        help="Number of alternating iterations (overrides config).",
    )
    parser.add_argument(
        "--num_rounds", type=int, default=None,
        help="Number of deliberation rounds (overrides config).",
    )
    cli_args = parser.parse_args()

    args = resolve_config(
        cli_args.config, "step05_train_society", STEP_DEFAULTS,
        common_keys=COMMON_KEYS,
        allowed_datasets=ALLOWED_DATASETS,
    )

    # CLI overrides
    if cli_args.num_iterations is not None:
        args.num_iterations = cli_args.num_iterations
    if cli_args.num_rounds is not None:
        args.num_rounds = cli_args.num_rounds

    return args


def load_agent_registry(
    actor_dir: str,
    critic_dir: str,
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Load actor and critic registries."""
    actor_registry_file = os.path.join(actor_dir, "actor_registry.json")
    critic_registry_file = os.path.join(critic_dir, "critic_registry.json")

    actor_paths = {}
    critic_paths = {}

    if os.path.exists(actor_registry_file):
        with open(actor_registry_file) as f:
            data = json.load(f)
            actor_paths = {
                style: info["model_path"]
                for style, info in data["actors"].items()
            }
        logger.info(f"Loaded {len(actor_paths)} actors from registry")
    else:
        logger.warning(f"Actor registry not found: {actor_registry_file}")

    if os.path.exists(critic_registry_file):
        with open(critic_registry_file) as f:
            data = json.load(f)
            critic_paths = {
                error_type: info["model_path"]
                for error_type, info in data["critics"].items()
            }
        logger.info(f"Loaded {len(critic_paths)} critics from registry")
    else:
        logger.warning(f"Critic registry not found: {critic_registry_file}")

    return actor_paths, critic_paths


def create_agent_registry(
    actor_paths: Dict[str, str],
    critic_paths: Dict[str, str],
    base_model: str,
) -> "AgentRegistry":
    """Create AgentRegistry from loaded paths."""
    from src.society.agent_registry import (
        AgentRegistry,
        AgentConfig,
        AgentType,
        ThinkingStyle,
        ErrorType,
    )

    registry = AgentRegistry()

    # Register actors
    for style, path in actor_paths.items():
        try:
            thinking_style = ThinkingStyle(style)
        except ValueError:
            thinking_style = ThinkingStyle.ANALYTICAL

        config = AgentConfig(
            name=f"actor_{style}",
            agent_type=AgentType.ACTOR,
            thinking_style=thinking_style,
            model_path=path,
            system_prompt="",
            temperature=0.7,
            max_tokens=512,
        )
        registry.register(config)

    # Register critics
    for error_type, path in critic_paths.items():
        try:
            error = ErrorType(error_type)
        except ValueError:
            error = ErrorType.LOGIC

        # Map error type to thinking style for critics
        error_to_style = {
            ErrorType.ARITHMETIC: ThinkingStyle.ANALYTICAL,
            ErrorType.LOGIC: ThinkingStyle.ANALYTICAL,
            ErrorType.HALLUCINATION: ThinkingStyle.SKEPTICAL,
            ErrorType.VERIFICATION: ThinkingStyle.SKEPTICAL,
            ErrorType.INTERPRETATION: ThinkingStyle.ANALYTICAL,
            ErrorType.COMPLETENESS: ThinkingStyle.PRECISE,
        }

        thinking_style = error_to_style.get(error, ThinkingStyle.ANALYTICAL)

        config = AgentConfig(
            name=f"critic_{error_type}",
            agent_type=AgentType.CRITIC,
            thinking_style=thinking_style,
            model_path=path,
            system_prompt="",
            temperature=0.7,
            max_tokens=256,
        )
        registry.register(config)

    logger.info(f"Created registry with {len(registry.list_actors())} actors and {len(registry.list_critics())} critics")

    return registry


def main():
    args = parse_args()

    # Setup directories
    output_dir = args.output_dir
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Society Alternating Training")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Actor dir: {args.actor_base_dir}")
    logger.info(f"  Critic dir: {args.critic_base_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Iterations: {args.num_iterations}")
    logger.info(f"  Rounds: {args.num_rounds}")
    logger.info("=" * 60)

    # Load dataset
    logger.info("[Step 1] Loading dataset...")
    from src.data.loader import load_dataset

    data = load_dataset(args.dataset, seed=args.seed)
    train_data = data.get("train", [])

    if args.max_samples:
        train_data = train_data[:args.max_samples]

    logger.info(f"  Training samples: {len(train_data)}")

    # Load agent registries
    logger.info("[Step 2] Loading agent registries...")
    actor_paths, critic_paths = load_agent_registry(
        args.actor_base_dir,
        args.critic_base_dir,
    )

    if not actor_paths or not critic_paths:
        logger.error("No actors or critics found. Please run diversification scripts first.")
        return

    # Create agent registry
    logger.info("[Step 3] Creating agent registry...")
    registry = create_agent_registry(
        actor_paths,
        critic_paths,
        args.model_name,
    )

    # Run society alternating training
    logger.info("[Step 4] Running society alternating training...")
    from src.society.society_trainer import society_alternating_train

    result = society_alternating_train(
        registry=registry,
        dataset=train_data,
        dataset_name=args.dataset,
        output_base_dir=output_dir,
        num_iterations=args.num_iterations,
        num_rounds=args.num_rounds,
        num_simulations=args.num_simulations,
        reward_threshold=args.reward_threshold,
        lora_r=args.lora_r,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        beta=args.beta,
        seed=args.seed,
        checkpoint_dir=checkpoint_dir,
    )

    # Save final registry
    logger.info("[Step 5] Saving final registry...")

    final_registry_file = os.path.join(output_dir, "final_agent_registry.json")
    with open(final_registry_file, "w") as f:
        json.dump({
            "actors": {
                name: {"model_path": path}
                for name, path in result.actor_paths.items()
            },
            "critics": {
                name: {"model_path": path}
                for name, path in result.critic_paths.items()
            },
            "metrics": result.metrics,
            "training_config": {
                "num_iterations": args.num_iterations,
                "num_rounds": args.num_rounds,
                "lora_r": args.lora_r,
                "learning_rate": args.learning_rate,
                "beta": args.beta,
            },
        }, f, indent=2)

    logger.info(f"  Final registry: {final_registry_file}")

    logger.info("=" * 60)
    logger.info("Society training complete!")
    logger.info(f"  Trained actors: {len(result.actor_paths)}")
    logger.info(f"  Trained critics: {len(result.critic_paths)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
