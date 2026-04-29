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
from typing import TYPE_CHECKING, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.utils.config import ConfigManager

if TYPE_CHECKING:
    from src.society.agent_registry import AgentRegistry

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "output/society",
    "actor_base_dir": "output/society/actors",
    "critic_base_dir": "output/society/critics",
    "output_dir": "output/society/society",
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
    "min_pairs_per_critic": 64,
    "min_specialty_items": 32,
    "min_specialty_ratio": 0.08,
    "specialty_ratio": 0.7,
    "general_ratio": 0.2,
    "calibration_ratio": 0.1,
    "seed": 42,
    "device": 0,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 8192,
    "checkpoint_dir": "output/society/checkpoints",
    "actor_temperature": 0.7,
    "actor_max_tokens": 512,
    "critic_max_tokens": 256,
    "api_key": "",
    "api_base": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "api_model": "glm-4-flash",
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

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    args = cfg.step("step05_train_society", defaults=STEP_DEFAULTS).to_namespace()

    # Preserve config path for logging
    args.config = cli_args.config

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
            trained_count = 0
            frozen_count = 0
            for skill_name, info in data["critics"].items():
                path = info.get("model_path", "")
                status = info.get("status", "active" if path else "frozen_base")
                critic_paths[skill_name] = path
                if path:
                    trained_count += 1
                else:
                    frozen_count += 1
                    logger.info(
                        f"critic_{skill_name}: {status}, participates with "
                        "base model only"
                    )
        logger.info(
            f"Loaded {len(critic_paths)} critics from registry "
            f"({trained_count} trained LoRA, {frozen_count} frozen_base)"
        )
    else:
        logger.warning(f"Critic registry not found: {critic_registry_file}")

    return actor_paths, critic_paths


def create_agent_registry(
    actor_paths: Dict[str, str],
    critic_paths: Dict[str, str],
    base_model: str,
    actor_temperature: float = 0.7,
    actor_max_tokens: int = 512,
    critic_max_tokens: int = 256,
) -> "AgentRegistry":
    """Create AgentRegistry from loaded paths.

    model_path = base model (shared across all agents).
    lora_path  = per-agent LoRA adapter checkpoint.
    """
    from src.society.agent_registry import (
        AgentRegistry,
        AgentConfig,
        AgentRole,
        resolve_reasoning_style,
        resolve_critic_skill,
    )

    registry = AgentRegistry(base_model_path=base_model)

    # Register actors: model_path=base model, lora_path=LoRA checkpoint
    for style, path in actor_paths.items():
        try:
            reasoning_style = resolve_reasoning_style(style)
        except ValueError as e:
            logger.error(f"Cannot resolve actor style '{style}': {e}")
            raise

        config = AgentConfig(
            name=f"actor_{reasoning_style.value}",
            role=AgentRole.ACTOR,
            reasoning_style=reasoning_style,
            model_path=base_model,
            lora_path=path,
            system_prompt="",
            temperature=actor_temperature,
            max_tokens=actor_max_tokens,
        )
        registry.register(config)

    # Register critics: model_path=base model, lora_path=LoRA checkpoint
    for skill_name, path in critic_paths.items():
        try:
            skill = resolve_critic_skill(skill_name)
        except ValueError as e:
            logger.error(f"Cannot resolve critic skill '{skill_name}': {e}")
            raise

        config = AgentConfig(
            name=f"critic_{skill.value}",
            role=AgentRole.CRITIC,
            error_specialty=skill,
            model_path=base_model,
            lora_path=path,
            system_prompt="",
            temperature=actor_temperature,
            max_tokens=critic_max_tokens,
        )
        registry.register(config)
        if not path:
            logger.info(
                f"critic_{skill.value}: frozen_base, participates with base "
                "model only (no LoRA)"
            )

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
    # For datasets without a train split (e.g. MMLU), flatten all splits
    if not train_data:
        for split_data in data.values():
            train_data.extend(split_data)

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
        sys.exit(1)

    # Create agent registry
    logger.info("[Step 3] Creating agent registry...")
    registry = create_agent_registry(
        actor_paths,
        critic_paths,
        args.model_name,
        actor_temperature=args.actor_temperature,
        actor_max_tokens=args.actor_max_tokens,
        critic_max_tokens=args.critic_max_tokens,
    )

    # Run society alternating training
    logger.info("[Step 4] Running society alternating training...")
    from src.society.society_trainer import society_alternating_train

    # Resolve API key for live error-profile classification
    api_key = getattr(args, "api_key", "")
    if not api_key:
        api_key = os.environ.get("GLM_API_KEY", "")
    if api_key:
        os.environ["GLM_API_KEY"] = api_key
    else:
        logger.warning(
            "GLM_API_KEY not set. Unseen raw pairs will be routed to general pool."
        )
    api_base = getattr(args, "api_base", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
    api_model = getattr(args, "api_model", "glm-4-flash")

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
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        beta=args.beta,
        seed=args.seed,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_samples=len(train_data),
        min_pairs_per_critic=args.min_pairs_per_critic,
        min_specialty_items=args.min_specialty_items,
        min_specialty_ratio=args.min_specialty_ratio,
        specialty_ratio=args.specialty_ratio,
        general_ratio=args.general_ratio,
        calibration_ratio=args.calibration_ratio,
        classifications_cache_dir=args.cache_dir + "/classified",
        api_key=api_key,
        api_base=api_base,
        api_model=api_model,
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
                "base_model": args.model_name,
                "num_iterations": args.num_iterations,
                "num_rounds": args.num_rounds,
                "lora_r": args.lora_r,
                "learning_rate": args.learning_rate,
                "beta": args.beta,
                "active_selection": {
                    "min_pairs_per_critic": args.min_pairs_per_critic,
                    "min_specialty_items": args.min_specialty_items,
                    "min_specialty_ratio": args.min_specialty_ratio,
                },
                "critic_training_mix": {
                    "specialty_ratio": args.specialty_ratio,
                    "general_ratio": args.general_ratio,
                    "calibration_ratio": args.calibration_ratio,
                },
            },
        }, f, indent=2)

    logger.info(f"  Final registry: {final_registry_file}")

    logger.info("=" * 60)
    logger.info("Society training complete!")
    logger.info(f"  Trained actors: {len(result.actor_paths)}")
    logger.info(f"  Trained critics: {len(result.critic_paths)}")
    logger.info("=" * 60)

    # Fail if no agents were actually trained (prevents empty Phase 5 from being marked done)
    total_pairs = sum(
        v.get("pairs", 0) for v in result.metrics.values()
        if isinstance(v, dict)
    )
    if total_pairs == 0:
        logger.error("No preference pairs generated for any agent. Exiting with error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
