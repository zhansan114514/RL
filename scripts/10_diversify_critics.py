"""
Diversify Critics by training specialized LoRA adapters for each error type.

For each Critic:
1. Load first diversified Actor + base model
2. Run deliberation to generate trajectories
3. Filter by error type
4. Build DPO preference pairs (chosen=Critic correct guidance, rejected=Critic invalid guidance)
5. Train with DPO
6. Save to cache/society/critics/{agent_id}/

Usage:
    python scripts/10_diversify_critics.py \
        --config configs/society/experiment_h100.yaml \
        --error_types arithmetic logic hallucination verification
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import resolve_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

ALLOWED_DATASETS = ("boolq", "mmlu", "bbh", "sciq", "arc", "math", "gsm8k")
COMMON_KEYS = ("model_name", "dataset", "cache_dir", "input_dir", "output_dir", "seed")

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "cache/society",
    "input_dir": "cache/society/classified",
    "actor_dir": "cache/society/actors",
    "output_dir": "cache/society/critics",
    "error_types": ["arithmetic", "logic", "hallucination", "verification"],
    "lora_r": 256,
    "lora_alpha": 512,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "num_epochs": 1,
    "max_length": 2048,
    "beta": 0.1,
    "num_rounds": 5,
    "seed": 42,
    "device": 0,
    "dtype": "bfloat16",
}


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Diversify Critics",
    )
    parser.add_argument(
        "--config", type=str, default="configs/society/experiment_h100.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    return resolve_config(
        cli_args.config, "step04_diversify_critics", STEP_DEFAULTS,
        common_keys=COMMON_KEYS,
        allowed_datasets=ALLOWED_DATASETS,
    )


def load_classified_data(input_dir: str) -> Dict[str, Any]:
    """Load classified data."""
    classified_file = os.path.join(input_dir, "classified_data.json")

    if not os.path.exists(classified_file):
        raise FileNotFoundError(f"Classified data not found: {classified_file}")

    with open(classified_file) as f:
        data = json.load(f)

    return data


def load_actor_registry(actor_dir: str) -> Dict[str, str]:
    """Load actor registry."""
    registry_file = os.path.join(actor_dir, "actor_registry.json")

    if not os.path.exists(registry_file):
        raise FileNotFoundError(f"Actor registry not found: {registry_file}")

    with open(registry_file) as f:
        data = json.load(f)

    # Extract model paths
    actor_paths = {
        style: info["model_path"]
        for style, info in data["actors"].items()
    }

    logger.info(f"Loaded {len(actor_paths)} actors from registry")

    return actor_paths


def build_critic_preference_pairs(
    classified_results: List[Dict],
    trajectories: Dict[str, Any],
    error_type: str,
    actor_model,
    base_model,
    dataset_name: str,
    num_rounds: int,
) -> List[Dict[str, Any]]:
    """
    Build DPO preference pairs for Critic training.

    Chosen: Critic provides correct guidance leading to correct answer
    Rejected: Critic provides invalid guidance leading to wrong answer
    """
    from src.algorithms.deliberation import deliberate
    from src.algorithms.reward import extract_answer

    preference_pairs = []

    # Filter samples with this error type
    error_samples = [
        r for r in classified_results
        if r.get("error_type") == error_type
    ]

    logger.info(f"  Found {len(error_samples)} samples for error type '{error_type}'")

    for result in error_samples[:50]:  # Limit for efficiency
        sample_id = result["sample_id"]

        if sample_id not in trajectories:
            continue

        traj = trajectories[sample_id]
        sample = traj["sample"]

        # Run deliberation to generate trajectories
        try:
            trajectory = deliberate(
                actor_model,
                base_model,  # Use base model as critic
                sample,
                dataset_name,
                num_rounds=num_rounds,
                max_tokens=512,
                temperature=0.7,
            )
        except Exception as e:
            logger.warning(f"  Deliberation failed for {sample_id}: {e}")
            continue

        # Extract chosen (rounds where answer improved) and rejected
        correct_answer = traj.get("consensus_answer", "")

        for i, round_data in enumerate(trajectory):
            actor_answer = round_data.get("actor_answer", "")
            critic_feedback = round_data.get("critic_feedback", "")

            # Check if this round led to improvement
            extracted_answer = extract_answer(actor_answer, sample.get("task_type", "math"))

            if extracted_answer == correct_answer:
                # This is a good guidance example
                chosen = critic_feedback
                # Find a rejected example (previous round or generic)
                rejected = "This solution looks correct."

                preference_pairs.append({
                    "sample": sample,
                    "chosen": chosen,
                    "rejected": rejected,
                    "metadata": {
                        "error_type": error_type,
                        "sample_id": sample_id,
                        "round": i,
                    },
                })

        if len(preference_pairs) >= 100:  # Limit pairs per error type
            break

    logger.info(f"  Built {len(preference_pairs)} preference pairs")
    return preference_pairs


def train_critic_dpo(
    model_name: str,
    preference_pairs: List[Dict],
    error_type: str,
    output_dir: str,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    max_length: int,
    beta: float,
    seed: int,
    device: int,
) -> str:
    """Train Critic with DPO."""
    from src.training.dpo_trainer import train_dpo
    from src.utils.model_utils import detect_model_type

    model_type = detect_model_type(model_name)

    # Create output directory
    critic_output_dir = os.path.join(output_dir, f"critic_{error_type}")
    os.makedirs(critic_output_dir, exist_ok=True)

    logger.info(f"  Training DPO for '{error_type}'...")
    logger.info(f"    Pairs: {len(preference_pairs)}")
    logger.info(f"    Output: {critic_output_dir}")

    # Train DPO
    checkpoint_path = train_dpo(
        model_name=model_name,
        preference_pairs=preference_pairs,
        dataset_name="society",
        output_dir=critic_output_dir,
        model_type=model_type,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_length=max_length,
        beta=beta,
        seed=seed,
        device=device,
    )

    logger.info(f"  Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def main():
    args = parse_args()

    # Setup directories
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Diversify Critics")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Input dir: {input_dir}")
    logger.info(f"  Actor dir: {args.actor_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Error types: {args.error_types}")
    logger.info("=" * 60)

    # Load classified data
    logger.info("[Step 1] Loading classified data...")
    classified_data = load_classified_data(input_dir)
    classified_results = classified_data["results"]

    # Load trajectories
    logger.info("[Step 2] Loading bootstrap trajectories...")
    from scripts._utils import load_yaml_config

    # Load trajectories directly
    bootstrap_dir = os.path.join(args.cache_dir, "bootstrap")
    trajectory_file = os.path.join(bootstrap_dir, "trajectories.jsonl")

    trajectories = {}
    if os.path.exists(trajectory_file):
        with open(trajectory_file) as f:
            for line in f:
                if line.strip():
                    traj = json.loads(line)
                    trajectories[traj["sample_id"]] = traj
        logger.info(f"  Loaded {len(trajectories)} trajectories")

    # Load first actor
    logger.info("[Step 3] Loading first diversified actor...")
    actor_paths = load_actor_registry(args.actor_dir)

    # Get first available actor
    first_actor_style = next(iter(actor_paths))
    first_actor_path = actor_paths[first_actor_style]

    logger.info(f"  Using actor: {first_actor_style} -> {first_actor_path}")

    # Load models
    logger.info("[Step 4] Loading models...")
    from src.inference.vllm_server import VLLMInference

    actor_model = VLLMInference(
        first_actor_path,
        cuda_device=args.device,
        dtype=args.dtype,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
    )

    base_model = VLLMInference(
        args.model_name,
        cuda_device=args.device,
        dtype=args.dtype,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
    )

    # Train each critic
    logger.info("[Step 5] Training specialized critics...")

    critic_paths = {}

    for error_type in args.error_types:
        logger.info(f"\n--- Training Critic: {error_type} ---")

        # Build preference pairs for this error type
        preference_pairs = build_critic_preference_pairs(
            classified_results,
            trajectories,
            error_type,
            actor_model,
            base_model,
            args.dataset,
            args.num_rounds,
        )

        if not preference_pairs:
            logger.warning(f"  No preference pairs for '{error_type}', skipping")
            continue

        # Train DPO
        checkpoint_path = train_critic_dpo(
            model_name=args.model_name,
            preference_pairs=preference_pairs,
            error_type=error_type,
            output_dir=output_dir,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            beta=args.beta,
            seed=args.seed,
            device=args.device,
        )

        critic_paths[error_type] = checkpoint_path

    # Save critic registry
    logger.info("\n[Step 6] Saving critic registry...")

    registry_file = os.path.join(output_dir, "critic_registry.json")
    with open(registry_file, "w") as f:
        json.dump({
            "critics": {
                error_type: {
                    "error_type": error_type,
                    "model_path": path,
                    "base_model": args.model_name,
                }
                for error_type, path in critic_paths.items()
            },
            "metadata": {
                "base_model": args.model_name,
                "num_critics": len(critic_paths),
            },
        }, f, indent=2)

    logger.info(f"  Registry saved: {registry_file}")

    logger.info("=" * 60)
    logger.info("Critic diversification complete!")
    logger.info(f"  Trained {len(critic_paths)} critics:")
    for error_type, path in critic_paths.items():
        logger.info(f"    {error_type}: {path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
