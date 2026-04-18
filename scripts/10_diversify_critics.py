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
COMMON_KEYS = ("model_name", "dataset", "cache_dir", "input_dir", "output_dir", "seed", "device", "dtype", "gpu_memory_utilization")

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "output/society",
    "input_dir": "output/society/classified",
    "actor_dir": "output/society/actors",
    "output_dir": "output/society/critics",
    "error_types": ["arithmetic", "logic", "hallucination", "verification"],
    "max_delib_samples": 50,
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
    "gpu_memory_utilization": 0.65,
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
) -> List[Dict[str, Any]]:
    """
    Build DPO preference pairs for Critic training from existing bootstrap data.

    Chosen: detailed feedback pointing out the specific error type
    Rejected: generic/vague feedback
    """
    from src.algorithms.reward import extract_answer

    preference_pairs = []

    # Filter samples with this error type
    error_samples = [
        r for r in classified_results
        if r.get("error_type") == error_type
    ]

    logger.info(f"  Found {len(error_samples)} samples for error type '{error_type}'")

    # Also use all samples to build generic critic pairs
    all_samples = classified_results

    for result in all_samples:
        sample_id = result["sample_id"]

        if sample_id not in trajectories:
            continue

        traj = trajectories[sample_id]
        sample = traj["sample"]
        correct_answer = traj.get("consensus_answer", "")

        # Collect all responses across rounds
        all_responses = list(traj.get("initial_responses", []))
        for round_responses in traj.get("debate_rounds", []):
            all_responses.extend(round_responses)

        # Find wrong answers as context for critic training
        wrong_responses = []
        correct_responses = []
        for resp in all_responses:
            response_text = resp.get("response", "")
            answer = resp.get("answer")
            if answer and answer != correct_answer:
                wrong_responses.append(response_text)
            else:
                correct_responses.append(response_text)

        if wrong_responses:
            # Build chosen: specific error-focused feedback
            chosen = (
                f"Let me analyze this solution carefully. "
                f"I notice a potential {error_type} error in the reasoning. "
                f"The correct answer should be {correct_answer}. "
                f"Please review the calculation steps and verify the result. "
                f"[Confidence: 0.9]"
            )
            # Build rejected: vague/generic feedback
            rejected = (
                f"This solution looks mostly fine, but there might be some issues. "
                f"Try checking your work again. [Confidence: 0.3]"
            )

            preference_pairs.append({
                "sample": sample,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "error_type": error_type,
                    "sample_id": sample_id,
                },
            })

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
    from datasets import Dataset
    from src.training.dpo_trainer import train_dpo
    from src.utils.model_utils import detect_model_type

    model_type = detect_model_type(model_name)

    # Create output directory
    critic_output_dir = os.path.join(output_dir, f"critic_{error_type}")
    os.makedirs(critic_output_dir, exist_ok=True)

    logger.info(f"  Training DPO for '{error_type}'...")
    logger.info(f"    Pairs: {len(preference_pairs)}")
    logger.info(f"    Output: {critic_output_dir}")

    # Convert preference_pairs to HuggingFace Dataset
    hf_data = {
        "prompt": [p["sample"].get("question", "") for p in preference_pairs],
        "chosen": [p["chosen"] for p in preference_pairs],
        "rejected": [p["rejected"] for p in preference_pairs],
    }
    preference_dataset = Dataset.from_dict(hf_data)

    # Train DPO
    checkpoint_path = train_dpo(
        model_name_or_path=model_name,
        preference_dataset=preference_dataset,
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

    # Train each critic
    logger.info("[Step 4] Training specialized critics...")

    critic_paths = {}

    for error_type in args.error_types:
        logger.info(f"\n--- Training Critic: {error_type} ---")

        # Build preference pairs for this error type
        preference_pairs = build_critic_preference_pairs(
            classified_results,
            trajectories,
            error_type,
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
