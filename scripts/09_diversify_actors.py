"""
Diversify Actors by training specialized LoRA adapters for each thinking style.

For each Actor:
1. Load base model
2. Filter data by reasoning style
3. Build DPO preference pairs
4. Train with DPO (lora_r=256, alpha=512, lr=5e-5, beta=0.1)
5. Save to cache/society/actors/{agent_id}/

Usage:
    python scripts/09_diversify_actors.py \
        --config configs/society/experiment_h100.yaml \
        --thinking_styles algebraic direct backtracking
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
    "output_dir": "output/society/actors",
    "thinking_styles": ["algebraic", "direct", "backtracking"],
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
}


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Diversify Actors",
    )
    parser.add_argument(
        "--config", type=str, default="configs/society/experiment_h100.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    return resolve_config(
        cli_args.config, "step03_diversify_actors", STEP_DEFAULTS,
        common_keys=COMMON_KEYS,
        allowed_datasets=ALLOWED_DATASETS,
    )


def load_classified_data(input_dir: str) -> Dict[str, Any]:
    """Load classified data from classification step."""
    classified_file = os.path.join(input_dir, "classified_data.json")

    if not os.path.exists(classified_file):
        raise FileNotFoundError(f"Classified data not found: {classified_file}")

    with open(classified_file) as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data['results'])} classified samples")

    return data


def load_bootstrap_trajectories(cache_dir: str) -> Dict[str, Any]:
    """Load bootstrap trajectories."""
    bootstrap_dir = os.path.join(cache_dir, "bootstrap")
    trajectory_file = os.path.join(bootstrap_dir, "trajectories.jsonl")

    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"Trajectories not found: {trajectory_file}")

    trajectories = {}
    with open(trajectory_file) as f:
        for line in f:
            if line.strip():
                traj = json.loads(line)
                sample_id = traj["sample_id"]
                trajectories[sample_id] = traj

    logger.info(f"Loaded {len(trajectories)} trajectories")

    return trajectories


def build_preference_pairs_for_style(
    classified_results: List[Dict],
    trajectories: Dict[str, Any],
    thinking_style: str,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    """
    Build DPO preference pairs for a specific thinking style.

    Chosen: correct response with matching style
    Rejected: incorrect response OR correct response with different style

    Fallback for small datasets: if style-specific samples are too few,
    use ALL samples and pick different agent responses as chosen/rejected.
    """
    from src.algorithms.reward import extract_answer

    preference_pairs = []

    # Filter samples with this reasoning style
    style_samples = [
        r for r in classified_results
        if r.get("reasoning_style") == thinking_style
    ]

    # Fallback: if too few style-specific samples, use ALL samples
    use_fallback = len(style_samples) < 2
    if use_fallback:
        logger.info(f"  Only {len(style_samples)} samples for style '{thinking_style}', using ALL samples")
        style_samples = classified_results
    else:
        logger.info(f"  Found {len(style_samples)} samples for style '{thinking_style}'")

    for result in style_samples:
        sample_id = result["sample_id"]

        if sample_id not in trajectories:
            continue

        traj = trajectories[sample_id]
        sample = traj["sample"]
        correct_answer = traj.get("consensus_answer", "")

        # Get responses from all rounds (initial + debate) for more diversity
        all_responses = list(traj.get("initial_responses", []))
        for round_responses in traj.get("debate_rounds", []):
            all_responses.extend(round_responses)

        if not all_responses:
            continue

        # Collect correct and incorrect responses
        correct_responses = []
        incorrect_responses = []
        for resp in all_responses:
            response_text = resp.get("response", "")
            answer = resp.get("answer")
            task_type = sample.get("task_type", "math")
            extracted_answer = extract_answer(response_text, task_type)
            if extracted_answer == correct_answer:
                correct_responses.append(response_text)
            else:
                incorrect_responses.append(response_text)

        chosen_response = None
        rejected_response = None

        if correct_responses and incorrect_responses:
            # Ideal: correct vs incorrect
            chosen_response = correct_responses[0]
            rejected_response = incorrect_responses[0]
        elif len(correct_responses) >= 2:
            # All correct: use different agent responses as chosen/rejected
            chosen_response = correct_responses[0]
            rejected_response = correct_responses[-1]
        elif len(incorrect_responses) >= 2:
            chosen_response = incorrect_responses[0]
            rejected_response = incorrect_responses[-1]

        if chosen_response and rejected_response and chosen_response != rejected_response:
            preference_pairs.append({
                "sample": sample,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "metadata": {
                    "thinking_style": thinking_style,
                    "sample_id": sample_id,
                    "fallback": use_fallback,
                },
            })

    logger.info(f"  Built {len(preference_pairs)} preference pairs")
    return preference_pairs


def train_actor_dpo(
    model_name: str,
    preference_pairs: List[Dict],
    thinking_style: str,
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
    """Train Actor with DPO."""
    from datasets import Dataset
    from src.training.dpo_trainer import train_dpo
    from src.utils.model_utils import detect_model_type

    model_type = detect_model_type(model_name)

    # Create output directory
    actor_output_dir = os.path.join(output_dir, f"actor_{thinking_style}")
    os.makedirs(actor_output_dir, exist_ok=True)

    logger.info(f"  Training DPO for '{thinking_style}'...")
    logger.info(f"    Pairs: {len(preference_pairs)}")
    logger.info(f"    Output: {actor_output_dir}")

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
        output_dir=actor_output_dir,
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
    logger.info("Diversify Actors")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Input dir: {input_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Thinking styles: {args.thinking_styles}")
    logger.info("=" * 60)

    # Load classified data
    logger.info("[Step 1] Loading classified data...")
    classified_data = load_classified_data(input_dir)
    classified_results = classified_data["results"]

    # Load trajectories
    logger.info("[Step 2] Loading bootstrap trajectories...")
    trajectories = load_bootstrap_trajectories(args.cache_dir)

    # Train each actor
    logger.info("[Step 3] Training specialized actors...")

    actor_paths = {}

    for thinking_style in args.thinking_styles:
        logger.info(f"\n--- Training Actor: {thinking_style} ---")

        # Build preference pairs for this style
        preference_pairs = build_preference_pairs_for_style(
            classified_results,
            trajectories,
            thinking_style,
            args.dataset,
        )

        if not preference_pairs:
            logger.warning(f"  No preference pairs for '{thinking_style}', skipping")
            continue

        # Train DPO
        checkpoint_path = train_actor_dpo(
            model_name=args.model_name,
            preference_pairs=preference_pairs,
            thinking_style=thinking_style,
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

        actor_paths[thinking_style] = checkpoint_path

    # Save actor registry
    logger.info("\n[Step 4] Saving actor registry...")

    registry_file = os.path.join(output_dir, "actor_registry.json")
    with open(registry_file, "w") as f:
        json.dump({
            "actors": {
                style: {
                    "thinking_style": style,
                    "model_path": path,
                    "base_model": args.model_name,
                }
                for style, path in actor_paths.items()
            },
            "metadata": {
                "base_model": args.model_name,
                "num_actors": len(actor_paths),
            },
        }, f, indent=2)

    logger.info(f"  Registry saved: {registry_file}")

    logger.info("=" * 60)
    logger.info("Actor diversification complete!")
    logger.info(f"  Trained {len(actor_paths)} actors:")
    for style, path in actor_paths.items():
        logger.info(f"    {style}: {path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
