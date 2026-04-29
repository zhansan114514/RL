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
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "output/society",
    "input_dir": "output/society/classified",
    "output_dir": "output/society/actors",
    "reasoning_styles": ["algebraic", "direct", "backtracking"],
    "lora_r": 256,
    "lora_alpha": 512,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
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

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    return cfg.step("step03_diversify_actors", defaults=STEP_DEFAULTS).to_namespace()


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

    Uses per-response labels (not sample-level aggregates) so that
    each Actor trains only on responses whose style matches its own.

    Chosen: correct response with matching style
    Rejected: incorrect response (from same sample)

    If too few style-specific responses are found, returns an empty list.
    """
    from src.algorithms.reward import extract_answer

    preference_pairs = []

    # Build a per-response style lookup:
    # sample_id -> {response_index: reasoning_style}
    per_response_lookup: Dict[str, Dict[int, str]] = {}
    for r in classified_results:
        sid = r["sample_id"]
        per_response_lookup[sid] = {}
        for ri, label in enumerate(r.get("per_response_labels", [])):
            if label.get("reasoning_style"):
                per_response_lookup[sid][ri] = label["reasoning_style"]

    # Collect all samples that have at least one response matching this style
    matched_sample_count = 0
    for r in classified_results:
        sample_id = r["sample_id"]
        resp_labels = r.get("per_response_labels", [])

        # Check if any response in this sample matches the target style
        has_matching_style = any(
            label.get("reasoning_style") == thinking_style
            for label in resp_labels
        )
        if has_matching_style:
            matched_sample_count += 1

    # Strict: skip if too few style-specific samples
    if matched_sample_count < 2:
        logger.warning(
            f"  Only {matched_sample_count} samples with style '{thinking_style}', "
            f"skipping (need at least 2 for meaningful specialization)"
        )
        return preference_pairs

    logger.info(f"  Found {matched_sample_count} samples with style '{thinking_style}'")

    for r in classified_results:
        sample_id = r["sample_id"]
        resp_labels = r.get("per_response_labels", [])

        if sample_id not in trajectories:
            continue

        traj = trajectories[sample_id]
        sample = traj["sample"]
        correct_answer = sample.get("answer", "")
        task_type = sample.get("task_type", "math")

        # Get responses from all rounds (initial + debate) for more diversity
        all_responses = list(traj.get("initial_responses", []))
        for round_responses in traj.get("debate_rounds", []):
            all_responses.extend(round_responses)

        if not all_responses:
            continue

        # Collect style-matching correct responses and any incorrect responses
        style_correct_responses = []
        incorrect_responses = []
        for ri, resp in enumerate(all_responses):
            response_text = resp.get("response", "")
            extracted_answer = extract_answer(response_text, task_type)

            from src.algorithms.reward import math_answers_equal
            if task_type == "math":
                is_correct = math_answers_equal(extracted_answer or "", correct_answer)
            else:
                is_correct = (extracted_answer or "").upper() == (correct_answer or "").upper()

            if is_correct:
                # Only include as "chosen" if the per-response label matches this style
                resp_style = per_response_lookup.get(sample_id, {}).get(ri)
                if resp_style == thinking_style:
                    style_correct_responses.append(response_text)
                # Skip correct responses that don't match this style
            else:
                incorrect_responses.append(response_text)

        chosen_response = None
        rejected_response = None

        if style_correct_responses and incorrect_responses:
            # Ideal: style-matching correct vs incorrect — strong preference signal
            chosen_response = style_correct_responses[0]
            rejected_response = incorrect_responses[0]
        elif len(style_correct_responses) >= 2:
            # All matching-style correct: no meaningful preference signal.
            continue
        elif len(incorrect_responses) >= 2:
            # All incorrect: no meaningful preference signal, skip
            continue

        if chosen_response and rejected_response and chosen_response != rejected_response:
            preference_pairs.append({
                "sample": sample,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "metadata": {
                    "thinking_style": thinking_style,
                    "sample_id": sample_id,
                },
            })

    logger.info(f"  Built {len(preference_pairs)} preference pairs")
    return preference_pairs


def train_actor_dpo(
    model_name: str,
    preference_pairs: List[Dict],
    thinking_style: str,
    output_dir: str,
    dataset_name: str,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
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
    from src.prompts.formatter import format_prompt
    from src.prompts.templates import PromptType

    model_type = detect_model_type(model_name)

    # Create output directory
    actor_output_dir = os.path.join(output_dir, f"actor_{thinking_style}")
    os.makedirs(actor_output_dir, exist_ok=True)

    logger.info(f"  Training DPO for '{thinking_style}'...")
    logger.info(f"    Pairs: {len(preference_pairs)}")
    logger.info(f"    Output: {actor_output_dir}")

    # Reconstruct full prompts using the same template as generation
    prompts = [
        format_prompt(dataset_name, PromptType.SINGLE_SHOT, p["sample"])
        for p in preference_pairs
    ]

    # Convert preference_pairs to HuggingFace Dataset
    hf_data = {
        "prompt": prompts,
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
        gradient_accumulation_steps=gradient_accumulation_steps,
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
    logger.info(f"  Reasoning styles: {args.reasoning_styles}")
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

    for thinking_style in args.reasoning_styles:
        logger.info(f"\n--- Training Actor: {thinking_style} ---")

        # Check for cached preference pairs
        pairs_cache = os.path.join(output_dir, f"pairs_{thinking_style}.json")
        if os.path.exists(pairs_cache):
            with open(pairs_cache) as f:
                preference_pairs = json.load(f)
            logger.info(f"  Loaded {len(preference_pairs)} cached pairs for '{thinking_style}'")
        else:
            # Build preference pairs for this style
            preference_pairs = build_preference_pairs_for_style(
                classified_results,
                trajectories,
                thinking_style,
                args.dataset,
            )

            if preference_pairs:
                with open(pairs_cache, "w") as f:
                    json.dump(preference_pairs, f)
                logger.info(f"  Cached {len(preference_pairs)} pairs to {pairs_cache}")

        if not preference_pairs:
            logger.warning(f"  No preference pairs for '{thinking_style}', skipping")
            continue

        # Train DPO
        checkpoint_path = train_actor_dpo(
            model_name=args.model_name,
            preference_pairs=preference_pairs,
            thinking_style=thinking_style,
            output_dir=output_dir,
            dataset_name=args.dataset,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
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
