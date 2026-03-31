"""
Step 1: Generate deliberation trajectories.

For each training sample, run natural + guided deliberation and
save the raw trajectory data for subsequent preference pair construction.

Usage:
    python scripts/01_generate_trajectories.py \
    --config configs/config.yaml
"""

from __future__ import annotations

import json
import logging
import os
import traceback

from _utils import resolve_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

ALLOWED_DATASETS = ("boolq", "mmlu", "bbh", "sciq", "arc")
COMMON_KEYS = ("model_name", "dataset", "max_samples", "seed")

STEP_DEFAULTS = {
    "model_name": "google/gemma-2-2b-it",
    "dataset": "boolq",
    "output_dir": "experiments/gemma2_boolq/trajectories",
    "num_rounds": 5,
    "num_simulations": 5,
    "reward_threshold": 0.0,
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
        description="Generate deliberation trajectories",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    return resolve_config(
        cli_args.config, "step01", STEP_DEFAULTS,
        common_keys=COMMON_KEYS,
        allowed_datasets=ALLOWED_DATASETS,
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading dataset: {args.dataset}")
    from src.data.loader import load_dataset

    data = load_dataset(args.dataset, seed=args.seed)
    train_data = data.get("train", [])
    if args.max_samples:
        train_data = train_data[:args.max_samples]
    logger.info(f"  Samples: {len(train_data)}")

    logger.info(f"Loading models: {args.model_name}")
    logger.info(f"  Actor on GPU {args.actor_device}, Critic on GPU {args.critic_device}")
    from src.inference.vllm_server import VLLMInference

    actor = VLLMInference(
        args.model_name,
        cuda_device=args.actor_device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    critic = VLLMInference(
        args.model_name,
        cuda_device=args.critic_device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    logger.info("Generating trajectories...")
    from src.trajectory.generator import generate_trajectories

    all_pairs = []
    for i, sample in enumerate(train_data):
        logger.info(f"  [{i+1}/{len(train_data)}]")
        try:
            pairs = generate_trajectories(
                actor, critic, sample, args.dataset,
                num_rounds=args.num_rounds,
                reward_threshold=args.reward_threshold,
                num_simulations=args.num_simulations,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            all_pairs.extend(pairs)
        except Exception as e:
            logger.warning(f"  Sample {i} failed: {e}")
            logger.debug(f"  Traceback:\n{traceback.format_exc()}")

    output_path = os.path.join(args.output_dir, "trajectory_pairs.json")
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    logger.info(f"Generated {len(all_pairs)} preference pairs")
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
