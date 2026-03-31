"""
Step 1: Generate deliberation trajectories.

For each training sample, run natural + guided deliberation and
save the raw trajectory data for subsequent preference pair construction.

Usage:
    python scripts/01_generate_trajectories.py \
    --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
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
    "actor_device": 13,
    "critic_device": 13,
    "dtype": "float32",
    "gpu_memory_utilization": 0.45,
}


def _load_yaml_config(config_path: str) -> dict:
    omegaconf_module = importlib.import_module("omegaconf")
    OmegaConf = omegaconf_module.OmegaConf

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    loaded = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(loaded, resolve=True)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a top-level key-value mapping.")
    return cfg


def _resolve_config(config_path: str) -> argparse.Namespace:
    cfg = _load_yaml_config(config_path)
    common_cfg = cfg.get("common", {})
    step_cfg = cfg.get("step01", {})

    if not isinstance(common_cfg, dict):
        raise ValueError("Config key 'common' must be a mapping.")
    if not isinstance(step_cfg, dict):
        raise ValueError("Config key 'step01' must be a mapping.")

    merged = dict(STEP_DEFAULTS)

    for key in COMMON_KEYS:
        if key in common_cfg and common_cfg[key] is not None:
            merged[key] = common_cfg[key]

    for key in STEP_DEFAULTS:
        if key in step_cfg and step_cfg[key] is not None:
            merged[key] = step_cfg[key]

    if merged["dataset"] not in ALLOWED_DATASETS:
        raise ValueError(
            f"Invalid dataset '{merged['dataset']}'. "
            f"Expected one of: {', '.join(ALLOWED_DATASETS)}"
        )

    return argparse.Namespace(config=config_path, **merged)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate deliberation trajectories")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    return _resolve_config(cli_args.config)


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
            import traceback
            logger.warning(f"  Sample {i} failed: {e}")
            logger.debug(f"  Traceback:\n{traceback.format_exc()}")

    output_path = os.path.join(args.output_dir, "trajectory_pairs.json")
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)

    logger.info(f"Generated {len(all_pairs)} preference pairs")
    logger.info(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
