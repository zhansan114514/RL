"""
ACC-Collab full pipeline script.

Usage:
    python scripts/06_full_pipeline.py \
    --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


ALLOWED_DATASETS = ("boolq", "mmlu", "bbh", "sciq", "arc")

COMMON_KEYS = ("model_name", "dataset", "max_samples", "seed", "use_wandb")

STEP_DEFAULTS = {
    "model_name": "google/gemma-2-2b-it",
    "dataset": "boolq",
    "output_dir": "experiments/gemma2_boolq",
    "num_iterations": 1,
    "num_rounds": 5,
    "lora_r": 256,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "num_epochs": 1,
    "beta": 0.1,
    "num_simulations": 5,
    "reward_threshold": 0.0,
    "seed": 42,
    "max_samples": None,
    "skip_training": False,
    "use_wandb": False,
}


def _load_yaml_config(config_path: str) -> dict:
    """Load YAML config for pipeline runtime arguments."""
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
    step_cfg = cfg.get("step06", {})

    if not isinstance(common_cfg, dict):
        raise ValueError("Config key 'common' must be a mapping.")
    if not isinstance(step_cfg, dict):
        raise ValueError("Config key 'step06' must be a mapping.")

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
    parser = argparse.ArgumentParser(description="ACC-Collab Pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    return _resolve_config(cli_args.config)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ACC-Collab Pipeline")
    if args.config:
        logger.info(f"  Config: {args.config}")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60)

    # --- Step 1: Load dataset ---
    logger.info("[Step 1] Loading dataset...")
    from src.data.loader import load_dataset

    data = load_dataset(args.dataset, seed=args.seed)
    train_data = data.get("train", [])
    val_data = data.get("validation", [])
    test_data = data.get("test", [])

    if args.max_samples:
        train_data = train_data[:args.max_samples]
        val_data = val_data[:args.max_samples]
        test_data = test_data[:args.max_samples]

    logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    if not args.skip_training:
        # --- Step 2: Alternating training ---
        logger.info("[Step 2] Running alternating Actor-Critic training...")
        from src.training.alternating import alternating_train
        from src.utils.model_utils import detect_model_type

        # Detect model type from name
        model_type = detect_model_type(args.model_name)

        result = alternating_train(
            actor_path=args.model_name,
            critic_path=args.model_name,
            dataset=train_data,
            dataset_name=args.dataset,
            output_base_dir=args.output_dir,
            model_type=model_type,
            num_iterations=args.num_iterations,
            num_rounds=args.num_rounds,
            reward_threshold=args.reward_threshold,
            num_simulations=args.num_simulations,
            lora_r=args.lora_r,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            beta=args.beta,
            seed=args.seed,
            val_dataset=val_data if val_data else None,
            early_stopping_patience=getattr(args, "early_stopping_patience", None),
        )
        actor_path = result["actor_path"]
        critic_path = result["critic_path"]
        logger.info(f"  Final actor: {actor_path}")
        logger.info(f"  Final critic: {critic_path}")
    else:
        actor_path = args.model_name
        critic_path = args.model_name

    # --- Step 3: Evaluation ---
    logger.info("[Step 3] Evaluating...")
    from src.inference.vllm_server import VLLMInference
    from src.evaluation.benchmarks import evaluate_benchmark

    actor_model = VLLMInference(actor_path)
    critic_model = VLLMInference(critic_path)

    results = evaluate_benchmark(
        actor_model, critic_model, test_data, args.dataset,
        num_rounds=args.num_rounds,
    )

    logger.info("Results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    # Save results
    import json
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
