"""
Step 3: Train the Critic model with DPO.

Loads the preference dataset for critic and runs DPO training.

Usage:
    python scripts/03_train_critic.py \
    --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


COMMON_KEYS = ("model_name", "seed", "use_wandb")

STEP_DEFAULTS = {
    "model_name": "google/gemma-2-2b-it",
    "preference_dir": "experiments/gemma2_boolq/preferences",
    "output_dir": "experiments/gemma2_boolq/critic",
    "model_type": None,
    "lora_r": 256,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "num_epochs": 1,
    "max_length": 2048,
    "seed": 42,
    "use_wandb": False,
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
    step_cfg = cfg.get("step03", {})

    if not isinstance(common_cfg, dict):
        raise ValueError("Config key 'common' must be a mapping.")
    if not isinstance(step_cfg, dict):
        raise ValueError("Config key 'step03' must be a mapping.")

    merged = dict(STEP_DEFAULTS)

    for key in COMMON_KEYS:
        if key in common_cfg and common_cfg[key] is not None:
            merged[key] = common_cfg[key]

    for key in STEP_DEFAULTS:
        if key in step_cfg and step_cfg[key] is not None:
            merged[key] = step_cfg[key]

    return argparse.Namespace(config=config_path, **merged)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Critic with DPO")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    return _resolve_config(cli_args.config)


def main():
    args = parse_args()

    from src.utils.model_utils import detect_model_type
    model_type = args.model_type or detect_model_type(args.model_name)
    logger.info(f"Model: {args.model_name} (type={model_type})")

    # Load preference dataset
    from datasets import load_from_disk

    pref_path = os.path.join(args.preference_dir, "critic_preferences")
    if not os.path.exists(pref_path):
        logger.error(f"Preference dataset not found: {pref_path}")
        logger.error("Please run scripts/02_build_preferences.py first")
        raise FileNotFoundError(f"Preference dataset not found: {pref_path}")

    logger.info(f"Loading preferences: {pref_path}")
    dataset = load_from_disk(pref_path)
    logger.info(f"  Pairs: {len(dataset)}")

    from src.training.dpo_trainer import train_dpo

    output = train_dpo(
        model_name_or_path=args.model_name,
        preference_dataset=dataset,
        output_dir=args.output_dir,
        model_type=model_type,
        lora_r=args.lora_r,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        seed=args.seed,
        use_wandb=args.use_wandb,
    )
    logger.info(f"Critic saved: {output}")


if __name__ == "__main__":
    # Restrict to single GPU BEFORE any CUDA initialization to avoid NVML errors on V100
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
