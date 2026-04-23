"""
Step 4: Train the Actor model with DPO.

Loads the preference dataset for actor and runs DPO training.

Usage:
    python scripts/04_train_actor.py \
    --config configs/config.yaml
"""

from __future__ import annotations

import logging
import os

from _utils import setup_logging
from src.utils.config import ConfigManager

# Apply NVML fix if needed (for PyTorch 2.10+ with old NVIDIA drivers)
try:
    from src.utils import nvml_fix
    nvml_fix.auto_apply_nvml_fix()
except ImportError:
    pass  # NVML fix module not available

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "google/gemma-2-2b-it",
    "preference_dir": "experiments/gemma2_boolq/preferences",
    "output_dir": "experiments/gemma2_boolq/actor",
    "model_type": None,
    "lora_r": 256,
    "learning_rate": 5e-5,
    "batch_size": 1,
    "num_epochs": 1,
    "max_length": 1024,
    "seed": 42,
    "use_wandb": False,
}


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Train Actor with DPO",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    cfg = ConfigManager.initialize(config_path=cli_args.config)
    return cfg.step("step04", defaults=STEP_DEFAULTS).to_namespace()


def main():
    args = parse_args()

    from src.utils.model_utils import detect_model_type
    model_type = args.model_type or detect_model_type(args.model_name)
    logger.info(f"Model: {args.model_name} (type={model_type})")

    from datasets import load_from_disk

    pref_path = os.path.join(args.preference_dir, "actor_preferences")
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
    logger.info(f"Actor saved: {output}")


if __name__ == "__main__":
    main()
