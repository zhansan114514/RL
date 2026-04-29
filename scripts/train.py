"""
Unified training entry point for Actor and Critic DPO training.

Usage:
    python scripts/train.py --agent critic --config configs/config.yaml
    python scripts/train.py --agent actor --config configs/config.yaml
"""

from __future__ import annotations

import logging
import os

from _utils import setup_logging
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)

# Defaults shared between actor and critic; only preference_dir and
# output_dir differ and are filled in parse_args() based on --agent.
_STEP_DEFAULTS_COMMON = {
    "model_name": "google/gemma-2-2b-it",
    "model_type": None,
    "lora_r": 256,
    "learning_rate": 5e-5,
    "batch_size": 1,
    "num_epochs": 1,
    "max_length": 1024,
    "seed": 42,
    "use_wandb": False,
}

AGENT_DEFAULTS = {
    "critic": {
        **_STEP_DEFAULTS_COMMON,
        "preference_dir": "experiments/gemma2_boolq/preferences",
        "output_dir": "experiments/gemma2_boolq/critic",
    },
    "actor": {
        **_STEP_DEFAULTS_COMMON,
        "preference_dir": "experiments/gemma2_boolq/preferences",
        "output_dir": "experiments/gemma2_boolq/actor",
    },
}

# Mapping from agent name to config step key and preference subdirectory.
AGENT_CONFIG = {
    "critic": {"step_key": "step03", "pref_subdir": "critic_preferences"},
    "actor": {"step_key": "step04", "pref_subdir": "actor_preferences"},
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified DPO training for Actor or Critic",
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=["actor", "critic"],
        help="Which agent to train: 'actor' or 'critic'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()

    agent = cli_args.agent
    step_key = AGENT_CONFIG[agent]["step_key"]
    defaults = AGENT_DEFAULTS[agent]

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    args = cfg.step(step_key, defaults=defaults).to_namespace()
    return args, agent


def main():
    args, agent = parse_args()

    # Fix seed for reproducibility
    from src.utils.seeding import fix_seed
    fix_seed(args.seed)

    from src.utils.model_utils import detect_model_type
    model_type = args.model_type or detect_model_type(args.model_name)
    logger.info(f"Agent: {agent} | Model: {args.model_name} (type={model_type})")

    from datasets import load_from_disk

    pref_subdir = AGENT_CONFIG[agent]["pref_subdir"]
    pref_path = os.path.join(args.preference_dir, pref_subdir)
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
    logger.info(f"{agent.capitalize()} saved: {output}")


if __name__ == "__main__":
    main()
