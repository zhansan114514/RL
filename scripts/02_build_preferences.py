"""
Step 2: Build preference datasets from trajectory pairs.

Reads trajectory_pairs.json, splits by agent (actor/critic),
and saves HuggingFace-format preference datasets.

Usage:
    python scripts/02_build_preferences.py \
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


STEP_DEFAULTS = {
    "input_dir": "experiments/gemma2_boolq/trajectories",
    "output_dir": "experiments/gemma2_boolq/preferences",
    "min_delta": 0.0,
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
    step_cfg = cfg.get("step02", {})
    if not isinstance(step_cfg, dict):
        raise ValueError("Config key 'step02' must be a mapping.")

    merged = dict(STEP_DEFAULTS)
    for key in STEP_DEFAULTS:
        if key in step_cfg and step_cfg[key] is not None:
            merged[key] = step_cfg[key]

    return argparse.Namespace(config=config_path, **merged)


def parse_args():
    parser = argparse.ArgumentParser(description="Build preference datasets")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    return _resolve_config(cli_args.config)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    input_path = os.path.join(args.input_dir, "trajectory_pairs.json")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        logger.error("Please run scripts/01_generate_trajectories.py first")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading: {input_path}")
    with open(input_path) as f:
        all_pairs = json.load(f)

    logger.info(f"Total pairs: {len(all_pairs)}")

    from src.trajectory.preference import build_preference_dataset, to_hf_dataset

    for agent in ["actor", "critic"]:
        prefs = build_preference_dataset(all_pairs, min_delta=args.min_delta, agent=agent)
        if not prefs:
            logger.warning(f"No preference pairs for {agent}, skipping")
            continue

        hf_dataset = to_hf_dataset(prefs)
        output_path = os.path.join(args.output_dir, f"{agent}_preferences")
        hf_dataset.save_to_disk(output_path)
        logger.info(f"  {agent}: {len(prefs)} pairs -> {output_path}")


if __name__ == "__main__":
    main()
