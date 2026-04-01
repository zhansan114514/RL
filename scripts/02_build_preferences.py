"""
Step 2: Build preference datasets from trajectory pairs.

Reads trajectory_pairs.json, splits by agent (actor/critic),
and saves HuggingFace-format preference datasets.

Usage:
    python scripts/02_build_preferences.py \
    --config configs/config.yaml
"""

from __future__ import annotations

import json
import logging
import os

from _utils import resolve_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "input_dir": "experiments/gemma2_boolq/trajectories",
    "output_dir": "experiments/gemma2_boolq/preferences",
    "min_delta": 0.0,
}


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Build preference datasets",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    return resolve_config(cli_args.config, "step02", STEP_DEFAULTS)


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

    from src.trajectory.preference import build_preference_dataset, convert_to_hf_dataset

    for agent in ["actor", "critic"]:
        prefs = build_preference_dataset(all_pairs, min_delta=args.min_delta, agent=agent)
        if not prefs:
            logger.warning(f"No preference pairs for {agent}, skipping")
            continue

        hf_dataset = convert_to_hf_dataset(prefs)
        output_path = os.path.join(args.output_dir, f"{agent}_preferences")
        hf_dataset.save_to_disk(output_path)
        logger.info(f"  {agent}: {len(prefs)} pairs -> {output_path}")


if __name__ == "__main__":
    main()
