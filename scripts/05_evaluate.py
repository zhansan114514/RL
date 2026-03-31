"""
Step 5: Evaluate Actor-Critic team on benchmark datasets.

Loads trained actor and critic models, runs deliberation on test data,
and reports accuracy with confidence intervals.

Usage:
    python scripts/05_evaluate.py \
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

COMMON_KEYS = ("dataset", "max_samples", "seed")

STEP_DEFAULTS = {
    "actor_path": "experiments/gemma2_boolq/actor",
    "critic_path": "experiments/gemma2_boolq/critic",
    "dataset": "boolq",
    "output_dir": "experiments/gemma2_boolq/eval",
    "num_rounds": 5,
    "max_tokens": 256,
    "temperature": 0.7,
    "max_samples": None,
    "seed": 42,
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
    step_cfg = cfg.get("step05", {})

    if not isinstance(common_cfg, dict):
        raise ValueError("Config key 'common' must be a mapping.")
    if not isinstance(step_cfg, dict):
        raise ValueError("Config key 'step05' must be a mapping.")

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
    parser = argparse.ArgumentParser(description="Evaluate Actor-Critic team")
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
    test_data = data.get("test", [])
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    logger.info(f"  Test samples: {len(test_data)}")

    logger.info(f"Loading actor: {args.actor_path}")
    logger.info(f"Loading critic: {args.critic_path}")
    from src.inference.vllm_server import VLLMInference

    actor = VLLMInference(args.actor_path)
    critic = VLLMInference(args.critic_path)

    logger.info("Evaluating...")
    from src.evaluation.benchmarks import evaluate_benchmark

    results = evaluate_benchmark(
        actor, critic, test_data, args.dataset,
        num_rounds=args.num_rounds,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    logger.info("=" * 50)
    logger.info(f"Dataset: {results['dataset']}")
    logger.info(f"Samples: {results['num_samples']}")
    logger.info(f"Initial accuracy: {results['initial_accuracy']:.3f}")
    logger.info(f"Final accuracy:   {results['final_accuracy']:.3f} +/- {results['ci_margin']:.3f}")
    logger.info(f"Improvement rate:  {results['improvement_rate']:.3f}")
    logger.info(f"Per-round: {[f'{a:.3f}' for a in results['per_round_accuracy']]}")
    logger.info("=" * 50)

    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
