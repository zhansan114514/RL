"""
Run only A5 ablation: 3 Actors + 4 Critics + Router (full system).
Skips all other ablations for quick validation.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "mmlu",
    "seed": 42,
    "device": 5,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.60,
    "max_model_len": 4096,
    "society_dir": "output/society_mmlu/society",
    "output_dir": "output/society_mmlu/eval",
    "num_rounds": 2,
    "max_tokens": 512,
    "temperature": 0.7,
    "max_samples": None,
    "router_top_k": 2,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/society/experiment_mmlu.yaml")
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    args = cfg.step("step06_evaluate", defaults=STEP_DEFAULTS).to_namespace()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("A5 Evaluation: 3 Actors + 4 Critics + Router")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    # Load registry
    registry_path = os.path.join(args.society_dir, "final_agent_registry.json")
    with open(registry_path) as f:
        registry = json.load(f)

    base_model = registry.get("training_config", {}).get("base_model", args.model_name)
    all_actor_names = list(registry["actors"].keys())
    all_critic_names = list(registry["critics"].keys())

    logger.info(f"  Actors: {all_actor_names}")
    logger.info(f"  Critics: {all_critic_names}")

    # Load dataset
    from src.data.loader import load_dataset as load_data
    data = load_data(args.dataset, seed=args.seed)
    samples = data.get("test", []) or data.get("validation", [])
    if args.max_samples:
        samples = samples[:args.max_samples]
    logger.info(f"  Eval samples: {len(samples)}")

    # Import helpers from 12_society_evaluate
    from importlib import import_module
    eval_module = import_module("scripts.12_society_evaluate")
    _build_agent_configs = eval_module._build_agent_configs
    _run_deliberation_on_samples = eval_module._run_deliberation_on_samples

    # Build engine
    from src.inference.vllm_server import VLLMInference
    total_lora = len(all_actor_names) + len(all_critic_names)

    logger.info("Loading engine...")
    engine = VLLMInference(
        base_model,
        cuda_device=args.device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_lora=True,
        max_loras=total_lora,
        max_lora_rank=256,
    )

    try:
        a_configs, c_configs, lora = _build_agent_configs(
            registry,
            actor_names=all_actor_names[:3],
            critic_names=all_critic_names,
        )

        logger.info("[A5] Running 3 Actors + 4 Critics + Router...")
        result = _run_deliberation_on_samples(
            engine, a_configs, c_configs, samples, args.dataset,
            lora, args.num_rounds, args.max_tokens, args.temperature,
            router_top_k=args.router_top_k,
            router_uniform=False,
        )

        logger.info(f"  A5: initial={result.initial_accuracy:.3f} final={result.final_accuracy:.3f}")

        # Save results
        results_file = os.path.join(output_dir, "a5_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "A5_full_system": {
                    "initial_accuracy": result.initial_accuracy,
                    "final_accuracy": result.final_accuracy,
                    "improvement": result.final_accuracy - result.initial_accuracy,
                    "num_samples": len(samples),
                    "num_rounds": args.num_rounds,
                },
            }, f, indent=2)
        logger.info(f"  Results saved to {results_file}")

    finally:
        del engine
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    main()
