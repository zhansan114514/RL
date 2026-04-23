"""
ACC-Collab full pipeline script.

Usage:
    python scripts/06_full_pipeline.py \
    --config configs/experiment_gpu1.yaml
"""

from __future__ import annotations

import json
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
    "dataset": "boolq",
    "cache_dir": "cache",
    "output_dir": None,          # auto-derived from cache_dir if not set
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
    "reuse_trajectories": False,
    "use_wandb": False,
    "actor_device": 0,
    "critic_device": 0,
    "dtype": "float32",
    "gpu_memory_utilization": 0.8,
    "max_model_len": 4096,
}


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="ACC-Collab Pipeline",
    )
    parser.add_argument(
        "--config", type=str, default="configs/experiment_gpu1.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()
    cfg = ConfigManager.initialize(config_path=cli_args.config)
    return cfg.step("step06", defaults=STEP_DEFAULTS).to_namespace()


def main():
    args = parse_args()

    # Derive output_dir from cache_dir if not explicitly set
    cache_dir = getattr(args, "cache_dir", "cache") or "cache"
    output_dir = getattr(args, "output_dir", None) or cache_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, "logs"), exist_ok=True)

    # Clean stale trajectory files if not reusing
    reuse = getattr(args, "reuse_trajectories", False)
    if not reuse:
        traj_dir = os.path.join(cache_dir, "trajectories")
        for f in os.listdir(traj_dir):
            if f.endswith(".jsonl"):
                os.remove(os.path.join(traj_dir, f))
                logger.info(f"Cleaned stale trajectory: {f}")

    logger.info("=" * 60)
    logger.info("ACC-Collab Pipeline")
    if args.config:
        logger.info(f"  Config: {args.config}")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Cache dir: {cache_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Reuse trajectories: {reuse}")
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

    # Fallback: if no test split, use validation set
    if not test_data and val_data:
        logger.warning("No test split found, using validation set for evaluation.")
        test_data = val_data

    logger.info(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    if not args.skip_training:
        # --- Step 2: Alternating training ---
        logger.info("[Step 2] Running alternating Actor-Critic training...")
        from src.training.scheduler import alternating_train
        from src.utils.model_utils import detect_model_type

        model_type = detect_model_type(args.model_name)

        result = alternating_train(
            actor_path=args.model_name,
            critic_path=args.model_name,
            dataset=train_data,
            dataset_name=args.dataset,
            output_base_dir=output_dir,
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
            actor_device=args.actor_device,
            critic_device=args.critic_device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cache_dir=cache_dir,
            reuse_trajectories=reuse,
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

    actor_model = VLLMInference(
        actor_path,
        cuda_device=args.actor_device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    critic_model = VLLMInference(
        critic_path,
        cuda_device=args.critic_device,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    results = evaluate_benchmark(
        actor_model, critic_model, test_data, args.dataset,
        num_rounds=args.num_rounds,
    )

    logger.info("Results:")
    for k, v in results.items():
        if k != "sample_details":
            logger.info(f"  {k}: {v}")

    # Save full results (including sample details)
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {results_path}")

    # Save a separate summary file (compact, no sample details)
    summary = {k: v for k, v in results.items() if k != "sample_details"}
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")

    # Print final experiment summary
    _print_final_summary(args, results, output_dir)


def _print_final_summary(args, results: dict, output_dir: str) -> None:
    """Print a comprehensive final experiment summary."""
    import time
    sep = "#" * 70

    logger.info("")
    logger.info(sep)
    logger.info("  EXPERIMENT COMPLETE - FINAL SUMMARY")
    logger.info(sep)
    logger.info(f"  Model:        {args.model_name}")
    logger.info(f"  Dataset:      {args.dataset}")
    logger.info(f"  Iterations:   {args.num_iterations}")
    logger.info(f"  Rounds:       {args.num_rounds}")
    logger.info(f"  Simulations:  {args.num_simulations}")
    logger.info(f"  Train samples:{args.max_samples or 'all'}")
    logger.info(f"  Test samples: {results['num_samples']}")
    logger.info(f"  Output dir:   {output_dir}")
    logger.info(sep[:-1])
    logger.info(f"  Initial Accuracy:  {results['initial_accuracy']:.4f} ({results['initial_accuracy']*100:.2f}%)")
    logger.info(f"  Final Accuracy:    {results['final_accuracy']:.4f} ({results['final_accuracy']*100:.2f}%)")
    logger.info(f"  95% CI:            {results['ci_95']}")
    logger.info(f"  Improvement Rate:  {results['improvement_rate']:.4f} ({results['improvement_rate']*100:.2f}%)")
    logger.info(f"  Absolute Gain:     {results['absolute_improvement']:+.4f} ({results['absolute_improvement']*100:+.2f}pp)")
    logger.info(f"  Per-round acc:     {[f'{a:.3f}' for a in results['per_round_accuracy']]}")
    flip = results['flip_statistics']
    logger.info(f"  Flipped correct:   {flip['flipped_to_correct']}/{results['num_samples']}")
    logger.info(f"  Flipped wrong:     {flip['flipped_to_wrong']}/{results['num_samples']}")
    logger.info(f"  Eval time:         {results['eval_time_seconds']}s")
    logger.info(sep)
    logger.info("")
    logger.info(f"Full results: {os.path.join(output_dir, 'results.json')}")
    logger.info(f"Summary:      {os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
