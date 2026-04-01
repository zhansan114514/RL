"""
Alternating training scheduler for Actor-Critic.

Implements the iterative best-response optimization:
1. Fix actor -> train critic
2. Fix critic -> train actor
3. Repeat (ACC-Collab+ does 2 rounds)

Refactored from alternating.py to use model_manager and trainer.
All hardcoded parameters (max_model_len, gpu_memory_utilization, etc.)
are now read from ConfigManager.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.training.model_manager import create_model_pair, cleanup_models
from src.training.trainer import train_agent

logger = logging.getLogger(__name__)


def alternating_train(
    actor_path: str,
    critic_path: str,
    dataset: list[dict],
    dataset_name: str,
    output_base_dir: str,
    model_type: str = "gemma2",
    num_iterations: int = 1,
    num_rounds: int = 5,
    reward_threshold: float = 0.0,
    num_simulations: int = 5,
    lora_r: int = 256,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    num_epochs: int = 1,
    beta: float = 0.1,
    max_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 42,
    val_dataset: Optional[list[dict]] = None,
    early_stopping_patience: Optional[int] = None,
    min_improvement: float = 0.0,
    actor_device: int = 0,
    critic_device: int = 1,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.45,
) -> dict[str, str]:
    """
    Run alternating Actor-Critic training.

    Args:
        actor_path: Path to actor base model.
        critic_path: Path to critic base model.
        dataset: List of standardized samples (training set).
        dataset_name: Dataset name for prompts.
        output_base_dir: Base directory for outputs.
        model_type: Model architecture type.
        num_iterations: Number of alternating rounds (1=ACC-Collab, 2=ACC-Collab+).
        num_rounds: Deliberation rounds per sample.
        reward_threshold: Minimum delta for preference pairs.
        num_simulations: MC roll-out simulations.
        lora_r: LoRA rank.
        learning_rate: Learning rate.
        batch_size: Training batch size.
        num_epochs: Epochs per training step.
        beta: DPO beta parameter.
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.
        seed: Random seed.
        val_dataset: Optional validation dataset for performance monitoring.
        early_stopping_patience: Optional patience for early stopping.
        min_improvement: Minimum improvement threshold to reset patience.
        actor_device: CUDA device index for actor.
        critic_device: CUDA device index for critic.
        dtype: Model dtype for inference.
        gpu_memory_utilization: GPU memory fraction for vLLM.

    Returns:
        Dict with final actor_path, critic_path, and validation_metrics.
    """
    current_actor_path = actor_path
    current_critic_path = critic_path

    best_val_acc = 0.0
    patience_counter = 0
    val_metrics_history = []

    for iteration in range(num_iterations):
        logger.info(f"=== Iteration {iteration + 1}/{num_iterations} ===")

        # Step 1: Train critic (fix actor)
        logger.info("Step 1: Training Critic (actor fixed)")
        actor_model, critic_model = create_model_pair(
            current_actor_path, current_critic_path,
            actor_device=actor_device, critic_device=critic_device,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
        )

        current_critic_path = train_agent(
            "critic",
            actor_model, critic_model, dataset, dataset_name,
            current_critic_path, output_base_dir, iteration, model_type,
            num_rounds=num_rounds, reward_threshold=reward_threshold,
            num_simulations=num_simulations, max_tokens=max_tokens,
            temperature=temperature, seed=seed,
            lora_r=lora_r, learning_rate=learning_rate,
            batch_size=batch_size, num_epochs=num_epochs, beta=beta,
        )

        cleanup_models(actor_model, critic_model)

        # Step 2: Train actor (fix critic)
        logger.info("Step 2: Training Actor (critic fixed)")
        actor_model, critic_model = create_model_pair(
            current_actor_path, current_critic_path,
            actor_device=actor_device, critic_device=critic_device,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
        )

        current_actor_path = train_agent(
            "actor",
            actor_model, critic_model, dataset, dataset_name,
            current_actor_path, output_base_dir, iteration, model_type,
            num_rounds=num_rounds, reward_threshold=reward_threshold,
            num_simulations=num_simulations, max_tokens=max_tokens,
            temperature=temperature, seed=seed,
            lora_r=lora_r, learning_rate=learning_rate,
            batch_size=batch_size, num_epochs=num_epochs, beta=beta,
        )

        cleanup_models(actor_model, critic_model)

        # Validation
        if val_dataset is not None:
            val_metrics = _run_validation(
                current_actor_path, current_critic_path,
                val_dataset, dataset_name,
                actor_device=actor_device, critic_device=critic_device,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                num_rounds=num_rounds, max_tokens=max_tokens,
                temperature=temperature,
            )
            val_metrics_history.append(val_metrics)
            val_acc = val_metrics["val_accuracy"]

            logger.info(
                f"  Iteration {iteration} Val Accuracy: "
                f"{val_acc:.3f} +/- {val_metrics['val_ci_margin']:.3f}"
            )

            # Early stopping check
            if early_stopping_patience is not None:
                stop = _check_early_stopping(
                    val_acc, best_val_acc, min_improvement,
                    patience_counter, early_stopping_patience,
                )
                if stop["improved"]:
                    best_val_acc = val_acc
                    patience_counter = 0
                    logger.info(f"  New best validation accuracy: {best_val_acc:.3f}")
                else:
                    patience_counter = stop["patience_counter"]
                    logger.info(
                        f"  No improvement for {patience_counter}/{early_stopping_patience} iterations"
                    )
                    if stop["should_stop"]:
                        logger.info("Early stopping triggered!")
                        break

    result = {
        "actor_path": current_actor_path,
        "critic_path": current_critic_path,
    }
    if val_metrics_history:
        result["validation_metrics"] = val_metrics_history
        result["best_val_accuracy"] = best_val_acc

    return result


def _run_validation(
    actor_path: str,
    critic_path: str,
    val_dataset: list[dict],
    dataset_name: str,
    *,
    actor_device: int = 0,
    critic_device: int = 0,
    gpu_memory_utilization: float = 0.45,
    dtype: str = "auto",
    num_rounds: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> dict:
    """Run validation evaluation and return metrics."""
    from src.evaluation.benchmarks import evaluate_benchmark

    logger.info(f"Evaluating on validation set ({len(val_dataset)} samples)...")
    actor_model, critic_model = create_model_pair(
        actor_path, critic_path,
        actor_device=actor_device, critic_device=critic_device,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )

    try:
        val_results = evaluate_benchmark(
            actor_model, critic_model, val_dataset, dataset_name,
            num_rounds=num_rounds, max_tokens=max_tokens,
            temperature=temperature,
        )
        return {
            "iteration": 0,  # Will be set by caller
            "val_accuracy": val_results["final_accuracy"],
            "val_ci_margin": val_results["ci_margin"],
            "improvement_rate": val_results["improvement_rate"],
        }
    finally:
        cleanup_models(actor_model, critic_model)


def _check_early_stopping(
    val_acc: float,
    best_val_acc: float,
    min_improvement: float,
    patience_counter: int,
    patience_limit: int,
) -> dict:
    """Check if training should early-stop. Returns {improved, patience_counter, should_stop}."""
    if val_acc > best_val_acc + min_improvement:
        return {"improved": True, "patience_counter": 0, "should_stop": False}
    else:
        new_counter = patience_counter + 1
        return {
            "improved": False,
            "patience_counter": new_counter,
            "should_stop": new_counter >= patience_limit,
        }
