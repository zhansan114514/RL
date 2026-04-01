"""
Alternating training scheduler for Actor-Critic.

Implements the iterative best-response optimization:
1. Fix actor -> train critic
2. Fix critic -> train actor
3. Repeat (ACC-Collab+ does 2 rounds)
"""

from __future__ import annotations

import gc
import logging
import os
import traceback
from typing import Optional

import torch

from src.trajectory.generator import generate_trajectories
from src.trajectory.preference import build_preference_dataset, to_hf_dataset
from src.training.dpo_trainer import train_dpo

logger = logging.getLogger(__name__)


def _cleanup_gpu() -> None:
    """Release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_pairs_and_train(
    agent: str,
    actor_model,
    critic_model,
    dataset: list[dict],
    dataset_name: str,
    current_model_path: str,
    output_base_dir: str,
    iteration: int,
    model_type: str,
    *,
    num_rounds: int = 5,
    reward_threshold: float = 0.0,
    num_simulations: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 42,
    lora_r: int = 256,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    num_epochs: int = 1,
    beta: float = 0.1,
) -> str:
    """Run trajectory generation + DPO training for one agent.

    Returns the output path of the trained model.
    """
    pairs = []
    for i, sample in enumerate(dataset):
        logger.info(f"  Generating trajectories: {i+1}/{len(dataset)}")
        try:
            batch = generate_trajectories(
                actor_model, critic_model, sample, dataset_name,
                num_rounds=num_rounds,
                reward_threshold=reward_threshold,
                num_simulations=num_simulations,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
            )
            pairs.extend(batch)
        except Exception as e:
            logger.warning(f"  Sample {i} failed: {e}")
            logger.debug(f"  Traceback:\n{traceback.format_exc()}")
            continue

    if not pairs:
        logger.error(f"No {agent} pairs generated from dataset!")
        logger.error("This may indicate a problem with:")
        logger.error("  - Data loading (check if dataset is not empty)")
        logger.error("  - Model inference (check if models are working correctly)")
        logger.error("  - Reward estimation (check if MC roll-out is functioning)")
        raise ValueError(
            f"No {agent} preference pairs generated in iteration {iteration}. "
            "Cannot continue training without data."
        )

    prefs = build_preference_dataset(pairs, agent=agent)
    hf_dataset = to_hf_dataset(prefs)

    logger.info(f"Cleaning up GPU memory before {agent} training...")
    _cleanup_gpu()

    output_dir = os.path.join(output_base_dir, f"{agent}_iter{iteration}")
    result_path = train_dpo(
        model_name_or_path=current_model_path,
        preference_dataset=hf_dataset,
        output_dir=output_dir,
        model_type=model_type,
        lora_r=lora_r,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        beta=beta,
        seed=seed,
    )
    logger.info(f"{agent.capitalize()} saved: {result_path}")
    return result_path


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
    from src.inference.vllm_server import VLLMInference

    current_actor_path = actor_path
    current_critic_path = critic_path

    # Track validation metrics for early stopping
    best_val_acc = 0.0
    patience_counter = 0
    val_metrics_history = []

    for iteration in range(num_iterations):
        logger.info(f"=== Iteration {iteration + 1}/{num_iterations} ===")

        # Step 1: Train critic (fix actor)
        logger.info("Step 1: Training Critic (actor fixed)")
        actor_model = VLLMInference(
            current_actor_path,
            gpu_memory_utilization=gpu_memory_utilization,
            cuda_device=actor_device,
            dtype=dtype,
            max_model_len=4096,  # Accommodate deliberation context
        )
        critic_model = VLLMInference(
            current_critic_path,
            gpu_memory_utilization=gpu_memory_utilization,
            cuda_device=critic_device,
            dtype=dtype,
            max_model_len=4096,  # Accommodate deliberation context
        )

        current_critic_path = _generate_pairs_and_train(
            "critic",
            actor_model, critic_model, dataset, dataset_name,
            current_critic_path, output_base_dir, iteration,
            model_type,
            num_rounds=num_rounds,
            reward_threshold=reward_threshold,
            num_simulations=num_simulations,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            lora_r=lora_r,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            beta=beta,
        )

        # Clean up models to free GPU memory before next phase
        logger.info("Cleaning up models after critic training phase...")
        actor_model.cleanup()
        critic_model.cleanup()
        del actor_model, critic_model

        # Step 2: Train actor (fix critic)
        logger.info("Step 2: Training Actor (critic fixed)")
        actor_model = VLLMInference(
            current_actor_path,
            gpu_memory_utilization=gpu_memory_utilization,
            cuda_device=actor_device,
            dtype=dtype,
            max_model_len=4096,  # Accommodate deliberation context
        )
        critic_model = VLLMInference(
            current_critic_path,
            gpu_memory_utilization=gpu_memory_utilization,
            cuda_device=critic_device,
            dtype=dtype,
            max_model_len=4096,  # Accommodate deliberation context
        )

        current_actor_path = _generate_pairs_and_train(
            "actor",
            actor_model, critic_model, dataset, dataset_name,
            current_actor_path, output_base_dir, iteration,
            model_type,
            num_rounds=num_rounds,
            reward_threshold=reward_threshold,
            num_simulations=num_simulations,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            lora_r=lora_r,
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs,
            beta=beta,
        )

        # Clean up models before validation
        logger.info("Cleaning up models after actor training phase...")
        actor_model.cleanup()
        critic_model.cleanup()
        del actor_model, critic_model

        # Validation set evaluation (if provided)
        if val_dataset is not None:
            logger.info(f"Evaluating on validation set ({len(val_dataset)} samples)...")
            from src.evaluation.benchmarks import evaluate_benchmark

            actor_model = VLLMInference(
                current_actor_path,
                gpu_memory_utilization=gpu_memory_utilization,
                cuda_device=actor_device,
                dtype=dtype,
                max_model_len=4096,  # Accommodate deliberation context
            )
            critic_model = VLLMInference(
                current_critic_path,
                gpu_memory_utilization=gpu_memory_utilization,
                cuda_device=critic_device,
                dtype=dtype,
                max_model_len=4096,  # Accommodate deliberation context
            )

            val_results = evaluate_benchmark(
                actor_model, critic_model, val_dataset, dataset_name,
                num_rounds=num_rounds, max_tokens=max_tokens,
                temperature=temperature,
            )

            val_acc = val_results["final_accuracy"]
            val_metrics_history.append({
                "iteration": iteration,
                "val_accuracy": val_acc,
                "val_ci_margin": val_results["ci_margin"],
                "improvement_rate": val_results["improvement_rate"],
            })

            logger.info(
                f"  Iteration {iteration} Val Accuracy: {val_acc:.3f} +/- {val_results['ci_margin']:.3f}"
            )

            # Early stopping check
            if early_stopping_patience is not None:
                if val_acc > best_val_acc + min_improvement:
                    best_val_acc = val_acc
                    patience_counter = 0
                    logger.info(f"  New best validation accuracy: {best_val_acc:.3f}")
                else:
                    patience_counter += 1
                    logger.info(
                        f"  No improvement for {patience_counter}/{early_stopping_patience} iterations"
                    )
                    if patience_counter >= early_stopping_patience:
                        logger.info("Early stopping triggered!")
                        break

            # Clean up validation models
            logger.info("Cleaning up validation models...")
            actor_model.cleanup()
            critic_model.cleanup()
            del actor_model, critic_model

    # Prepare return value
    result = {
        "actor_path": current_actor_path,
        "critic_path": current_critic_path,
    }

    if val_metrics_history:
        result["validation_metrics"] = val_metrics_history
        result["best_val_accuracy"] = best_val_acc

    return result
