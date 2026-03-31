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
from typing import Optional

import torch

from src.trajectory.generator import generate_trajectories
from src.trajectory.preference import build_preference_dataset, to_hf_dataset
from src.training.dpo_trainer import train_dpo

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
        early_stopping_patience: Optional patience for early stopping (if val_dataset provided).
        min_improvement: Minimum improvement threshold to reset patience.

    Returns:
        Dict with final actor_path, critic_path, and validation_metrics (if val_dataset provided).
    """
    import os

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
        actor_model = VLLMInference(current_actor_path, gpu_memory_utilization=0.45,
                                     cuda_device=actor_device)
        critic_model = VLLMInference(current_critic_path, gpu_memory_utilization=0.45,
                                      cuda_device=critic_device)

        critic_pairs = []
        for i, sample in enumerate(dataset):
            logger.info(f"  Generating trajectories: {i+1}/{len(dataset)}")
            try:
                pairs = generate_trajectories(
                    actor_model, critic_model, sample, dataset_name,
                    num_rounds=num_rounds,
                    reward_threshold=reward_threshold,
                    num_simulations=num_simulations,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                critic_pairs.extend(pairs)
            except Exception as e:
                logger.warning(f"  Sample {i} failed: {e}")
                continue

        if critic_pairs:
            critic_prefs = build_preference_dataset(critic_pairs, agent="critic")
            critic_hf = to_hf_dataset(critic_prefs)

            # Clean up GPU memory before training
            logger.info("Cleaning up GPU memory before critic training...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            critic_output = os.path.join(
                output_base_dir, f"critic_iter{iteration}")
            current_critic_path = train_dpo(
                model_name_or_path=current_critic_path,
                preference_dataset=critic_hf,
                output_dir=critic_output,
                model_type=model_type,
                lora_r=lora_r,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                beta=beta,
                seed=seed,
            )
            logger.info(f"Critic saved: {current_critic_path}")
        else:
            logger.warning("No critic pairs generated, skipping critic training.")

        # Clean up models to free GPU memory before next phase
        logger.info("Cleaning up models after critic training phase...")
        actor_model.cleanup()
        critic_model.cleanup()
        del actor_model, critic_model

        # Step 2: Train actor (fix critic)
        logger.info("Step 2: Training Actor (critic fixed)")
        actor_model = VLLMInference(current_actor_path, gpu_memory_utilization=0.8,
                                     cuda_device=actor_device)
        critic_model = VLLMInference(current_critic_path, gpu_memory_utilization=0.8,
                                      cuda_device=critic_device)

        actor_pairs = []
        for i, sample in enumerate(dataset):
            logger.info(f"  Generating trajectories: {i+1}/{len(dataset)}")
            try:
                pairs = generate_trajectories(
                    actor_model, critic_model, sample, dataset_name,
                    num_rounds=num_rounds,
                    reward_threshold=reward_threshold,
                    num_simulations=num_simulations,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                actor_pairs.extend(pairs)
            except Exception as e:
                logger.warning(f"  Sample {i} failed: {e}")
                continue

        if actor_pairs:
            actor_prefs = build_preference_dataset(actor_pairs, agent="actor")
            actor_hf = to_hf_dataset(actor_prefs)

            # Clean up GPU memory before training
            logger.info("Cleaning up GPU memory before actor training...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            actor_output = os.path.join(
                output_base_dir, f"actor_iter{iteration}")
            current_actor_path = train_dpo(
                model_name_or_path=current_actor_path,
                preference_dataset=actor_hf,
                output_dir=actor_output,
                model_type=model_type,
                lora_r=lora_r,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                beta=beta,
                seed=seed,
            )
            logger.info(f"Actor saved: {current_actor_path}")
        else:
            logger.warning("No actor pairs generated, skipping actor training.")

        # Clean up models before validation
        logger.info("Cleaning up models after actor training phase...")
        actor_model.cleanup()
        critic_model.cleanup()
        del actor_model, critic_model

        # Validation set evaluation (if provided)
        if val_dataset is not None:
            logger.info(f"Evaluating on validation set ({len(val_dataset)} samples)...")
            actor_model = VLLMInference(current_actor_path, gpu_memory_utilization=0.8,
                                         cuda_device=actor_device)
            critic_model = VLLMInference(current_critic_path, gpu_memory_utilization=0.8,
                                          cuda_device=critic_device)

            from src.evaluation.benchmarks import evaluate_benchmark
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
