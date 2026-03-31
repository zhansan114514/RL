"""
Alternating training scheduler for Actor-Critic.

Implements the iterative best-response optimization:
1. Fix actor -> train critic
2. Fix critic -> train actor
3. Repeat (ACC-Collab+ does 2 rounds)
"""

from __future__ import annotations

import logging
from typing import Optional

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
    max_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 42,
) -> dict[str, str]:
    """
    Run alternating Actor-Critic training.

    Args:
        actor_path: Path to actor base model.
        critic_path: Path to critic base model.
        dataset: List of standardized samples.
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
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.
        seed: Random seed.

    Returns:
        Dict with final actor_path and critic_path.
    """
    import os

    from src.inference.vllm_server import VLLMInference

    current_actor_path = actor_path
    current_critic_path = critic_path

    for iteration in range(num_iterations):
        logger.info(f"=== Iteration {iteration + 1}/{num_iterations} ===")

        # Step 1: Train critic (fix actor)
        logger.info("Step 1: Training Critic (actor fixed)")
        actor_model = VLLMInference(current_actor_path)
        critic_model = VLLMInference(current_critic_path)

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

            critic_output = os.path.join(
                output_base_dir, f"critic_iter{iteration}")
            current_critic_path = train_dpo(
                model_name_or_path=critic_path,
                preference_dataset=critic_hf,
                output_dir=critic_output,
                model_type=model_type,
                lora_r=lora_r,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                seed=seed,
            )
            logger.info(f"Critic saved: {current_critic_path}")
        else:
            logger.warning("No critic pairs generated, skipping critic training.")

        # Step 2: Train actor (fix critic)
        logger.info("Step 2: Training Actor (critic fixed)")
        actor_model = VLLMInference(current_actor_path)
        critic_model = VLLMInference(current_critic_path)

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

            actor_output = os.path.join(
                output_base_dir, f"actor_iter{iteration}")
            current_actor_path = train_dpo(
                model_name_or_path=actor_path,
                preference_dataset=actor_hf,
                output_dir=actor_output,
                model_type=model_type,
                lora_r=lora_r,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                seed=seed,
            )
            logger.info(f"Actor saved: {current_actor_path}")
        else:
            logger.warning("No actor pairs generated, skipping actor training.")

    return {
        "actor_path": current_actor_path,
        "critic_path": current_critic_path,
    }
