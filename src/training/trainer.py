"""
Single-agent training: trajectory generation + DPO.

Extracted from alternating.py to separate concerns:
- This module handles training ONE agent (actor or critic)
- The alternating schedule is in scheduler.py
"""

from __future__ import annotations

import logging
import os
import traceback

from src.algorithms.trajectory import generate_trajectories
from src.trajectory.preference import build_preference_dataset, convert_to_hf_dataset
from src.training.dpo_trainer import train_dpo

logger = logging.getLogger(__name__)


def train_agent(
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
    """
    Run trajectory generation + DPO training for one agent.

    Args:
        agent: "actor" or "critic".
        actor_model: VLLMInference for the actor.
        critic_model: VLLMInference for the critic.
        dataset: Training samples.
        dataset_name: Dataset name for prompt selection.
        current_model_path: Path to the model being trained.
        output_base_dir: Output directory.
        iteration: Current iteration number.
        model_type: Model architecture type.
        num_rounds: Deliberation rounds per sample.
        reward_threshold: Min delta for preference pairs.
        num_simulations: MC roll-out simulations.
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.
        seed: Random seed.
        lora_r: LoRA rank.
        learning_rate: Learning rate.
        batch_size: Batch size.
        num_epochs: Training epochs.
        beta: DPO beta.

    Returns:
        Path to the trained model.
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
    hf_dataset = convert_to_hf_dataset(prefs)

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
