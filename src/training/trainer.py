"""
Single-agent training: trajectory generation + DPO.

Extracted from alternating.py to separate concerns:
- This module handles training ONE agent (actor or critic)
- The alternating schedule is in scheduler.py
"""

from __future__ import annotations

import json
import logging
import os
import traceback

from src.algorithms.trajectory import generate_trajectories
from src.trajectory.preference import build_preference_dataset, convert_to_hf_dataset
from src.training.dpo_trainer import train_dpo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory persistence (JSONL)
# ---------------------------------------------------------------------------

def _trajectory_path(cache_dir: str, agent: str, iteration: int) -> str:
    """Return the JSONL path for a given agent and iteration."""
    d = os.path.join(cache_dir, "trajectories")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{agent}_iter{iteration}.jsonl")


def save_trajectory_batch(
    pairs: list[dict],
    cache_dir: str,
    agent: str,
    iteration: int,
) -> str:
    """Append trajectory pairs to a JSONL file. Returns the file path."""
    path = _trajectory_path(cache_dir, agent, iteration)
    with open(path, "a", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(pairs)} trajectory pairs to {path}")
    return path


def load_trajectory_data(
    cache_dir: str,
    agent: str,
    iteration: int,
) -> list[dict] | None:
    """Load trajectory pairs from JSONL. Returns None if file missing."""
    path = _trajectory_path(cache_dir, agent, iteration)
    if not os.path.exists(path):
        return None
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    logger.info(f"Loaded {len(pairs)} trajectory pairs from {path}")
    return pairs


def generate_trajectory_data(
    actor_model,
    critic_model,
    dataset: list[dict],
    dataset_name: str,
    *,
    num_rounds: int = 5,
    reward_threshold: float = 0.0,
    num_simulations: int = 5,
    max_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 42,
    cache_dir: str | None = None,
    agent: str = "critic",
    iteration: int = 0,
) -> list[dict]:
    """Generate deliberation trajectory pairs for all samples.

    This function is separated from DPO training so that vLLM models
    can be cleaned up before loading training models, avoiding GPU OOM.

    Args:
        actor_model: VLLMInference for the actor.
        critic_model: VLLMInference for the critic.
        dataset: Training samples.
        dataset_name: Dataset name for prompt selection.
        num_rounds: Deliberation rounds per sample.
        reward_threshold: Min delta for preference pairs.
        num_simulations: MC roll-out simulations.
        max_tokens: Max generation tokens.
        temperature: Sampling temperature.
        seed: Random seed.
        cache_dir: If set, persist trajectory pairs to disk as JSONL.
        agent: Which agent is being trained ("actor" or "critic").
        iteration: Current alternating-training iteration.

    Returns:
        List of trajectory pairs (preference data).
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

            # Persist incrementally so we don't lose work on crash
            if cache_dir and batch:
                save_trajectory_batch(batch, cache_dir, agent, iteration)

        except Exception as e:
            logger.warning(f"  Sample {i} failed: {e}")
            logger.debug(f"  Traceback:\n{traceback.format_exc()}")
            continue
    return pairs


def train_dpo_from_pairs(
    agent: str,
    pairs: list[dict],
    current_model_path: str,
    output_base_dir: str,
    iteration: int,
    model_type: str,
    *,
    lora_r: int = 256,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    num_epochs: int = 1,
    beta: float = 0.1,
    seed: int = 42,
    device: int = 0,
) -> str:
    """Build preference dataset from trajectory pairs and run DPO training.

    Args:
        agent: "actor" or "critic".
        pairs: Trajectory pairs from generate_trajectory_data.
        current_model_path: Path to the model being trained.
        output_base_dir: Output directory.
        iteration: Current iteration number.
        model_type: Model architecture type.
        lora_r: LoRA rank.
        learning_rate: Learning rate.
        batch_size: Batch size.
        num_epochs: Training epochs.
        beta: DPO beta.
        seed: Random seed.
        device: CUDA device for DPO training.

    Returns:
        Path to the trained model.
    """
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
        device=device,
    )
    logger.info(f"{agent.capitalize()} saved: {result_path}")
    return result_path


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
    device: int = 0,
    cache_dir: str | None = None,
) -> str:
    """
    Run trajectory generation + DPO training for one agent.

    This is a convenience wrapper that combines generate_trajectory_data
    and train_dpo_from_pairs. Prefer calling them separately in the
    scheduler to allow GPU cleanup between phases.

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
        device: CUDA device for DPO training.
        cache_dir: If set, persist and reuse trajectory data.

    Returns:
        Path to the trained model.
    """
    pairs = generate_trajectory_data(
        actor_model, critic_model, dataset, dataset_name,
        num_rounds=num_rounds,
        reward_threshold=reward_threshold,
        num_simulations=num_simulations,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        cache_dir=cache_dir,
        agent=agent,
        iteration=iteration,
    )
    return train_dpo_from_pairs(
        agent, pairs, current_model_path, output_base_dir, iteration, model_type,
        lora_r=lora_r, learning_rate=learning_rate,
        batch_size=batch_size, num_epochs=num_epochs, beta=beta,
        seed=seed, device=device,
    )
