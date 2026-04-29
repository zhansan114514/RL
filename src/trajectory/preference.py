"""
Preference data builder: filters and formats trajectory pairs for DPO training.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


def build_preference_dataset(
    all_pairs: list[dict],
    min_delta: float = 0.0,
    agent: str = "actor",
) -> list[dict]:
    """
    Build a preference dataset from generated trajectory pairs.

    Args:
        all_pairs: Raw preference pairs from generate_trajectories.
        min_delta: Minimum delta threshold for quality filtering.
        agent: "actor" or "critic" to select which agent's pairs to use.

    Returns:
        List of preference samples with keys:
            prompt, chosen, rejected, round, delta
    """
    dataset = []
    for pair in all_pairs:
        if pair.get("delta", 0) < min_delta:
            continue

        if agent == "actor":
            dataset.append({
                "prompt": pair.get("actor_prompt", ""),
                "chosen": pair["positive"],
                "rejected": pair["negative"],
                "round": pair["round"],
                "delta": pair["delta"],
                "direction": pair.get("direction", ""),
            })
        elif agent == "critic":
            dataset.append({
                "prompt": pair.get("critic_prompt", ""),
                "chosen": pair["positive_critic"],
                "rejected": pair["negative_critic"],
                "round": pair["round"],
                "delta": pair["delta"],
                "direction": pair.get("direction", ""),
            })

    logger.info(f"Built {len(dataset)} preference pairs for {agent}")
    return dataset


def convert_to_hf_dataset(
    preference_data: list[dict],
) -> "Dataset":
    """Convert preference data list to HuggingFace Dataset."""
    from datasets import Dataset
    return Dataset.from_list(preference_data)
