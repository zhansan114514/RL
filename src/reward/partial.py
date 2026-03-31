"""
Partial Trajectory Reward computation.

Implements:
- compute_reward_delta(): core delta_y / delta_not_y formula
- compute_trajectory_rewards(): per-round reward estimation for a full trajectory
- select_preference_pairs(): filter pairs by delta threshold
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_reward_delta(
    reward_guided: float,
    reward_natural: float,
) -> float:
    """
    Compute delta = reward_guided - reward_natural.

    For delta_y: guided towards correct answer vs natural.
    For delta_not_y: natural vs guided away from correct answer.
    """
    return reward_guided - reward_natural


def compute_trajectory_rewards(
    rewards_natural: list[float],
    rewards_guided_correct: list[float],
    rewards_guided_wrong: list[float],
) -> list[dict]:
    """
    Compute per-round reward deltas for a deliberation trajectory.

    Args:
        rewards_natural: Reward at each round from natural deliberation.
        rewards_guided_correct: Reward at each round from guided-correct (z_y).
        rewards_guided_wrong: Reward at each round from guided-wrong (z_not_y).

    Returns:
        List of dicts per round with keys:
            round, v_natural, v_y, v_not_y, delta_y, delta_not_y
    """
    n = min(len(rewards_natural), len(rewards_guided_correct), len(rewards_guided_wrong))
    results = []

    for t in range(n):
        v_natural = rewards_natural[t]
        v_y = rewards_guided_correct[t]
        v_not_y = rewards_guided_wrong[t]

        results.append({
            "round": t,
            "v_natural": v_natural,
            "v_y": v_y,
            "v_not_y": v_not_y,
            "delta_y": compute_reward_delta(v_y, v_natural),
            "delta_not_y": compute_reward_delta(v_natural, v_not_y),
        })

    return results


def select_preference_pairs(
    trajectory_rewards: list[dict],
    epsilon: float = 0.0,
) -> list[dict]:
    """
    Select preference pairs based on reward deltas (Algorithm 1, line 10-13).

    Args:
        trajectory_rewards: Output from compute_trajectory_rewards.
        epsilon: Minimum delta threshold for filtering.

    Returns:
        List of valid preference pair descriptors:
            - delta_y >= epsilon: positive=guided_correct, negative=natural
            - delta_not_y >= epsilon: positive=natural, negative=guided_wrong
    """
    pairs = []

    for tr in trajectory_rewards:
        t = tr["round"]

        if tr["delta_y"] >= epsilon:
            pairs.append({
                "round": t,
                "direction": "towards",
                "positive": "guided_correct",
                "negative": "natural",
                "delta": tr["delta_y"],
            })

        if tr["delta_not_y"] >= epsilon:
            pairs.append({
                "round": t,
                "direction": "away",
                "positive": "natural",
                "negative": "guided_wrong",
                "delta": tr["delta_not_y"],
            })

    return pairs


def validate_reward_range(reward: float) -> bool:
    """Check that a reward value is in valid [0, 1] range."""
    return 0.0 <= reward <= 1.0


def clamp_reward(reward: float) -> float:
    """Clamp reward to [0, 1] range."""
    return max(0.0, min(1.0, reward))
