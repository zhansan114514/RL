"""
Partial Trajectory Reward computation.

Implements:
- compute_reward_delta(): core delta_y / delta_not_y formula
"""

from __future__ import annotations


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
