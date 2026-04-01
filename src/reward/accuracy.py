"""
Compatibility layer: imports have moved to src.algorithms.reward.

This file re-exports all public symbols for backward compatibility.
The canonical implementation is in src/algorithms/reward.py.
"""

from src.algorithms.reward import (
    extract_answer,
    normalize_answer,
    compute_accuracy,
    compute_accuracy_with_ci,
    compute_per_round_accuracy,
    compute_improvement_rate,
)
