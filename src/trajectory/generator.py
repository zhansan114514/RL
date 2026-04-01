"""
Compatibility layer: imports have moved to src.algorithms.trajectory.

This file re-exports all public symbols for backward compatibility.
The canonical implementation is in src/algorithms/trajectory.py.
"""

from src.algorithms.trajectory import generate_trajectories, _make_guided_prompt
