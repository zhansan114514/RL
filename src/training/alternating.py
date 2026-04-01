"""
Compatibility layer: alternating training has been split into:
- scheduler.py: alternating_train() scheduling logic
- trainer.py: train_agent() single-agent training
- model_manager.py: VLLMInference creation and cleanup

This file re-exports all public symbols for backward compatibility.
"""

from src.training.scheduler import alternating_train
from src.training.trainer import train_agent
from src.training.model_manager import create_model_pair, cleanup_models
