"""
Model utility functions for detecting model types and architectures.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_model_type(model_name: str) -> str:
    """
    Detect model architecture type from model name.

    Uses simple string matching on the model name/path to determine
    the architecture type. This is needed for selecting appropriate
    LoRA target modules and other model-specific configurations.

    Args:
        model_name: Model name or path (e.g., "google/gemma-2-2b-it").

    Returns:
        Model architecture type: "llama3", "mistral", "gemma2", "qwen2.5", or "qwen3".
        Defaults to "llama3" if no match is found.

    Examples:
        >>> detect_model_type("meta-llama/Llama-3-8b")
        'llama3'
        >>> detect_model_type("mistralai/Mistral-7B")
        'mistral'
        >>> detect_model_type("google/gemma-2-2b-it")
        'gemma2'
        >>> detect_model_type("Qwen/Qwen2.5-7B-Instruct")
        'qwen2.5'
        >>> detect_model_type("Qwen/Qwen3-8B")
        'qwen3'
    """
    name = model_name.lower()
    if "llama" in name:
        return "llama3"
    elif "mistral" in name:
        return "mistral"
    elif "gemma" in name:
        return "gemma2"
    elif "qwen" in name:
        # Qwen2.5 must be checked before generic Qwen3
        # Match "qwen2.5" or "qwen-2.5" specifically, avoid false positives like "Qwen3-25B"
        if "qwen2.5" in name.replace(" ", "").replace("-", ""):
            return "qwen2.5"
        return "qwen3"
    else:
        # Default fallback
        logger.warning(
            f"Could not detect model type from '{model_name}', defaulting to 'llama3'"
        )
        return "llama3"
