"""
Model lifecycle management for training.

Centralizes VLLMInference creation and cleanup to eliminate
repeated model instantiation code (was 6 copies in alternating_train).
"""

from __future__ import annotations

import gc
import logging

import torch

from src.utils.config import ConfigManager

logger = logging.getLogger(__name__)


def _get_config(key: str, default):
    """Get config value with graceful fallback when ConfigManager is not initialized."""
    if ConfigManager.is_initialized():
        return ConfigManager.instance().get(key, default)
    return default


def create_inference_model(
    model_path: str,
    cuda_device: int = 0,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    dtype: str | None = None,
    enable_lora: bool = False,
    max_loras: int = 1,
    max_lora_rank: int = 256,
) -> "VLLMInference":
    """
    Create a VLLMInference instance with config-driven defaults.

    All parameters can be overridden explicitly; if not provided,
    they are read from the ConfigManager.

    Args:
        model_path: Path or name of the model.
        cuda_device: CUDA device index.
        gpu_memory_utilization: Override for GPU memory fraction.
        max_model_len: Override for max model context length.
        dtype: Override for model dtype.
        enable_lora: Enable dynamic LoRA adapter loading.
        max_loras: Max concurrent LoRA adapters (only when enable_lora=True).
        max_lora_rank: Max LoRA rank (only when enable_lora=True).

    Returns:
        VLLMInference instance.
    """
    from src.inference.vllm_server import VLLMInference

    gpu_mem = gpu_memory_utilization if gpu_memory_utilization is not None else _get_config("inference.gpu_memory_utilization", 0.45)
    max_len = max_model_len if max_model_len is not None else _get_config("inference.max_model_len", 4096)
    model_dtype = dtype if dtype is not None else _get_config("inference.dtype", "auto")

    return VLLMInference(
        model_path,
        gpu_memory_utilization=gpu_mem,
        cuda_device=cuda_device,
        dtype=model_dtype,
        max_model_len=max_len,
        enable_lora=enable_lora,
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
    )


def create_model_pair(
    actor_path: str,
    critic_path: str,
    actor_device: int = 0,
    critic_device: int = 0,
    gpu_memory_utilization: float | None = None,
    max_model_len: int | None = None,
    dtype: str | None = None,
    enable_lora: bool = False,
    max_loras: int = 1,
    max_lora_rank: int = 256,
) -> tuple["VLLMInference", "VLLMInference"]:
    """Create actor + critic inference models as a pair."""
    actor = create_inference_model(
        actor_path, cuda_device=actor_device,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len, dtype=dtype,
        enable_lora=enable_lora, max_loras=max_loras,
        max_lora_rank=max_lora_rank,
    )
    critic = create_inference_model(
        critic_path, cuda_device=critic_device,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len, dtype=dtype,
        enable_lora=enable_lora, max_loras=max_loras,
        max_lora_rank=max_lora_rank,
    )
    return actor, critic


def cleanup_models(*models: "VLLMInference") -> None:
    """Clean up models and release GPU memory."""
    for model in models:
        if hasattr(model, "cleanup"):
            try:
                model.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up model: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("GPU memory cleaned up")
