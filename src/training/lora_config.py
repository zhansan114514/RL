"""
LoRA configuration for parameter-efficient fine-tuning.

Paper specifies LoRA rank=256 with NLL regularization.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Default target modules for common architectures
LLAMA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MISTRAL_TARGET_MODULES = LLAMA_TARGET_MODULES

GEMMA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MODEL_TARGET_MODULES = {
    "llama3": LLAMA_TARGET_MODULES,
    "mistral": MISTRAL_TARGET_MODULES,
    "gemma2": GEMMA_TARGET_MODULES,
}


def get_lora_config(
    model_type: str = "llama3",
    r: int = 256,
    lora_alpha: int = 512,
    lora_dropout: float = 0.0,
):
    """
    Create a LoRA configuration.

    Args:
        model_type: Model architecture type.
        r: LoRA rank (paper: 256).
        lora_alpha: LoRA alpha (default: 2*r).
        lora_dropout: Dropout rate.

    Returns:
        peft.LoraConfig instance.
    """
    from peft import LoraConfig, TaskType

    target_modules = MODEL_TARGET_MODULES.get(model_type, LLAMA_TARGET_MODULES)

    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )

    logger.info(
        f"LoRA config: r={r}, alpha={lora_alpha}, "
        f"targets={target_modules}"
    )
    return config
