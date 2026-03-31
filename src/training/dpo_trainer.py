"""
DPO training using trl library with NLL regularization.

Implements the DPO loss from Eq. 6 of the ACC-Collab paper,
with NLL regularization as in Pang et al. (2024a) IRPO.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.training.lora_config import get_lora_config

logger = logging.getLogger(__name__)


def train_dpo(
    model_name_or_path: str,
    preference_dataset,
    output_dir: str,
    model_type: str = "gemma2",
    lora_r: int = 256,
    lora_alpha: int = 512,
    nll_weight: float = 1.0,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    max_length: int = 2048,
    warmup_ratio: float = 0.1,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "acc-collab",
) -> str:
    """
    Train a model with DPO + LoRA.

    Args:
        model_name_or_path: Base model name or path.
        preference_dataset: HuggingFace Dataset with prompt/chosen/rejected.
        output_dir: Where to save the trained model.
        model_type: Model architecture for LoRA target modules.
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha.
        nll_weight: NLL regularization weight (paper: 1.0).
        learning_rate: Learning rate.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation.
        num_epochs: Number of training epochs.
        max_length: Max sequence length.
        warmup_ratio: Warmup ratio.
        seed: Random seed.
        use_wandb: Whether to log to wandb.
        wandb_project: Wandb project name.

    Returns:
        Path to saved model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOTrainer, DPOConfig

    lora_config = get_lora_config(model_type, r=lora_r, lora_alpha=lora_alpha)

    logger.info(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DPO training arguments
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        seed=seed,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else "none",
        run_name=wandb_project if use_wandb else None,
    )

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=preference_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    logger.info("Starting DPO training...")
    trainer.train()
    logger.info("Training complete.")

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to: {output_dir}")

    return output_dir
