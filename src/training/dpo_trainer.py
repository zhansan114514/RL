"""
DPO training using trl library.

Implements the DPO loss from Eq. 6 of the ACC-Collab paper.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

import torch

from src.training.lora_config import get_lora_config

logger = logging.getLogger(__name__)


def train_dpo(
    model_name_or_path: str,
    preference_dataset,
    output_dir: str,
    model_type: str = "gemma2",
    lora_r: int = 256,
    lora_alpha: int = 512,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    max_length: int = 2048,
    warmup_ratio: float = 0.1,
    beta: float = 0.1,
    loss_type: str = "sigmoid",
    max_grad_norm: float = 1.0,
    optim: str = "adamw_torch",
    weight_decay: float = 0.01,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "acc-collab",
    gradient_checkpointing: bool = True,
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
        learning_rate: Learning rate.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Gradient accumulation.
        num_epochs: Number of training epochs.
        max_length: Max sequence length.
        warmup_ratio: Warmup ratio.
        beta: DPO beta parameter controlling deviation from reference policy.
        loss_type: DPO loss type ("sigmoid", "hinge", "ipo", etc.).
        max_grad_norm: Max gradient norm for clipping (important for FP16).
        optim: Optimizer type.
        weight_decay: Weight decay for optimizer.
        seed: Random seed.
        use_wandb: Whether to log to wandb.
        wandb_project: Wandb project name.

    Returns:
        Path to saved model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import DPOTrainer, DPOConfig

    lora_config = get_lora_config(model_type, r=lora_r, lora_alpha=lora_alpha)

    # Detect hardware capabilities for dtype selection
    # V100 (cc 7.0) doesn't support bf16. Gemma2 doesn't support fp16
    # (numerical instability). So on V100 we must use float32.
    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(including_emulation=False)
    )
    if use_bf16:
        torch_dtype = torch.bfloat16
    else:
        # Check if fp16 is safe for the model (Gemma2 doesn't support fp16)
        model_name_lower = model_name_or_path.lower()
        if "gemma" in model_name_lower:
            torch_dtype = torch.float32
            logger.warning(
                "Gemma models don't support fp16 (numerical instability). "
                "Falling back to float32 on this hardware."
            )
        else:
            torch_dtype = torch.float16
    logger.info(f"Using dtype: {torch_dtype} (bf16 supported: {use_bf16})")

    logger.info(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
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
        beta=beta,
        loss_type=loss_type,
        max_grad_norm=max_grad_norm,
        optim=optim,
        weight_decay=weight_decay,
        seed=seed,
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=(not use_bf16 and torch_dtype != torch.float32),
        gradient_checkpointing=gradient_checkpointing,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else "none",
        run_name=wandb_project if use_wandb else None,
    )

    # Initialize DPO trainer with trl version compatibility
    # trl <0.12 uses tokenizer=, >=0.12 uses processing_class=
    import trl
    trl_version = tuple(int(x) for x in trl.__version__.split('.')[:2])
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=preference_dataset,
        peft_config=lora_config,
    )
    if trl_version >= (0, 12):
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DPOTrainer(**trainer_kwargs)

    logger.info("Starting DPO training...")
    trainer.train()
    logger.info("Training complete.")

    # Save LoRA adapter first, then merge into base model for vLLM compatibility
    adapter_dir = output_dir + "_adapter"
    trainer.save_model(adapter_dir)
    logger.info(f"LoRA adapter saved to: {adapter_dir}")

    # Merge LoRA weights into base model
    import os

    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info("Merging LoRA weights into base model...")
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
        merged_model = merged_model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Merged model saved to: {output_dir}")

        del base_model, merged_model
    else:
        # Fallback: no adapter found (e.g. in mock/test environments)
        logger.warning("No adapter_config.json found, saving model as-is.")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass

    return output_dir
