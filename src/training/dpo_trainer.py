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
    device: int = 0,
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

    # Resolve target physical GPU id
    import os
    _prev_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    _all_gpus = _prev_cuda_vis.split(",") if _prev_cuda_vis else [
        str(i) for i in range(torch.cuda.device_count())
    ]
    if device < len(_all_gpus):
        target_physical = _all_gpus[device].strip()
    else:
        target_physical = str(device)

    # Save preference dataset to disk and run DPO training in a subprocess.
    # This is necessary because CUDA context may already be initialized by
    # vLLM, making CUDA_VISIBLE_DEVICES ineffective in the current process.
    # A subprocess gets a fresh CUDA context where CUDA_VISIBLE_DEVICES works.
    _temp_dir = None
    if preference_dataset is not None:
        import tempfile
        from datasets import load_from_disk
        _temp_dir = tempfile.mkdtemp(prefix="dpo_data_")
        dataset_path = os.path.join(_temp_dir, "preference_dataset")
        preference_dataset.save_to_disk(dataset_path)
        logger.info(f"Saved preference dataset to {dataset_path} for subprocess training")
    else:
        dataset_path = None

    # Build subprocess config
    _config_path = os.path.join(
        _temp_dir or tempfile.mkdtemp(prefix="dpo_cfg_"), "config.json"
    )
    _config = {
        "model_name_or_path": model_name_or_path,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "model_type": model_type,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_epochs": num_epochs,
        "max_length": max_length,
        "warmup_ratio": warmup_ratio,
        "beta": beta,
        "loss_type": loss_type,
        "max_grad_norm": max_grad_norm,
        "optim": optim,
        "weight_decay": weight_decay,
        "seed": seed,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "gradient_checkpointing": gradient_checkpointing,
    }
    with open(_config_path, "w") as f:
        import json
        json.dump(_config, f)

    # Run training in subprocess with isolated CUDA context
    import subprocess
    import sys

    _runner_script = os.path.join(os.path.dirname(__file__), "_dpo_runner.py")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = target_physical
    # Ensure src package is importable in the subprocess
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = project_root + ((":" + existing_pp) if existing_pp else "")
    logger.info(f"DPO training on physical GPU {target_physical} (isolated subprocess)")

    result = subprocess.run(
        [sys.executable, _runner_script, _config_path],
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                logger.info(f"[DPO worker] {line}")
    if result.returncode != 0:
        logger.error(f"DPO subprocess failed (exit {result.returncode}):")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-20:]:
                logger.error(f"  {line}")
        raise RuntimeError(f"DPO training failed with exit code {result.returncode}")

    # Cleanup temp files
    import shutil
    if _temp_dir and os.path.exists(_temp_dir):
        shutil.rmtree(_temp_dir, ignore_errors=True)

    return output_dir
