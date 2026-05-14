"""Supervised fine-tuning with response-only loss and LoRA adapters."""

from __future__ import annotations

import logging

from src.utils.runtime_env import configure_runtime_libraries

configure_runtime_libraries()

import torch  # noqa: E402

logger = logging.getLogger(__name__)


def train_sft(
    model_name_or_path: str,
    sft_dataset,
    output_dir: str,
    model_type: str = "qwen3",
    lora_r: int = 256,
    lora_alpha: int = 512,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 1,
    max_length: int = 2048,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    optim: str = "adamw_torch",
    weight_decay: float = 0.01,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "diverse-actor-critic-society",
    gradient_checkpointing: bool = True,
    merge_lora: bool = False,
    device: int = 0,
    timeout_per_1k: int = 1800,
) -> str:
    """Train a LoRA adapter with causal LM loss on response tokens only."""

    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(including_emulation=False)
    )
    if use_bf16:
        torch_dtype = torch.bfloat16
    else:
        model_name_lower = model_name_or_path.lower()
        if "gemma" in model_name_lower:
            torch_dtype = torch.float32
            logger.warning(
                "Gemma models don't support fp16 reliably; falling back to float32."
            )
        else:
            torch_dtype = torch.float16
    logger.info("Using dtype: %s (bf16 supported: %s)", torch_dtype, use_bf16)

    import math
    import os
    import subprocess
    import sys
    import tempfile

    _prev_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    _all_gpus = _prev_cuda_vis.split(",") if _prev_cuda_vis else [
        str(i) for i in range(torch.cuda.device_count())
    ]
    if device < len(_all_gpus):
        target_physical = _all_gpus[device].strip()
    else:
        target_physical = str(device)

    _temp_dir = tempfile.mkdtemp(prefix="sft_data_")
    dataset_path = os.path.join(_temp_dir, "sft_dataset")
    sft_dataset.save_to_disk(dataset_path)
    logger.info("Saved SFT dataset to %s for subprocess training", dataset_path)

    _config_path = os.path.join(_temp_dir, "config.json")
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
        "max_grad_norm": max_grad_norm,
        "optim": optim,
        "weight_decay": weight_decay,
        "seed": seed,
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "gradient_checkpointing": gradient_checkpointing,
        "merge_lora": merge_lora,
    }
    with open(_config_path, "w") as f:
        import json
        json.dump(_config, f)

    _runner_script = os.path.join(os.path.dirname(__file__), "_sft_runner.py")
    env = os.environ.copy()
    configure_runtime_libraries(env, preload=False)
    env["CUDA_VISIBLE_DEVICES"] = target_physical
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = project_root + ((":" + existing_pp) if existing_pp else "")
    logger.info("SFT training on physical GPU %s (isolated subprocess)", target_physical)

    n_examples = len(sft_dataset) if sft_dataset is not None else 1
    _timeout = max(1800, timeout_per_1k * math.ceil(n_examples / 1000))
    logger.info(
        "SFT timeout: %ss (%s examples, %ss/1k examples)",
        _timeout,
        n_examples,
        timeout_per_1k,
    )

    try:
        result = subprocess.run(
            [sys.executable, _runner_script, _config_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=_timeout,
        )

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    logger.info("[SFT worker] %s", line)
        if result.returncode != 0:
            logger.error("SFT subprocess failed (exit %s):", result.returncode)
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-20:]:
                    logger.error("  %s", line)
            raise RuntimeError(f"SFT training failed with exit code {result.returncode}")
    finally:
        import shutil
        if os.path.exists(_temp_dir):
            shutil.rmtree(_temp_dir, ignore_errors=True)

    return output_dir if merge_lora else output_dir + "_adapter"
