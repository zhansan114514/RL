"""
DPO training runner for isolated subprocess execution.

This module is called by dpo_trainer.py via subprocess to run DPO training
in a fresh CUDA context where CUDA_VISIBLE_DEVICES works correctly.
It reads config from a JSON file and loads the dataset from disk.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys

# Patch ctypes.CDLL to inject missing NVML symbols for older NVIDIA drivers.
# PyTorch 2.10 calls nvmlDeviceGetNvLinkRemoteDeviceType via dlopen("libnvidia-ml.so.1"),
# but this symbol doesn't exist in driver 450.x, causing RuntimeError.
# We intercept the CDLL call and add the missing symbol as a no-op stub.
import ctypes
_orig_cdll_init = ctypes.CDLL.__init__

def _patched_cdll_init(self, name, *args, **kwargs):
    _orig_cdll_init(self, name, *args, **kwargs)
    if name and "libnvidia-ml" in str(name):
        try:
            self.nvmlDeviceGetNvLinkRemoteDeviceType
        except AttributeError:
            # Stub: returns NVML_ERROR_FUNCTION_NOT_FOUND (13)
            _STUB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p)
            setattr(self, "nvmlDeviceGetNvLinkRemoteDeviceType",
                    _STUB(lambda dev, link, dtype: 13))

ctypes.CDLL.__init__ = _patched_cdll_init

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _run():
    if len(sys.argv) < 2:
        print("Usage: python _dpo_runner.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer, DPOConfig

    from src.training.lora_config import get_lora_config

    # Load preference dataset from disk
    dataset_path = cfg["dataset_path"]
    if dataset_path and os.path.exists(dataset_path):
        preference_dataset = load_from_disk(dataset_path)
        logger.info(f"Loaded {len(preference_dataset)} preference samples from {dataset_path}")
    else:
        logger.error(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    # dtype detection (same logic as dpo_trainer.py)
    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported(including_emulation=False)
    )
    if use_bf16:
        torch_dtype = torch.bfloat16
    else:
        model_name_lower = cfg["model_name_or_path"].lower()
        if "gemma" in model_name_lower:
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16

    lora_config = get_lora_config(
        cfg["model_type"], r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"]
    )

    # Load model on the single visible GPU (CUDA_VISIBLE_DEVICES is set by parent)
    logger.info(f"Loading model: {cfg['model_name_or_path']} (dtype={torch_dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DPO training arguments
    # Per ACC-Collab paper Appendix A: "we use a negative log-likelihood (NLL)
    # regularization term (with weight 1)" — achieved by adding "sft" to loss_types.
    loss_type_val = cfg.get("loss_type", "sigmoid")

    training_args = DPOConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        max_length=cfg["max_length"],
        beta=cfg["beta"],
        loss_type=[loss_type_val, "sft"],
        loss_weights=[1.0, 1.0],
        max_grad_norm=cfg["max_grad_norm"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        seed=cfg["seed"],
        logging_steps=10,
        save_strategy="epoch",
        bf16=use_bf16,
        fp16=(not use_bf16 and torch_dtype != torch.float32),
        gradient_checkpointing=cfg["gradient_checkpointing"],
        remove_unused_columns=False,
        report_to="wandb" if cfg.get("use_wandb") else "none",
        run_name=cfg.get("wandb_project") if cfg.get("use_wandb") else None,
        # Precompute reference log-probs to avoid holding two models in GPU memory.
        # Essential for V100 32GB with LoRA r=256 on 7B models.
        precompute_ref_log_probs=True,
    )

    # Initialize DPO trainer
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

    # Save LoRA adapter, then merge for vLLM compatibility
    output_dir = cfg["output_dir"]
    adapter_dir = output_dir + "_adapter"
    trainer.save_model(adapter_dir)
    logger.info(f"LoRA adapter saved to: {adapter_dir}")

    if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        logger.info("Merging LoRA weights into base model...")
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name_or_path"],
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
        logger.warning("No adapter_config.json found, saving model as-is.")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass

    logger.info("DPO runner finished successfully.")


if __name__ == "__main__":
    _run()
