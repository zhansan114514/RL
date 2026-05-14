"""SFT training runner for isolated subprocess execution."""

from __future__ import annotations

import gc
import json
import logging
import os
import sys

import ctypes

_orig_cdll_init = ctypes.CDLL.__init__


def _patched_cdll_init(self, name, *args, **kwargs):
    _orig_cdll_init(self, name, *args, **kwargs)
    if name and "libnvidia-ml" in str(name):
        try:
            self.nvmlDeviceGetNvLinkRemoteDeviceType
        except AttributeError:
            _STUB = ctypes.CFUNCTYPE(
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_void_p,
            )
            setattr(
                self,
                "nvmlDeviceGetNvLinkRemoteDeviceType",
                _STUB(lambda dev, link, dtype: 13),
            )


ctypes.CDLL.__init__ = _patched_cdll_init

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _tokenize_response_only(example, tokenizer, max_length: int):
    """Tokenize prompt+response and mask prompt tokens with -100 labels."""
    prompt = str(example.get("prompt") or "")
    response = str(example.get("response") or "")
    if not response:
        raise ValueError("SFT example has empty response")

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    response_ids = tokenizer(
        response,
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    if (
        prompt_ids
        and response_ids
        and prompt_ids[0] == response_ids[0]
        and getattr(tokenizer, "bos_token_id", None) == response_ids[0]
    ):
        response_ids = response_ids[1:]
    if not response_ids:
        raise ValueError("SFT example response produced no tokens")

    if len(response_ids) >= max_length:
        input_ids = response_ids[:max_length]
        labels = list(input_ids)
    else:
        prompt_budget = max_length - len(response_ids)
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class ResponseOnlyDataCollator:
    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        pad_id = self.tokenizer.pad_token_id
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [self.label_pad_token_id] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _run():
    if len(sys.argv) < 2:
        print("Usage: python _sft_runner.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        cfg = json.load(f)

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    from src.training.lora_config import get_lora_config

    dataset_path = cfg["dataset_path"]
    if dataset_path and os.path.exists(dataset_path):
        raw_dataset = load_from_disk(dataset_path)
        logger.info("Loaded %s SFT examples from %s", len(raw_dataset), dataset_path)
    else:
        logger.error("Dataset not found at %s", dataset_path)
        sys.exit(1)

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
        cfg["model_type"],
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
    )

    logger.info("Loading model: %s (dtype=%s)", cfg["model_name_or_path"], torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch_dtype,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    from peft import get_peft_model

    model = get_peft_model(model, lora_config)
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    tokenized = raw_dataset.map(
        lambda example: _tokenize_response_only(
            example,
            tokenizer=tokenizer,
            max_length=cfg["max_length"],
        ),
        remove_columns=list(raw_dataset.column_names),
        desc="Tokenizing response-only SFT data",
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=ResponseOnlyDataCollator(tokenizer),
    )

    logger.info("Starting SFT training...")
    trainer.train()
    logger.info("Training complete.")

    output_dir = cfg["output_dir"]
    adapter_dir = output_dir + "_adapter"
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("LoRA adapter saved to: %s", adapter_dir)

    merge_lora = cfg.get("merge_lora", False)
    if merge_lora and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
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
        logger.info("Merged model saved to: %s", output_dir)
        del base_model, merged_model
    elif merge_lora:
        logger.warning("No adapter_config.json found, saving model as-is.")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        logger.info("Skipping LoRA merge; downstream vLLM will load adapter directly.")

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass

    logger.info("SFT runner finished successfully.")


if __name__ == "__main__":
    _run()
