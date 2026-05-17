"""Tests for response-only SFT tokenization and LoRA config wiring."""

from __future__ import annotations


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 101
    eos_token_id = 102

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=None,
    ):
        del truncation
        ids = [101] if add_special_tokens else []
        ids.extend(ord(ch) for ch in text)
        if max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_response_only_tokenization_masks_prompt_tokens():
    from src.training._sft_runner import _tokenize_response_only

    tokenizer = FakeTokenizer()
    example = {"prompt": "Prompt:", "response": " Answer"}
    tokenized = _tokenize_response_only(example, tokenizer, max_length=128)

    prompt_len = len(tokenizer("Prompt:", add_special_tokens=True)["input_ids"])
    expected = tokenizer("Prompt: Answer", add_special_tokens=True)["input_ids"] + [
        tokenizer.eos_token_id
    ]
    assert tokenized["input_ids"] == expected
    assert tokenized["labels"][:prompt_len] == [-100] * prompt_len
    assert tokenized["labels"][prompt_len:] == tokenized["input_ids"][prompt_len:]
    assert tokenized["labels"][-1] == tokenizer.eos_token_id
    assert any(label != -100 for label in tokenized["labels"])


def test_response_tokens_contribute_when_prompt_consumes_context():
    from src.training._sft_runner import _tokenize_response_only

    tokenizer = FakeTokenizer()
    tokenized = _tokenize_response_only(
        {"prompt": "Long prompt", "response": " X"},
        tokenizer,
        max_length=4,
    )

    assert len(tokenized["input_ids"]) == 4
    assert tokenized["labels"][-1] != -100


def test_response_only_collator_pads_labels_with_ignore_index():
    from src.training._sft_runner import ResponseOnlyDataCollator

    batch = ResponseOnlyDataCollator(FakeTokenizer())([
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
        {"input_ids": [1], "attention_mask": [1], "labels": [-100]},
    ])

    assert batch["input_ids"].tolist() == [[1, 2], [1, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [-100, -100]]


def test_sft_lora_config_uses_expected_model_type():
    from src.training.lora_config import DEFAULT_TARGET_MODULES, get_lora_config

    config = get_lora_config(model_type="qwen3", r=256, lora_alpha=512)

    assert config.r == 256
    assert config.lora_alpha == 512
    assert set(config.target_modules) == set(DEFAULT_TARGET_MODULES)
