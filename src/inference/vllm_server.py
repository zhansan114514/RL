"""
vLLM-based inference service for actor and critic models.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VLLMInference:
    """Inference engine using vLLM for efficient LLM serving."""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048,
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self._llm = None
        self._tokenizer = None
        self._init_kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }

    def _ensure_loaded(self) -> None:
        """Lazy-load the vLLM engine on first use."""
        if self._llm is not None:
            return
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM required. pip install vllm")
        logger.info(f"Loading model: {self.model_name}")
        self._llm = LLM(self.model_name, **self._init_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

    def generate(
        self,
        prompts: str | list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        n: int = 1,
        stop: Optional[list[str]] = None,
    ) -> list[str]:
        """Generate text from prompts."""
        self._ensure_loaded()
        from vllm import SamplingParams

        if isinstance(prompts, str):
            prompts = [prompts]

        params = SamplingParams(
            max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, n=n, stop=stop,
        )
        outputs = self._llm.generate(prompts, params)
        return [c.text for o in outputs for c in o.outputs]

    def generate_single(
        self, prompt: str, max_tokens: int = 256, temperature: float = 0.7,
    ) -> str:
        """Generate a single response."""
        results = self.generate(prompt, max_tokens=max_tokens,
                                temperature=temperature, n=1)
        return results[0] if results else ""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Apply chat template to messages."""
        self._ensure_loaded()
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("Assistant:")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"VLLMInference(model={self.model_name})"
