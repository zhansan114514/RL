"""
vLLM-based inference service for actor and critic models.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class VLLMInference:
    """Inference engine using vLLM for efficient LLM serving.

    Supports multi-GPU placement via the ``cuda_device`` parameter.
    When set, vLLM's engine child process will be restricted to that
    physical GPU by temporarily overriding ``CUDA_VISIBLE_DEVICES``.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 1024,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        cuda_device: Optional[int] = None,
    ):
        self.model_name = model_name
        self._llm = None
        self._tokenizer = None
        self._cuda_device = cuda_device

        self._init_kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "enforce_eager": True,
            "disable_log_stats": True,
        }

    def _ensure_loaded(self) -> None:
        """Lazy-load the vLLM engine on first use."""
        if self._llm is not None:
            return
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("vLLM required. pip install vllm")

        # Map logical device index to the corresponding physical GPU ID.
        # e.g. if CUDA_VISIBLE_DEVICES="2,3" and cuda_device=1,
        # we want physical GPU 3, so set CUDA_VISIBLE_DEVICES="3".
        _prev_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if self._cuda_device is not None:
            if _prev_cuda_vis:
                visible = [d.strip() for d in _prev_cuda_vis.split(",")]
            else:
                import torch
                visible = [str(i) for i in range(torch.cuda.device_count())]

            if self._cuda_device >= len(visible):
                raise ValueError(
                    f"cuda_device={self._cuda_device} but only {len(visible)} "
                    f"GPUs visible (CUDA_VISIBLE_DEVICES={_prev_cuda_vis})"
                )
            target_physical = visible[self._cuda_device]
            os.environ["CUDA_VISIBLE_DEVICES"] = target_physical
            logger.info(
                f"Loading model on physical GPU {target_physical} "
                f"(logical device {self._cuda_device}): {self.model_name}"
            )
        else:
            logger.info(f"Loading model: {self.model_name}")

        self._llm = LLM(self.model_name, **self._init_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

        # Restore original CUDA_VISIBLE_DEVICES (child process already started).
        if self._cuda_device is not None:
            if _prev_cuda_vis is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = _prev_cuda_vis
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

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

    def cleanup(self) -> None:
        """Explicitly release GPU memory."""
        if self._llm is not None:
            try:
                import gc
                import torch
                del self._llm
                del self._tokenizer
                self._llm = None
                self._tokenizer = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Cleaned up model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Cleanup failed for {self.model_name}: {e}")

    def __del__(self) -> None:
        """Destructor to ensure GPU memory is released."""
        self.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False

    def __repr__(self) -> str:
        return f"VLLMInference(model={self.model_name})"
