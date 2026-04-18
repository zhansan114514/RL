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
            "enforce_eager": False,
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

        # V100 (compute capability 7.0) does not support BF16
        # vLLM's dtype="auto" handles this automatically, but we add
        # explicit logging for clarity
        requested_dtype = self._init_kwargs.get("dtype", "auto")
        if requested_dtype == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    # Check first device (vLLM will use same dtype for all)
                    dev_name = torch.cuda.get_device_name(0)
                    if "V100" in dev_name or "TITAN V" in dev_name:
                        logger.info(
                            f"Detected {dev_name} (no BF16 support). "
                            "vLLM will use float16 automatically."
                        )
            except Exception:
                pass  # Silently skip if CUDA detection fails

        # Set CUDA_VISIBLE_DEVICES to target the desired physical GPU.
        # cuda_device is always treated as a physical GPU ID.
        # Each vLLM instance spawns its own child process (EngineCore)
        # which inherits CUDA_VISIBLE_DEVICES at spawn time, so we can
        # safely change it between model loads without affecting already-
        # running engines.
        if self._cuda_device is not None:
            target_physical = str(self._cuda_device)
            os.environ["CUDA_VISIBLE_DEVICES"] = target_physical
            logger.info(
                f"Loading model on physical GPU {target_physical}: {self.model_name}"
            )
        else:
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
        seed: Optional[int] = None,
    ) -> list[str]:
        """Generate text from prompts."""
        self._ensure_loaded()
        from vllm import SamplingParams

        if isinstance(prompts, str):
            prompts = [prompts]

        sampling_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stop": stop,
        }
        if seed is not None:
            sampling_kwargs["seed"] = seed

        params = SamplingParams(**sampling_kwargs)
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
