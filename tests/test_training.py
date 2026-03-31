"""Tests for DPO training components."""

import pytest
from unittest.mock import patch, MagicMock


class TestLoRAConfig:
    """Test LoRA configuration creation."""

    def test_default_config(self):
        from src.training.lora_config import get_lora_config

        config = get_lora_config("llama3", r=256)
        assert config.r == 256
        assert config.lora_alpha == 512
        assert len(config.target_modules) > 0

    def test_gemma2_config(self):
        from src.training.lora_config import get_lora_config

        config = get_lora_config("gemma2")
        assert config.r == 256

    def test_custom_rank(self):
        from src.training.lora_config import get_lora_config

        config = get_lora_config("llama3", r=128, lora_alpha=256)
        assert config.r == 128
        assert config.lora_alpha == 256


class TestAlternatingTrain:
    """Test alternating training structure."""

    def test_alternating_returns_paths(self):
        """The function should return dict with actor_path and critic_path."""
        from src.training.alternating import alternating_train
        # This is a structural test - actual training requires GPU
        # We verify the function signature accepts expected args
        import inspect
        sig = inspect.signature(alternating_train)
        params = list(sig.parameters.keys())
        assert "actor_path" in params
        assert "critic_path" in params
        assert "num_iterations" in params
        assert "dataset" in params
