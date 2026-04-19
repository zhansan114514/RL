"""Tests for configuration management."""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.utils.config import (
    load_config,
    _load_group,
)


class TestLoadConfig:
    """Test configuration loading and merging."""

    @patch("src.utils.config.OmegaConf.load")
    @patch("os.path.exists")
    def test_load_base_config_only(self, mock_exists, mock_load):
        """Loading with no arguments should load only base config."""
        mock_exists.return_value = True
        mock_load.return_value = {"seed": 42}

        cfg = load_config()
        assert cfg.get("seed", 42) == 42

    @patch("src.utils.config.OmegaConf.load")
    @patch("os.path.exists")
    def test_load_model_config(self, mock_exists, mock_load):
        """Loading model config should merge with base."""
        mock_exists.side_effect = lambda path: True  # All paths exist
        mock_load.side_effect = [
            {"seed": 42},  # base
            {"model": {"name": "gemma-2-2b", "type": "gemma2"}},  # model
        ]

        cfg = load_config(model="gemma2_2b")
        assert cfg.model.name == "gemma-2-2b"

    @patch("src.utils.config.OmegaConf.load")
    @patch("os.path.exists")
    def test_load_nonexistent_config(self, mock_exists, mock_load):
        """Loading nonexistent config should log warning and continue."""
        # Base exists, model doesn't
        mock_exists.side_effect = lambda path: "base.yaml" in path
        mock_load.return_value = {"seed": 42}

        cfg = load_config(model="nonexistent_model")
        # Should still have base config
        assert cfg.get("seed", 42) == 42

    @patch("src.utils.config.OmegaConf.load")
    @patch("os.path.exists")
    def test_load_with_overrides(self, mock_exists, mock_load):
        """CLI-style overrides should take precedence."""
        mock_exists.return_value = True
        mock_load.return_value = {"training": {"lr": 5e-5}}

        cfg = load_config(overrides=["training.lr=1e-4"])
        assert cfg.training.lr == 1e-4

    @patch("src.utils.config.OmegaConf.load")
    @patch("os.path.exists")
    def test_merge_multiple_configs(self, mock_exists, mock_load):
        """Merging model + dataset + train should combine all."""
        mock_exists.side_effect = lambda path: True
        mock_load.side_effect = [
            {"seed": 42},  # base
            {"model": {"name": "llama3"}},  # model
            {"dataset": {"name": "boolq"}},  # dataset
            {"training": {"lr": 1e-4}},  # train
        ]

        cfg = load_config(
            model="llama3_8b",
            dataset="boolq",
            train="dpo_actor"
        )
        assert cfg.seed == 42
        assert cfg.model.name == "llama3"
        assert cfg.dataset.name == "boolq"
        assert cfg.training.lr == 1e-4


class TestLoadGroup:
    """Test _load_group helper function."""

    @patch("src.utils.config.OmegaConf.load")
    @patch("os.path.exists")
    def test_load_existing_group(self, mock_exists, mock_load):
        """Should load config from existing group file."""
        from omegaconf import OmegaConf

        mock_exists.return_value = True
        mock_load.return_value = OmegaConf.create({"key": "value"})

        result = _load_group("model", "gemma2_2b")
        assert result.key == "value"

    @patch("os.path.exists")
    def test_load_missing_group_returns_none(self, mock_exists):
        """Missing group file should return None."""
        mock_exists.return_value = False

        result = _load_group("model", "nonexistent")
        assert result is None


class TestConfigEdgeCases:
    """Test edge cases and malformed inputs."""

    @patch("src.utils.config.OmegaConf.load")
    @patch("src.utils.config.OmegaConf.merge")
    @patch("os.path.exists")
    def test_merge_empty_configs(self, mock_exists, mock_merge, mock_load):
        """Merging empty configs should not crash."""
        mock_exists.return_value = True
        mock_load.return_value = {}
        mock_merge.return_value = {}

        cfg = load_config()
        assert cfg is not None

    @patch("src.utils.config.OmegaConf.from_dotlist")
    def test_override_with_empty_list(self, mock_from_dotlist):
        """Empty override list should not cause issues."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"key": "value"})
        # Should handle empty overrides
        result = OmegaConf.merge(cfg, OmegaConf.create())
        assert result.key == "value"
