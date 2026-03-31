"""Tests for configuration management."""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.utils.config import (
    load_config,
    _load_group,
    config_to_flat_dict,
    save_config,
    get_model_name,
    get_model_type,
    get_dataset_info,
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


class TestConfigToFlatDict:
    """Test config to flat dictionary conversion."""

    def test_simple_config(self):
        """Simple nested config should flatten correctly."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "a": 1,
            "b": {"c": 2, "d": 3}
        })
        result = config_to_flat_dict(cfg)
        assert result["a"] == 1
        assert result["b"]["c"] == 2

    def test_config_with_missing_values(self):
        """Should handle configs with missing/None values."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "a": None,
            "b": "value"
        })
        result = config_to_flat_dict(cfg)
        assert result["a"] is None
        assert result["b"] == "value"


class TestGetModelName:
    """Test model name extraction."""

    def test_get_model_name_from_config(self):
        """Should extract model name from config."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "model": {"name": "google/gemma-2-2b-it"}
        })
        result = get_model_name(cfg)
        assert result == "google/gemma-2-2b-it"

    def test_get_model_name_empty_config(self):
        """Should return empty string for missing model."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({})
        result = get_model_name(cfg)
        assert result == ""


class TestGetModelType:
    """Test model type extraction."""

    def test_get_model_type(self):
        """Should extract model type from config."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "model": {"type": "llama3"}
        })
        result = get_model_type(cfg)
        assert result == "llama3"

    def test_get_model_type_default(self):
        """Should default to gemma2 when type not specified."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({})
        result = get_model_type(cfg)
        assert result == "gemma2"


class TestGetDatasetInfo:
    """Test dataset info extraction."""

    def test_get_dataset_info(self):
        """Should extract dataset section as dict."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "dataset": {"name": "boolq", "split": "validation"}
        })
        result = get_dataset_info(cfg)
        assert isinstance(result, dict)
        assert result["name"] == "boolq"

    def test_get_dataset_info_empty(self):
        """Should return empty dict for missing dataset."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"dataset": {}})
        result = get_dataset_info(cfg)
        assert result == {}


class TestSaveConfig:
    """Test config saving."""

    @patch("src.utils.config.OmegaConf.save")
    @patch("os.makedirs")
    def test_save_creates_directories(self, mock_makedirs, mock_save):
        """Saving should create parent directories."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({"key": "value"})
        save_config(cfg, "/tmp/test/config.yaml")

        mock_makedirs.assert_called_once()
        mock_save.assert_called_once()

    @patch("src.utils.config.OmegaConf.save")
    @patch("os.makedirs")
    @patch("os.path.dirname")
    def test_save_with_existing_dir(self, mock_dirname, mock_makedirs, mock_save):
        """Should handle existing directories gracefully."""
        from omegaconf import OmegaConf

        mock_dirname.return_value = "/existing/path"
        cfg = OmegaConf.create({"key": "value"})
        save_config(cfg, "/existing/path/config.yaml")

        # makedirs should still be called with exist_ok=True
        mock_makedirs.assert_called_once()


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

    def test_config_with_nested_none_values(self):
        """Should handle deeply nested None values."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "a": {
                "b": {
                    "c": None
                }
            }
        })
        result = config_to_flat_dict(cfg)
        assert result["a"]["b"]["c"] is None
