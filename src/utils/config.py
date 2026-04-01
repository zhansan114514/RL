"""
Configuration management with global ConfigManager singleton.

Usage:
    # At program entry point (once):
    from src.utils.config import ConfigManager
    cfg = ConfigManager.initialize(model="gemma2_2b", dataset="boolq")

    # Anywhere in the codebase:
    from src.utils.config import get_config, require_config
    max_len = get_config("inference.max_model_len", 4096)
    model_name = require_config("model.name")
"""

from __future__ import annotations

import os
import logging
from typing import Any, Optional

from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)

# Project root directory (three levels up from this file: utils/ -> src/ -> project/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")


class ConfigManager:
    """
    Global configuration manager (singleton).

    Load once at program start, access everywhere via get/require.
    All hardcoded parameters should be replaced with get_config() calls.
    """

    _instance: Optional[ConfigManager] = None
    _config: DictConfig

    def __init__(self):
        raise RuntimeError(
            "Do not instantiate ConfigManager directly. "
            "Use ConfigManager.initialize() or ConfigManager.instance()."
        )

    @classmethod
    def _create(cls) -> ConfigManager:
        """Internal: create instance bypassing __init__ guard."""
        obj = object.__new__(cls)
        obj._config = OmegaConf.create()
        return obj

    @classmethod
    def initialize(
        cls,
        config_path: Optional[str] = None,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        training: Optional[str] = None,
        overrides: Optional[list[str]] = None,
    ) -> ConfigManager:
        """
        Load configuration and create the singleton.

        Merge priority (later overrides earlier):
        1. configs/default.yaml (global defaults)
        2. configs/model/{model}.yaml
        3. configs/data/{dataset}.yaml
        4. configs/train/{training}.yaml
        5. config_path (experiment config)
        6. overrides list (CLI-style dotlist)

        Args:
            config_path: Path to an experiment config YAML (e.g. configs/config.yaml).
            model: Model config name (e.g. "gemma2_2b").
            dataset: Dataset config name (e.g. "boolq").
            training: Training config name (e.g. "dpo_actor").
            overrides: List of OmegaConf override strings (e.g. ["training.lr=1e-4"]).

        Returns:
            The initialized ConfigManager singleton.
        """
        configs = []

        # 1. Global defaults
        default_path = os.path.join(CONFIGS_DIR, "default.yaml")
        if os.path.exists(default_path):
            configs.append(OmegaConf.load(default_path))
        else:
            logger.warning(f"Default config not found: {default_path}")
            configs.append(OmegaConf.create())

        # 2. Model config
        if model:
            model_cfg = _load_group("model", model)
            if model_cfg:
                configs.append(model_cfg)

        # 3. Dataset config
        if dataset:
            data_cfg = _load_group("data", dataset)
            if data_cfg:
                configs.append(data_cfg)

        # 4. Training config
        if training:
            train_cfg = _load_group("train", training)
            if train_cfg:
                configs.append(train_cfg)

        # 5. Experiment config file
        if config_path and os.path.exists(config_path):
            exp_cfg = OmegaConf.load(config_path)
            configs.append(exp_cfg)

        # Merge all layers
        merged = OmegaConf.merge(*configs)

        # 6. CLI-style overrides (highest priority)
        if overrides:
            override_cfg = OmegaConf.from_dotlist(overrides)
            merged = OmegaConf.merge(merged, override_cfg)

        # Store
        instance = cls._create()
        instance._config = merged
        cls._instance = instance

        logger.debug(f"ConfigManager initialized with {len(configs)} config layers")
        return instance

    @classmethod
    def instance(cls) -> ConfigManager:
        """Get the initialized singleton. Raises if not initialized."""
        if cls._instance is None:
            raise RuntimeError(
                "ConfigManager not initialized. "
                "Call ConfigManager.initialize() at program start."
            )
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if the singleton has been initialized."""
        return cls._instance is not None

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing only)."""
        cls._instance = None

    # ---- Read interface ----

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dot-notation key.

        Examples:
            cfg.get("model.name")                    -> "google/gemma-2-2b-it"
            cfg.get("inference.max_model_len")       -> 4096
            cfg.get("nonexistent.key", "fallback")   -> "fallback"
        """
        keys = key.split(".")
        obj = self._config
        for k in keys:
            if hasattr(obj, "get"):
                obj = obj.get(k)
                if obj is None:
                    return default
            else:
                return default
        return obj if obj is not None else default

    def require(self, key: str) -> Any:
        """Get a config value, raising ConfigKeyError if missing."""
        val = self.get(key)
        if val is None:
            raise ConfigKeyError(f"Required config key '{key}' is missing or None")
        return val

    def section(self, key: str) -> dict:
        """
        Get a config section as a plain dict.

        Example:
            cfg.section("model") -> {"name": "...", "type": "..."}
        """
        obj = self.get(key)
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        return OmegaConf.to_container(obj, resolve=True) if hasattr(obj, "__iter__") else {}

    @property
    def raw(self) -> DictConfig:
        """Get the underlying OmegaConf DictConfig."""
        return self._config

    # ---- Write interface (for overrides and testing) ----

    def set(self, key: str, value: Any) -> None:
        """Override a single config value at runtime."""
        OmegaConf.update(self._config, key, value)

    def merge(self, overrides: dict) -> None:
        """Merge a batch of overrides at runtime."""
        self._config = OmegaConf.merge(self._config, OmegaConf.create(overrides))

    # ---- Persistence ----

    def save(self, path: str) -> None:
        """Save current effective config to a YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        OmegaConf.save(self._config, path)
        logger.info(f"Config saved to: {path}")

    def to_dict(self) -> dict:
        """Export config as a plain dict."""
        return OmegaConf.to_container(self._config, resolve=True, throw_on_missing=False)


class ConfigKeyError(Exception):
    """Raised when a required config key is missing."""
    pass


# ---- Module-level convenience functions ----

def get_config(key: str, default: Any = None) -> Any:
    """Shortcut: get a config value without importing ConfigManager."""
    return ConfigManager.instance().get(key, default)


def require_config(key: str) -> Any:
    """Shortcut: require a config value."""
    return ConfigManager.instance().require(key)


# ---- Backward-compatible functions ----

def load_config(
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    train: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """
    Backward-compatible config loading.

    Creates a ConfigManager singleton if not already initialized,
    then returns the raw DictConfig for legacy code.
    """
    mgr = ConfigManager.initialize(
        model=model, dataset=dataset, training=train, overrides=overrides
    )
    return mgr.raw


def _load_group(group: str, name: str) -> Optional[DictConfig]:
    """Load a config file from a config group directory."""
    path = os.path.join(CONFIGS_DIR, group, f"{name}.yaml")
    if os.path.exists(path):
        cfg = OmegaConf.load(path)
        logger.debug(f"Loaded config: {group}/{name}")
        return cfg
    logger.warning(f"Config not found: {path}")
    return None


def config_to_flat_dict(cfg: DictConfig) -> dict:
    """Convert OmegaConf config to a flat dictionary."""
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)


def save_config(cfg: DictConfig, path: str) -> None:
    """Save config to a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    OmegaConf.save(cfg, path)
    logger.info(f"Config saved to: {path}")


def get_model_name(cfg=None) -> str:
    """Extract model name from config (supports both DictConfig and ConfigManager)."""
    if isinstance(cfg, DictConfig):
        return cfg.get("model", {}).get("name", "") if cfg.get("model") else ""
    return ConfigManager.instance().get("model.name", "")


def get_model_type(cfg=None) -> str:
    """Extract model architecture type (supports both DictConfig and ConfigManager)."""
    if isinstance(cfg, DictConfig):
        return cfg.get("model", {}).get("type", "gemma2") if cfg.get("model") else "gemma2"
    return ConfigManager.instance().get("model.type", "gemma2")


def get_dataset_info(cfg=None) -> dict:
    """Extract dataset-related config as a plain dict."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg.get("dataset", {}), resolve=True)
    return ConfigManager.instance().section("dataset")
