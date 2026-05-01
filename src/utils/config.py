"""Configuration management with global ConfigManager singleton.

Usage:
    # At program entry point (once):
    from src.utils.config import ConfigManager
    cfg = ConfigManager.initialize(config_path="configs/society/experiment_mmlu.yaml")

    # Get global config values:
    max_len = cfg.get("inference.max_model_len", 4096)

    # Get step-specific config:
    step_cfg = cfg.step("step01_bootstrap", defaults={"num_agents": 5})
    args = step_cfg.to_namespace()  # argparse.Namespace for backward compat
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Optional

from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)

# Project root directory (three levels up from this file: utils/ -> src/ -> project/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
DEFAULT_CONFIG_PATH = os.path.join(CONFIGS_DIR, "default.yaml")
DEFAULT_LOCAL_CONFIG_PATH = os.path.join(CONFIGS_DIR, "local.yaml")

# Step-specific section key prefixes in experiment YAMLs
_STEP_PREFIXES = ("step", "phase")


class StepConfig:
    """Step-specific config with attribute access and argparse.Namespace compat."""

    def __init__(self, data: dict):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_namespace(self) -> argparse.Namespace:
        """Convert to argparse.Namespace for backward compatibility."""
        return argparse.Namespace(**self._data)

    def to_dict(self) -> dict:
        return dict(self._data)

    def __repr__(self) -> str:
        return f"StepConfig({self._data})"


class ConfigManager:
    """
    Global configuration manager (singleton).

    Load once at program start, access everywhere via get/require/step.
    Supports experiment YAMLs with ``common:`` and ``stepNN_name:`` sections.
    """

    _instance: Optional[ConfigManager] = None
    _config: DictConfig
    _loaded_paths: list[str]

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
        obj._loaded_paths = []
        return obj

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Resolve config paths from cwd first, then project root."""
        if os.path.isabs(path):
            return path
        if os.path.exists(path):
            return os.path.abspath(path)
        return os.path.join(PROJECT_ROOT, path)

    @classmethod
    def initialize(
        cls,
        config_path: Optional[str] = None,
        local_config_path: Optional[str] = None,
        overrides: Optional[list[str]] = None,
        load_local: bool = True,
    ) -> ConfigManager:
        """
        Load configuration and create the singleton.

        Merge priority (later overrides earlier):
        1. configs/default.yaml (global defaults)
        2. config_path experiment YAML
        3. configs/local.yaml (gitignored machine-local secrets/overrides)
        4. overrides list (CLI-style dotlist, highest priority)

        Args:
            config_path: Path to an experiment config YAML.
            local_config_path: Optional local secret/override YAML path.
            overrides: List of OmegaConf override strings (e.g. ["training.lr=1e-4"]).
            load_local: Whether to auto-load configs/local.yaml when present.

        Returns:
            The initialized ConfigManager singleton.
        """
        configs = []
        loaded_paths: list[str] = []

        # 1. Global defaults
        if os.path.exists(DEFAULT_CONFIG_PATH):
            configs.append(OmegaConf.load(DEFAULT_CONFIG_PATH))
            loaded_paths.append(DEFAULT_CONFIG_PATH)
        else:
            logger.warning(f"Default config not found: {DEFAULT_CONFIG_PATH}")
            configs.append(OmegaConf.create())

        # 2. Experiment config file
        resolved_config_path = cls._resolve_path(config_path) if config_path else None
        if resolved_config_path and os.path.exists(resolved_config_path):
            exp_cfg = OmegaConf.load(resolved_config_path)
            configs.append(exp_cfg)
            loaded_paths.append(resolved_config_path)
        elif resolved_config_path:
            logger.warning(f"Config file not found: {resolved_config_path}")

        # 3. Local config file for secrets and machine-local overrides
        env_local_path = os.environ.get("ACC_CONFIG_LOCAL", "")
        candidate_local_path = local_config_path or env_local_path or DEFAULT_LOCAL_CONFIG_PATH
        resolved_local_path = cls._resolve_path(candidate_local_path)
        if load_local and os.path.exists(resolved_local_path):
            local_cfg = OmegaConf.load(resolved_local_path)
            configs.append(local_cfg)
            loaded_paths.append(resolved_local_path)

        # Merge base layers
        merged = OmegaConf.merge(*configs)

        # 4. CLI-style overrides (highest priority)
        if overrides:
            override_cfg = OmegaConf.from_dotlist(overrides)
            merged = OmegaConf.merge(merged, override_cfg)

        # Store
        instance = cls._create()
        instance._config = merged
        instance._loaded_paths = loaded_paths
        cls._instance = instance

        logger.debug(f"ConfigManager initialized with loaded_paths={loaded_paths}")
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
        """Get a config value by dot-notation key."""
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
        """Get a config section as a plain dict."""
        obj = self.get(key)
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        return OmegaConf.to_container(obj, resolve=True) if hasattr(obj, "__iter__") else {}

    def step(self, step_key: str, defaults: Optional[dict] = None) -> StepConfig:
        """
        Get merged config for a specific pipeline step.

        Merge priority: ``defaults`` < ``common:`` section < top-level ``api:``
        section for API fields < step-specific section.

        Args:
            step_key: e.g. ``"step01_bootstrap"`` or ``"step01"``.
            defaults: Dict of default values (what was STEP_DEFAULTS).

        Returns:
            StepConfig with attribute access and ``to_namespace()``.
        """
        merged = dict(defaults) if defaults else {}

        cfg_dict = OmegaConf.to_container(self._config, resolve=True)
        if isinstance(cfg_dict, dict):
            common_cfg = cfg_dict.get("common", {})
            if isinstance(common_cfg, dict):
                for key, value in common_cfg.items():
                    if value is not None:
                        merged[key] = value

            api_cfg = cfg_dict.get("api", {})
            if isinstance(api_cfg, dict):
                api_aliases = {
                    "api_key": ("api_key", "key"),
                    "api_base": ("api_base", "base_url", "base"),
                    "api_model": ("api_model", "model"),
                }
                for target, aliases in api_aliases.items():
                    for alias in aliases:
                        value = api_cfg.get(alias)
                        if value:
                            merged[target] = value
                            break

            # Step-specific section overrides common and top-level API defaults.
            step_cfg = cfg_dict.get(step_key, {})
            if isinstance(step_cfg, dict):
                for key, value in step_cfg.items():
                    if value is not None:
                        merged[key] = value

        return StepConfig(merged)

    @property
    def loaded_paths(self) -> list[str]:
        """Config files loaded into the current effective config."""
        return list(self._loaded_paths)

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
