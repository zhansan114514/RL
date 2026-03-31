"""
Configuration management using Hydra/OmegaConf.

Provides a unified interface for loading and merging configs from
configs/base.yaml, configs/model/, configs/data/, and configs/train/.
"""

from __future__ import annotations

import os
import logging
from typing import Optional

from omegaconf import OmegaConf, DictConfig, ListConfig

logger = logging.getLogger(__name__)

# Project root directory (one level up from src/utils/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")


def load_config(
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    train: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """
    Load and merge configuration files.

    Args:
        model: Model config name (e.g. "gemma2_2b", "llama3_8b").
        dataset: Dataset config name (e.g. "boolq", "mmlu").
        train: Training config name (e.g. "dpo_actor", "dpo_critic").
        overrides: List of OmegaConf override strings (e.g. ["training.lr=1e-4"]).

    Returns:
        Merged DictConfig with all config groups.
    """
    configs = []

    # 1. Base config
    base_path = os.path.join(CONFIGS_DIR, "base.yaml")
    if os.path.exists(base_path):
        configs.append(OmegaConf.load(base_path))
    else:
        logger.warning(f"Base config not found: {base_path}")
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
    if train:
        train_cfg = _load_group("train", train)
        if train_cfg:
            configs.append(train_cfg)

    # Merge all
    merged = OmegaConf.merge(*configs)

    # Apply CLI-style overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, override_cfg)

    return merged


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
    """Convert OmegaConf config to a flat dictionary with dot-notation keys."""
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)


def save_config(cfg: DictConfig, path: str) -> None:
    """Save config to a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    OmegaConf.save(cfg, path)
    logger.info(f"Config saved to: {path}")


def get_model_name(cfg: DictConfig) -> str:
    """Extract model name from config."""
    return cfg.get("model", {}).get("name", "")


def get_model_type(cfg: DictConfig) -> str:
    """Extract model architecture type from config."""
    return cfg.get("model", {}).get("type", "gemma2")


def get_dataset_info(cfg: DictConfig) -> dict:
    """Extract dataset-related config as a plain dict."""
    return OmegaConf.to_container(cfg.get("dataset", {}), resolve=True)
