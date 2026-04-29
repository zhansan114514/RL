"""Shared utilities for pipeline scripts."""

from __future__ import annotations

import importlib
import logging
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO, seed: int | None = None) -> None:
    """Configure root logger (idempotent) and optionally fix random seed."""
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    if seed is not None:
        from src.utils.seeding import fix_seed
        fix_seed(seed)
        logging.getLogger(__name__).info(f"Random seed fixed to {seed}")


def load_yaml_config(config_path: str) -> dict:
    """Load a YAML config file and return a plain dict."""
    omegaconf_module = importlib.import_module("omegaconf")
    OmegaConf = omegaconf_module.OmegaConf

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    loaded = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(loaded, resolve=True)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a top-level key-value mapping.")
    return cfg
