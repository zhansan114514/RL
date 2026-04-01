"""Shared utilities for pipeline scripts."""

from __future__ import annotations

import argparse
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


def resolve_config(
    config_path: str,
    step_key: str,
    defaults: dict,
    common_keys: tuple[str, ...] = (),
    allowed_datasets: tuple[str, ...] | None = None,
) -> argparse.Namespace:
    """Merge defaults <- common section <- step-specific section.

    Args:
        config_path: Path to the YAML config file.
        step_key: Top-level key for this step (e.g. "step01").
        defaults: Dict of default values (STEP_DEFAULTS).
        common_keys: Keys to read from the "common" section.
        allowed_datasets: If set, validate the dataset key against this list.
    """
    cfg = load_yaml_config(config_path)
    common_cfg = cfg.get("common", {})
    step_cfg = cfg.get(step_key, {})

    if not isinstance(common_cfg, dict):
        raise ValueError("Config key 'common' must be a mapping.")
    if not isinstance(step_cfg, dict):
        raise ValueError(f"Config key '{step_key}' must be a mapping.")

    merged = dict(defaults)

    for key in common_keys:
        if key in common_cfg and common_cfg[key] is not None:
            merged[key] = common_cfg[key]

    for key in defaults:
        if key in step_cfg and step_cfg[key] is not None:
            merged[key] = step_cfg[key]

    if allowed_datasets and merged.get("dataset") not in allowed_datasets:
        raise ValueError(
            f"Invalid dataset '{merged['dataset']}'. "
            f"Expected one of: {', '.join(allowed_datasets)}"
        )

    return argparse.Namespace(config=config_path, **merged)
