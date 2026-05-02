"""Shared utilities for pipeline scripts."""

from __future__ import annotations

import logging
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.runtime_env import configure_runtime_libraries

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Configure before phase scripts import libraries that may dlopen conda
# extensions.  This prevents spawned vLLM/DPO workers from resolving the
# system libstdc++ when the conda one is required.
configure_runtime_libraries()


def setup_logging(level: int = logging.INFO, seed: int | None = None) -> None:
    """Configure root logger (idempotent) and optionally fix random seed."""
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    if seed is not None:
        from src.utils.seeding import fix_seed
        fix_seed(seed)
        logging.getLogger(__name__).info(f"Random seed fixed to {seed}")
