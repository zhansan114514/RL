"""
Logging and experiment tracking utilities.

Provides structured logging setup with rich formatting.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    rich_format: bool = True,
) -> None:
    """
    Configure project-wide logging.

    Args:
        level: Logging level.
        log_file: Optional file path to write logs.
        rich_format: Use rich-style formatting with timestamps.
    """
    if rich_format:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
    else:
        fmt = "%(levelname)s %(name)s: %(message)s"
        datefmt = None

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )
