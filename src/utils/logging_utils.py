"""
Logging and experiment tracking utilities.

Provides:
- Structured logging setup with rich formatting
- WandB integration for experiment tracking
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)


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


def setup_wandb(
    project: str = "acc-collab",
    name: Optional[str] = None,
    config: Optional[dict] = None,
    enabled: bool = True,
    dir: Optional[str] = None,
) -> Optional[str]:
    """
    Initialize WandB experiment tracking.

    Args:
        project: WandB project name.
        name: Experiment run name.
        config: Configuration dict to log.
        enabled: Whether to actually enable WandB (False = dry run).
        dir: WandB directory for local storage.

    Returns:
        WandB run URL, or None if disabled.
    """
    if not enabled:
        os.environ["WANDB_MODE"] = "disabled"
        logger.info("WandB disabled (dry run mode)")
        return None

    try:
        import wandb

        if dir:
            os.makedirs(dir, exist_ok=True)

        run = wandb.init(
            project=project,
            name=name,
            config=config,
            dir=dir,
            reinit=True,
        )
        url = run.url if hasattr(run, "url") else str(run.url)
        logger.info(f"WandB initialized: {url}")
        return url
    except ImportError:
        logger.warning("wandb not installed, skipping experiment tracking")
        return None
    except Exception as e:
        logger.warning(f"WandB init failed: {e}, continuing without tracking")
        return None


def finish_wandb() -> None:
    """Finish the current WandB run if active."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


class ExperimentLogger:
    """
    Unified experiment logger that logs to both Python logging and WandB.

    Usage:
        exp = ExperimentLogger("acc-collab", config=cfg)
        exp.log_metrics({"train/loss": 0.5, "train/lr": 1e-4}, step=100)
        exp.finish()
    """

    def __init__(
        self,
        project: str = "acc-collab",
        name: Optional[str] = None,
        config: Optional[dict] = None,
        use_wandb: bool = False,
        log_dir: Optional[str] = None,
    ):
        self.use_wandb = use_wandb
        self._py_logger = logging.getLogger("experiment")

        if log_dir:
            setup_logging(log_file=os.path.join(log_dir, "experiment.log"))

        if use_wandb:
            self._wandb_url = setup_wandb(
                project=project,
                name=name,
                config=config,
                enabled=True,
                dir=log_dir,
            )
        else:
            self._wandb_url = None

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """Log metrics to WandB and Python logger."""
        step_str = f" (step={step})" if step else ""
        self._py_logger.info(f"Metrics{step_str}: {metrics}")

        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics, step=step)
            except ImportError:
                pass

    def log_summary(self, summary: dict) -> None:
        """Log final summary metrics."""
        self._py_logger.info(f"Summary: {summary}")
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    for k, v in summary.items():
                        wandb.run.summary[k] = v
            except ImportError:
                pass

    def finish(self) -> None:
        """Finish the experiment."""
        if self.use_wandb:
            finish_wandb()
