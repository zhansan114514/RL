"""
Logging and experiment tracking utilities.

Provides:
- Structured logging setup with rich formatting
- WandB integration for experiment tracking
- JSONL-based local metrics persistence
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
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
    Unified experiment logger that logs to Python logging, WandB, and local JSONL.

    Usage:
        exp = ExperimentLogger("acc-collab", config=cfg)
        exp.log_metrics("train", {"loss": 0.5, "lr": 1e-4}, step=100)
        exp.finish()

    Or from ConfigManager:
        exp = ExperimentLogger.from_config(output_dir="experiments/run1")
    """

    def __init__(
        self,
        project: str = "acc-collab",
        name: Optional[str] = None,
        config: Optional[dict] = None,
        use_wandb: bool = False,
        log_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.use_wandb = use_wandb
        self._py_logger = logging.getLogger("experiment")
        self._output_dir = output_dir or log_dir
        self._metrics_file: Optional[str] = None

        # Set up local JSONL metrics file
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            self._metrics_file = os.path.join(self._output_dir, "metrics.jsonl")

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

    @classmethod
    def from_config(
        cls,
        output_dir: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> ExperimentLogger:
        """
        Create an ExperimentLogger using the initialized ConfigManager singleton.

        Reads wandb and logging settings from ConfigManager. If ConfigManager
        is not initialized, falls back to defaults.

        Args:
            output_dir: Override output directory. Defaults to config's log_dir.
            name: Experiment run name.
            config: Configuration dict to log to wandb.

        Returns:
            A configured ExperimentLogger instance.
        """
        from src.utils.config import ConfigManager

        project = "acc-collab"
        use_wandb = False
        log_dir = None

        if ConfigManager.is_initialized():
            cfg = ConfigManager.instance()
            project = cfg.get("wandb_project", "acc-collab")
            log_dir = cfg.get("log_dir", None)
            use_wandb = cfg.get("use_wandb", False)

        effective_output = output_dir or log_dir

        return cls(
            project=project,
            name=name,
            config=config,
            use_wandb=use_wandb,
            log_dir=log_dir,
            output_dir=effective_output,
        )

    def log_metrics(
        self,
        phase: str,
        metrics: dict,
        step: Optional[int] = None,
    ) -> None:
        """
        Record training/evaluation metrics.

        Logs to three destinations simultaneously:
        1. Python logger (always)
        2. WandB (if enabled)
        3. Local JSONL file (if output_dir is set)

        Args:
            phase: The experiment phase, e.g. "train", "eval", "trajectory".
            metrics: Dict of metric name -> value, e.g. {"loss": 0.5, "accuracy": 0.8}.
            step: Global step number or iteration index.
        """
        # 1. Print to Python logger
        step_str = f" (step={step})" if step is not None else ""
        self._py_logger.info(f"[{phase}] Metrics{step_str}: {metrics}")

        # 2. Log to WandB if enabled
        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
                    wandb.log(wandb_metrics, step=step)
            except ImportError:
                pass

        # 3. Append to local JSONL file
        if self._metrics_file:
            self._write_jsonl(phase, metrics, step)

    def _write_jsonl(
        self,
        phase: str,
        metrics: dict,
        step: Optional[int],
    ) -> None:
        """Write a single metrics record to the JSONL file."""
        record = {
            "phase": phase,
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        record.update(metrics)
        try:
            with open(self._metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            self._py_logger.warning(f"Failed to write metrics JSONL: {e}")

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
