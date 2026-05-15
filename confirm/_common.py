"""Shared helpers for Society phase confirmation scripts."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import ConfigManager  # noqa: E402


def init_config(config_path: str) -> ConfigManager:
    """Load a fresh effective experiment config."""
    ConfigManager.reset()
    return ConfigManager.initialize(config_path=config_path)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def default_report_path(cfg: ConfigManager, filename: str) -> Path:
    cache_dir = cfg.get("common.cache_dir", "output/society")
    return Path(str(cache_dir)) / "confirm" / filename


def resolve_output_path(
    explicit: str | None,
    cfg: ConfigManager,
    filename: str,
) -> Path:
    return Path(explicit) if explicit else default_report_path(cfg, filename)


def rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def gate(
    name: str,
    passed: bool | None,
    actual: Any = None,
    threshold: Any = None,
    details: str = "",
) -> dict[str, Any]:
    """Build a normalized gate record. passed=None means informational only."""
    status = "info" if passed is None else ("pass" if passed else "fail")
    return {
        "name": name,
        "status": status,
        "actual": actual,
        "threshold": threshold,
        "details": details,
    }


def overall_status(gates: list[dict[str, Any]]) -> str:
    return "fail" if any(item.get("status") == "fail" for item in gates) else "pass"


def print_summary(report_path: Path, gates: list[dict[str, Any]]) -> None:
    status = overall_status(gates)
    print(f"status: {status}")
    for item in gates:
        name = item.get("name", "")
        state = item.get("status", "")
        actual = item.get("actual")
        threshold = item.get("threshold")
        suffix = ""
        if threshold is not None:
            suffix = f" (actual={actual}, threshold={threshold})"
        elif actual is not None:
            suffix = f" (actual={actual})"
        print(f"- {state}: {name}{suffix}")
    print(f"report: {report_path}")
    if status == "fail":
        raise SystemExit(1)


def get_api_key(step_cfg: dict[str, Any] | None = None) -> str:
    step_cfg = step_cfg or {}
    return (
        str(step_cfg.get("api_key") or "")
        or os.environ.get("GPT_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("GLM_API_KEY", "")
    )


def sample_items(items: list[Any], max_items: int | None, seed: int) -> list[Any]:
    if max_items is None or max_items <= 0 or len(items) <= max_items:
        return list(items)
    selected = list(items)
    random.Random(seed).shuffle(selected)
    return selected[:max_items]


def format_float(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)
