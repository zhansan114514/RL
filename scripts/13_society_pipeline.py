"""
Diverse Actor-Critic Society 一键全流程 (Phase 1-6).
串联 scripts/07-12，断点续跑，日志统一输出到 output/ 目录。

Usage:
    python scripts/13_society_pipeline.py --config configs/society/experiment_mmlu.yaml
    # 跳过已完成的阶段
    python scripts/13_society_pipeline.py --config configs/society/experiment_mmlu.yaml --skip 1 2
    # 只运行指定阶段
    python scripts/13_society_pipeline.py --config configs/society/experiment_mmlu.yaml --only 5 6
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# 每个阶段的 (脚本文件名, 阶段描述)
PHASES = [
    ("07_bootstrap_actors.py", "Phase 1: Bootstrap 数据生成"),
    ("08_classify_data.py",    "Phase 2: 数据分类 (GLM API)"),
    ("09_diversify_actors.py", "Phase 3: Actor 分化训练"),
    ("10_diversify_critics.py","Phase 4: Critic 分化训练"),
    ("11_society_train.py",    "Phase 5: Society 交替训练"),
    ("12_society_evaluate.py", "Phase 6: 评估 + 消融实验"),
]

# 每个阶段完成后的标记文件（用于断点续跑）
MARKER_DIR = "output/society_mmlu/.pipeline_markers"


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Diverse Actor-Critic Society Pipeline")
    parser.add_argument("--config", type=str, default="configs/society/experiment_mmlu.yaml",
                        help="YAML config path")
    parser.add_argument("--skip", type=int, nargs="*", default=[],
                        help="Skip these phases (e.g. --skip 1 2)")
    parser.add_argument("--only", type=int, nargs="*", default=[],
                        help="Only run these phases (e.g. --only 5 6)")
    return parser.parse_args()


def get_marker_path(phase_num: int, cache_dir: str) -> str:
    return os.path.join(cache_dir, f".phase{phase_num}_done")


def is_phase_done(phase_num: int, cache_dir: str) -> bool:
    return os.path.exists(get_marker_path(phase_num, cache_dir))


def mark_phase_done(phase_num: int, cache_dir: str):
    marker = get_marker_path(phase_num, cache_dir)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def _get_api_key_from_config(config_path: str) -> str | None:
    """Extract api_key from step02_classify section of the YAML config."""
    try:
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("step02_classify", {}).get("api_key")
    except Exception:
        return None


def run_phase(phase_num: int, script: str, desc: str, config_path: str) -> bool:
    """Run a single phase as a subprocess. Returns True on success."""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(scripts_dir, script)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  [{phase_num}/6] {desc}")
    logger.info(f"  Script: {script}")
    logger.info(f"  Config: {config_path}")
    logger.info("=" * 70)

    start = time.time()
    try:
        sub_env = {**os.environ, "PYTHONPATH": os.path.dirname(scripts_dir)}
        # Phase 5 needs GLM API key for style/error classification
        if phase_num == 5 and not sub_env.get("GLM_API_KEY"):
            api_key = _get_api_key_from_config(config_path)
            if api_key:
                sub_env["GLM_API_KEY"] = api_key
                logger.info("  Injected GLM_API_KEY from config step02_classify.api_key")
        result = subprocess.run(
            [sys.executable, script_path, "--config", config_path],
            cwd=os.path.dirname(scripts_dir),
            env=sub_env,
            check=True,
        )
        elapsed = time.time() - start
        logger.info(f"  Phase {phase_num} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        logger.error(f"  Phase {phase_num} FAILED after {elapsed:.1f}s (exit code {e.returncode})")
        return False


def main():
    args = parse_args()
    config_path = args.config

    # Read cache_dir from config for markers
    cache_dir = "output/society_mmlu"
    try:
        from src.utils.config import ConfigManager
        cfg = ConfigManager.initialize(config_path=config_path)
        common = cfg._config.get("common", {})
        cache_dir = common.get("cache_dir", cache_dir)
    except Exception:
        pass

    os.makedirs(cache_dir, exist_ok=True)

    logger.info("#" * 70)
    logger.info("  Diverse Actor-Critic Society Pipeline")
    logger.info(f"  Config:   {config_path}")
    logger.info(f"  Output:   {cache_dir}")
    logger.info(f"  Log:      {cache_dir}/logs/")
    logger.info("#" * 70)

    pipeline_start = time.time()
    completed = []
    failed = []

    for i, (script, desc) in enumerate(PHASES):
        phase_num = i + 1

        # --only filter
        if args.only and phase_num not in args.only:
            logger.info(f"  [Phase {phase_num}] SKIPPED (not in --only)")
            continue

        # --skip filter
        if phase_num in args.skip:
            logger.info(f"  [Phase {phase_num}] SKIPPED (--skip)")
            continue

        # 断点续跑：跳过已完成的阶段
        if is_phase_done(phase_num, cache_dir):
            logger.info(f"  [Phase {phase_num}] SKIPPED (already done)")
            completed.append(phase_num)
            continue

        ok = run_phase(phase_num, script, desc, config_path)
        if ok:
            mark_phase_done(phase_num, cache_dir)
            completed.append(phase_num)
        else:
            failed.append(phase_num)
            logger.error(f"  Pipeline stopped at Phase {phase_num}. Fix and re-run.")
            break

    total_time = time.time() - pipeline_start

    # Summary
    logger.info("")
    logger.info("#" * 70)
    logger.info("  Pipeline Summary")
    logger.info("#" * 70)
    logger.info(f"  Completed: {completed}")
    logger.info(f"  Failed:    {failed}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
    logger.info(f"  Output dir: {cache_dir}")
    if not failed:
        logger.info("  All phases completed successfully!")
    logger.info("#" * 70)


if __name__ == "__main__":
    main()
