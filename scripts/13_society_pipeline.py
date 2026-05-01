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

import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.utils.config import ConfigManager

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


def _load_config(config_path: str) -> dict[str, Any]:
    cfg = ConfigManager.initialize(config_path=config_path)
    return cfg.to_dict()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_jsonable(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    )


def _hash_paths(paths: list[str]) -> str:
    entries: list[dict[str, Any]] = []
    for raw_path in sorted(set(paths)):
        path = Path(raw_path)
        if path.is_file():
            stat = path.stat()
            entries.append({
                "path": str(path),
                "type": "file",
                "size": stat.st_size,
                "sha256": _hash_file(path),
            })
        elif path.is_dir():
            files = []
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                stat = child.stat()
                files.append({
                    "path": str(child.relative_to(path)),
                    "size": stat.st_size,
                    "sha256": _hash_file(child),
                })
            entries.append({"path": str(path), "type": "dir", "files": files})
        else:
            entries.append({"path": str(path), "type": "missing"})
    return _hash_jsonable(entries)


def _git_commit(repo_root: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _config_hash(config_path: str) -> str:
    return _hash_file(Path(config_path))


def _step(config: dict[str, Any], key: str) -> dict[str, Any]:
    return config.get(key, {}) or {}


def _phase_input_paths(phase_num: int, config: dict[str, Any], cache_dir: str) -> list[str]:
    step02 = _step(config, "step02_classify")
    step03 = _step(config, "step03_diversify_actors")
    step04 = _step(config, "step04_diversify_critics")
    step05 = _step(config, "step05_train_society")
    step06 = _step(config, "step06_evaluate")

    bootstrap_dir = step02.get("input_dir") or os.path.join(cache_dir, "bootstrap")
    classified_dir = step03.get("input_dir") or step04.get("input_dir") or os.path.join(cache_dir, "classified")
    actors_dir = step04.get("actor_base_dir") or step05.get("actor_base_dir") or os.path.join(cache_dir, "actors")
    critics_dir = step05.get("critic_base_dir") or os.path.join(cache_dir, "critics")
    society_dir = step06.get("society_dir") or step05.get("output_dir") or os.path.join(cache_dir, "society")

    if phase_num == 1:
        return []
    if phase_num == 2:
        return [os.path.join(bootstrap_dir, "trajectories.jsonl")]
    if phase_num == 3:
        return [
            os.path.join(classified_dir, "classified_data.json"),
            os.path.join(bootstrap_dir, "trajectories.jsonl"),
        ]
    if phase_num == 4:
        return [
            os.path.join(classified_dir, "classified_data.json"),
            os.path.join(actors_dir, "actor_registry.json"),
        ]
    if phase_num == 5:
        return [
            os.path.join(classified_dir, "classified_data.json"),
            os.path.join(actors_dir, "actor_registry.json"),
            os.path.join(critics_dir, "critic_registry.json"),
        ]
    if phase_num == 6:
        return [os.path.join(society_dir, "final_agent_registry.json")]
    raise ValueError(f"Unknown phase: {phase_num}")


def build_phase_fingerprint(
    phase_num: int,
    config_path: str,
    config: dict[str, Any],
    cache_dir: str,
) -> dict[str, Any]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_paths = _phase_input_paths(phase_num, config, cache_dir)
    return {
        "config_hash": _config_hash(config_path),
        "git_commit": _git_commit(repo_root),
        "input_hash": _hash_paths(input_paths),
        "input_paths": input_paths,
    }


def get_marker_path(phase_num: int, cache_dir: str) -> str:
    return os.path.join(cache_dir, f".phase{phase_num}_done")


def is_phase_done(
    phase_num: int,
    cache_dir: str,
    expected_fingerprint: dict[str, Any],
) -> bool:
    marker = get_marker_path(phase_num, cache_dir)
    if not os.path.exists(marker):
        return False

    try:
        with open(marker) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.info(f"  [Phase {phase_num}] marker is not JSON; rerunning")
        return False

    for key in ("config_hash", "git_commit", "input_hash"):
        if payload.get(key) != expected_fingerprint.get(key):
            logger.info(
                f"  [Phase {phase_num}] marker {key} mismatch; rerunning"
            )
            return False

    return True


def mark_phase_done(
    phase_num: int,
    cache_dir: str,
    fingerprint: dict[str, Any],
):
    marker = get_marker_path(phase_num, cache_dir)
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        json.dump({
            "phase": phase_num,
            "completed_at": dt.datetime.now(dt.UTC).isoformat(),
            **fingerprint,
        }, f, indent=2)


def _get_api_key_from_config(config_path: str) -> str | None:
    """Extract api_key from the effective ConfigManager config."""
    cfg = ConfigManager.initialize(config_path=config_path)
    return cfg.step("step02_classify").get("api_key") or cfg.get("api.api_key")


def _build_subprocess_env(phase_num: int, config_path: str) -> dict[str, str]:
    """Build subprocess env and share the GLM key with strict classification phases."""
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    sub_env = {**os.environ, "PYTHONPATH": os.path.dirname(scripts_dir)}
    if phase_num in {4, 5} and not sub_env.get("GLM_API_KEY"):
        api_key = _get_api_key_from_config(config_path)
        if api_key:
            sub_env["GLM_API_KEY"] = api_key
            logger.info("  Injected GLM_API_KEY from effective config")
    return sub_env


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
        sub_env = _build_subprocess_env(phase_num, config_path)
        subprocess.run(
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
    config = _load_config(config_path)

    # Read cache_dir from config for markers
    cache_dir = "output/society_mmlu"
    common = config.get("common", {}) or {}
    cache_dir = common.get("cache_dir", cache_dir)

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

        fingerprint = build_phase_fingerprint(
            phase_num=phase_num,
            config_path=config_path,
            config=config,
            cache_dir=cache_dir,
        )

        # 断点续跑：跳过已完成的阶段
        if is_phase_done(phase_num, cache_dir, fingerprint):
            logger.info(f"  [Phase {phase_num}] SKIPPED (marker fingerprint matched)")
            completed.append(phase_num)
            continue

        ok = run_phase(phase_num, script, desc, config_path)
        if ok:
            mark_phase_done(phase_num, cache_dir, fingerprint)
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
