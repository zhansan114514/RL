"""
Diverse Actor-Critic Society 全流程统一脚本。

依次执行 6 个 Phase，日志和中间数据全部输出到 output/ 目录。
每个 Phase 完成后自动验证产出文件，失败则停止并报告。

Phase 1: Bootstrap   — 多 Agent 轨迹生成
Phase 2: Classify    — 推理风格 + 错误类型分类 (GLM API)
Phase 3: Diversify Actors  — 3 个 Actor LoRA DPO 训练
Phase 4: Diversify Critics — 4 个 Critic LoRA DPO 训练
Phase 5: Society Train     — N×M 交替训练
Phase 6: Evaluate          — A1-A5 消融实验

Usage:
    # 最简启动（使用默认配置）
    python scripts/run_society.py --config configs/society/experiment_v100.yaml

    # 指定 GPU
    CUDA_VISIBLE_DEVICES=4 python scripts/run_society.py --config configs/society/experiment_v100.yaml

    # 从某个 Phase 恢复（跳过已完成的阶段）
    python scripts/run_society.py --config configs/society/experiment_v100.yaml --start_phase 3

    # 只运行到某个 Phase
    python scripts/run_society.py --config configs/society/experiment_v100.yaml --end_phase 4

    # 跳过评估
    python scripts/run_society.py --config configs/society/experiment_v100.yaml --no_eval
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import load_yaml_config

# ============================================================
# 日志配置
# ============================================================

def setup_pipeline_logging(log_dir: str) -> logging.Logger:
    """配置全流程日志：同时输出到控制台和文件。"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("society_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件
    fh = logging.FileHandler(os.path.join(log_dir, "pipeline.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# Phase 定义
# ============================================================

PHASES = [
    {
        "id": 1,
        "name": "Bootstrap",
        "script": "scripts/07_bootstrap_actors.py",
        "step_key": "step01_bootstrap",
        "outputs": ["output/society/bootstrap/trajectories.jsonl"],
        "desc": "多 Agent 轨迹生成",
    },
    {
        "id": 2,
        "name": "Classify",
        "script": "scripts/08_classify_data.py",
        "step_key": "step02_classify",
        "outputs": ["output/society/classified/classified_data.json"],
        "desc": "推理风格 + 错误类型分类",
    },
    {
        "id": 3,
        "name": "Diversify Actors",
        "script": "scripts/09_diversify_actors.py",
        "step_key": "step03_diversify_actors",
        "outputs": ["output/society/actors/actor_registry.json"],
        "desc": "3 个 Actor LoRA DPO 训练",
    },
    {
        "id": 4,
        "name": "Diversify Critics",
        "script": "scripts/10_diversify_critics.py",
        "step_key": "step04_diversify_critics",
        "outputs": ["output/society/critics/critic_registry.json"],
        "desc": "4 个 Critic LoRA DPO 训练",
    },
    {
        "id": 5,
        "name": "Society Train",
        "script": "scripts/11_society_train.py",
        "step_key": "step05_train_society",
        "outputs": ["output/society/society/final_agent_registry.json"],
        "desc": "N×M 交替训练",
    },
    {
        "id": 6,
        "name": "Evaluate",
        "script": "scripts/12_society_evaluate.py",
        "step_key": "step06_evaluate",
        "outputs": ["output/society/eval/results.json"],
        "desc": "A1-A5 消融实验",
    },
]


# ============================================================
# 运行器
# ============================================================

def get_python_path() -> str:
    """获取当前 Python 解释器路径（确保子进程使用同一环境）。"""
    return sys.executable


def resolve_base_dir(config_path: str) -> str:
    """从配置文件中读取 base output 目录。"""
    cfg = load_yaml_config(config_path)
    return cfg.get("common", {}).get("cache_dir", "output/society")


def run_phase(
    phase: dict,
    config_path: str,
    python: str,
    logger: logging.Logger,
    no_eval: bool = False,
) -> bool:
    """运行单个 Phase，返回是否成功。"""
    phase_id = phase["id"]
    phase_name = phase["name"]

    # 跳过评估
    if phase_id == 6 and no_eval:
        logger.info(f"[Phase {phase_id}] {phase_name} — 已跳过 (--no_eval)")
        return True

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"[Phase {phase_id}/6] {phase_name} — {phase['desc']}")
    logger.info(f"  脚本: {phase['script']}")
    logger.info(f"  配置: {config_path}")
    logger.info("=" * 70)

    cmd = [python, phase["script"], "--config", config_path]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=False,  # 直接输出到终端
            text=True,
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            logger.error(f"[Phase {phase_id}] {phase_name} 失败 (exit code {result.returncode}), 耗时 {elapsed:.1f}s")
            return False

        # 验证产出文件
        for out_path in phase["outputs"]:
            if not os.path.exists(out_path):
                logger.error(f"[Phase {phase_id}] 产出文件缺失: {out_path}")
                return False

        logger.info(f"[Phase {phase_id}] {phase_name} 完成 ✓  耗时 {elapsed:.1f}s")
        return True

    except Exception as e:
        logger.error(f"[Phase {phase_id}] {phase_name} 异常: {e}")
        traceback.print_exc()
        return False


def print_final_summary(
    results: dict,
    total_time: float,
    logger: logging.Logger,
):
    """打印最终汇总。"""
    logger.info("")
    logger.info("=" * 70)
    logger.info("全流程汇总")
    logger.info("=" * 70)

    success = 0
    for phase_id, info in results.items():
        status = "✓ 成功" if info["ok"] else "✗ 失败"
        logger.info(f"  Phase {phase_id} [{info['name']:>20s}] {status}  {info['elapsed']:.1f}s")
        if info["ok"]:
            success += 1

    logger.info(f"\n  成功: {success}/6  总耗时: {total_time:.1f}s ({total_time / 60:.1f}min)")

    # 如果评估完成，打印结果
    eval_file = "output/society/eval/results.json"
    if os.path.exists(eval_file):
        with open(eval_file) as f:
            eval_data = json.load(f)

        logger.info("")
        logger.info("-" * 70)
        logger.info("消融实验结果")
        logger.info("-" * 70)
        logger.info(f"  {'配置':<30s} {'初始准确率':>10s} {'最终准确率':>10s} {'绝对提升':>10s}")
        logger.info(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

        for name, r in eval_data.get("ablation_results", {}).items():
            logger.info(
                f"  {name:<30s} {r['initial_accuracy']:>10.4f} {r['final_accuracy']:>10.4f} "
                f"{r['absolute_improvement']:>+10.4f}"
            )

    logger.info("")
    logger.info("=" * 70)
    logger.info("完成!")
    logger.info("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Diverse Actor-Critic Society 全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="YAML 配置文件路径 (如 configs/society/experiment_v100.yaml)",
    )
    parser.add_argument(
        "--start_phase", type=int, default=1, choices=range(1, 7),
        help="从第几个 Phase 开始 (默认 1)",
    )
    parser.add_argument(
        "--end_phase", type=int, default=6, choices=range(1, 7),
        help="运行到第几个 Phase 结束 (默认 6)",
    )
    parser.add_argument(
        "--no_eval", action="store_true",
        help="跳过 Phase 6 评估",
    )
    args = parser.parse_args()

    # 解析 output 目录
    base_dir = resolve_base_dir(args.config)
    log_dir = os.path.join(base_dir, "logs")
    logger = setup_pipeline_logging(log_dir)

    python = get_python_path()

    logger.info("=" * 70)
    logger.info("Diverse Actor-Critic Society 全流程")
    logger.info(f"  配置:    {args.config}")
    logger.info(f"  Python:  {python}")
    logger.info(f"  输出目录: {base_dir}")
    logger.info(f"  运行范围: Phase {args.start_phase} -> Phase {args.end_phase}")
    logger.info(f"  跳过评估: {args.no_eval}")
    logger.info(f"  日志文件: {log_dir}/pipeline.log")
    logger.info("=" * 70)

    # 保存运行配置
    run_meta = {
        "config": args.config,
        "start_phase": args.start_phase,
        "end_phase": args.end_phase,
        "no_eval": args.no_eval,
        "python": python,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_file = os.path.join(log_dir, "run_meta.json")
    with open(meta_file, "w") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    # 顺序执行各 Phase
    total_start = time.time()
    results = {}

    for phase in PHASES:
        pid = phase["id"]

        # 跳过范围外的 Phase
        if pid < args.start_phase or pid > args.end_phase:
            continue

        ok = run_phase(phase, args.config, python, logger, no_eval=args.no_eval)
        results[pid] = {
            "name": phase["name"],
            "ok": ok,
            "elapsed": time.time() - total_start - sum(r["elapsed"] for r in results.values()),
        }

        if not ok:
            logger.error(f"\nPhase {pid} ({phase['name']}) 失败，流程中止。")
            logger.error(f"修复后可使用 --start_phase {pid} 从此 Phase 恢复。")
            break

        # Phase 间释放 GPU
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    total_time = time.time() - total_start

    # 打印汇总
    print_final_summary(results, total_time, logger)

    # 保存运行结果摘要
    summary_file = os.path.join(log_dir, "run_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "total_time_seconds": total_time,
            "phases": {str(k): v for k, v in results.items()},
            "run_meta": run_meta,
        }, f, indent=2, ensure_ascii=False)

    # 返回码
    all_ok = all(r["ok"] for r in results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
