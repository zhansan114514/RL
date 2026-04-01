# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ACC-Collab 复现项目 -- 实现 ICLR 2025 论文《ACC-Collab: An Actor-Critic Approach to Multi-Agent LLM Collaboration》。核心思想：联合训练 Actor 和 Critic 两个 LLM 智能体，通过迭代审议（Deliberation）协作解决推理任务，使用 Guided Collaborative Trajectories 生成偏好数据，再用 DPO 进行优化。

## 架构与数据流

整个管线是一条单向数据流，理解这条链路就理解了项目：

```
数据集加载 (src.data)
  -> 自然审议 + 引导审议 (src.algorithms.deliberation)
  -> MC Roll-out 奖励估计 (src.algorithms.rollout)
  -> 偏好对构建 (src.trajectory)
  -> DPO 训练 (src.training)
  -> 评估 (src.evaluation)
```

关键概念映射：
- **zeta 函数** = `src.algorithms.reward.extract_answer`，从 LLM 文本中抽取结构化答案
- **Algorithm 1** = `src.algorithms.trajectory.generate_trajectories`，对每个样本生成自然轨迹、引导向正确答案的轨迹、引导向错误答案的轨迹，然后计算奖励差值筛选偏好对
- **交替训练** = `src.training.alternating.alternating_train`，先固定 Actor 训练 Critic，再固定 Critic 训练 Actor（1 轮 = ACC-Collab，2 轮 = ACC-Collab+）

## 模块职责与依赖关系

| 模块 | 职责 | 关键文件 | 核心依赖 |
|------|------|----------|----------|
| `src.data` | 数据加载与预处理，统一 5 个基准的格式 | `loader.py`, `preprocessor.py` | datasets (HuggingFace) |
| `src.prompts` | 6 类 Prompt 模板（single_shot / guided / deliberation_actor / deliberation_critic 及其 guided 变体），按数据集分派 | `templates.py`, `formatter.py` | 无外部依赖 |
| `src.algorithms` | 审议引擎、奖励计算、MC roll-out、Algorithm 1 轨迹生成 | `deliberation.py`, `reward.py`, `rollout.py`, `trajectory.py` | prompts, scipy, numpy |
| `src.trajectory` | 偏好数据集构建 | `preference.py` | algorithms |
| `src.training` | DPO 训练(trl)、LoRA 配置(peft)、交替训练调度 | `dpo_trainer.py`, `lora_config.py`, `alternating.py` | trl, peft, transformers |
| `src.inference` | vLLM 推理封装（懒加载） | `vllm_server.py` | vllm |
| `src.evaluation` | 基准评估：逐轮准确率、改进率 | `benchmarks.py` | deliberation, reward |
| `src.utils` | OmegaConf 配置合并、日志、WandB 集成 | `config.py`, `logging_utils.py` | omegaconf, wandb |

**注意**：`src.data.preprocessor` 和 `src.algorithms.reward` 都有各自的 `extract_answer` 实现，前者用于数据预处理阶段，后者用于奖励计算和评估阶段。两套实现的逻辑相似但返回格式不同（前者返回小写 "yes"/"no"，后者返回大写 "YES"/"NO"）。

## 配置系统

使用 OmegaConf 分层合并（不使用 Hydra CLI 入口）：

```
configs/base.yaml          -- 全局默认（seed=42, 审议5轮, LoRA r=256, lr=5e-5）
configs/model/*.yaml       -- 模型配置（gemma2_2b / llama3_8b / mistral_7b）
configs/data/*.yaml        -- 数据集配置（boolq / mmlu / bbh / sciq / arc）
configs/train/*.yaml       -- 训练配置（dpo_actor / dpo_critic，结构相同仅 agent 字段不同）
```

通过 `src.utils.config.load_config(model=..., dataset=..., train=...)` 加载并合并。

## 运行命令

```bash
# 环境搭建
conda create -n acc-collab python=3.10 -y && conda activate acc-collab
pip install -e ".[dev]"

# 运行全部测试
export PYTHONPATH=$(pwd)
pytest tests/ -v

# 代码检查
ruff check src/ tests/

# 一键全流程
python scripts/06_full_pipeline.py --model_name google/gemma-2-2b-it --dataset boolq --output_dir experiments/gemma2_boolq --max_samples 100

# 分步执行：01_生成轨迹 -> 02_构建偏好 -> 03_训练Critic -> 04_训练Actor -> 05_评估
```

Makefile 快捷方式：`make test` / `make lint` / `make clean`

## 管线脚本编号约定

脚本按执行顺序编号（01-06），06 是一键全流程脚本。每个脚本通过 `sys.path.insert(0, ...)` 保证从项目根目录导入 `src`。分步脚本之间通过 JSON 文件和 HuggingFace Dataset 磁盘格式传递中间数据。

## Prompt 模板系统

6 种 PromptType 枚举值对应论文 Appendix C 的 6 类模板：
1. `SINGLE_SHOT` -- Actor 初始回答
2. `GUIDED_SINGLE_SHOT` -- Actor 引导向目标答案
3. `DELIBERATION_ACTOR` -- Actor 审议轮次回答
4. `GUIDED_DELIBERATION_ACTOR` -- Actor 引导审议
5. `DELIBERATION_CRITIC` -- Critic 反馈
6. `GUIDED_DELIBERATION_CRITIC` -- Critic 引导反馈

BoolQ 使用独立的 yes_no 模板；MMLU/BBH/SCIQ/ARC 共享多选模板（`BBH_TEMPLATES = MMLU_TEMPLATES`）。

## 测试策略

6 个测试文件覆盖所有核心模块，使用 `unittest.mock` 隔离外部依赖（HuggingFace 数据加载、vLLM 推理）。关键测试模式：
- `test_data.py` / `test_evaluation.py` 中的 `extract_answer` 参数化测试验证 zeta 函数对各种响应格式的鲁棒性
- `test_evaluation.py` 中的 `TestEvaluateBenchmark` 使用 mock 验证端到端评估流程

## LoRA 目标模块

三种架构（llama3 / mistral / gemma2）共享相同的目标模块列表：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`。默认 rank=256, alpha=512。

## 模型类型检测

`_detect_model_type()` 函数在多个脚本中重复出现（`06_full_pipeline.py`、`03_train_critic.py`、`04_train_actor.py`），通过模型名称字符串匹配判断架构类型。名称含 "llama" -> llama3，"mistral" -> mistral，"gemma" -> gemma2。

## 硬件需求

- Gemma-2-2B：1x V100 (32GB) 可跑全流程
- Llama-3-8B 推理：1x V100/A100；训练需 1x A100/H800 (80GB)

## 项目忽略规则

`experiments/logs/`, `wandb/`, `outputs/`, `checkpoints/`, `models/`, `data/cache/` 均被 `.gitignore` 排除，这些都是运行时产物目录。
