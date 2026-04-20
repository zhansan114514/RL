# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

ACC-Collab 复现项目 -- 实现 ICLR 2025 论文《ACC-Collab: An Actor-Critic Approach to Multi-Agent LLM Collaboration》。核心思想：联合训练 Actor 和 Critic 两个 LLM 智能体，通过迭代审议（Deliberation）协作解决推理任务，使用 Guided Collaborative Trajectories 生成偏好数据，再用 DPO 进行优化。

**扩展实验**: 在 ACC-Collab 基础上结合 Multiagent FT 论文的多样化思维链思想，构建 "Diverse Actor-Critic Society"——3 个推理风格各异的 Actor + 4 个错误类型专长的 Critic + MoE 置信度路由，训练出既保持多样化思维链、又具备精细协作能力的社会。详见 `glimmering-greeting-yeti.md` 和 `finish.md`。

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
- **交替训练** = `src.training.scheduler.alternating_train`，先固定 Actor 训练 Critic，再固定 Critic 训练 Actor（1 轮 = ACC-Collab，2 轮 = ACC-Collab+）

## 模块职责与依赖关系

| 模块 | 职责 | 关键文件 | 核心依赖 |
|------|------|----------|----------|
| `src.data` | 数据加载与预处理，统一 7 个基准的格式 | `loader.py`, `preprocessor.py` | datasets (HuggingFace) |
| `src.prompts` | 6 类 Prompt 模板（single_shot / guided / deliberation_actor / deliberation_critic 及其 guided 变体），按数据集分派 | `templates.py`, `formatter.py` | 无外部依赖 |
| `src.algorithms` | 审议引擎、奖励计算、MC roll-out、Algorithm 1 轨迹生成 | `deliberation.py`, `reward.py`, `rollout.py`, `trajectory.py` | prompts, scipy, numpy |
| `src.trajectory` | 偏好数据集构建 | `preference.py` | algorithms |
| `src.training` | DPO 训练(trl)、LoRA 配置(peft)、交替训练调度 | `dpo_trainer.py`, `_dpo_runner.py`, `trainer.py`, `scheduler.py`, `model_manager.py`, `lora_config.py` | trl, peft, transformers |
| `src.inference` | vLLM 推理封装（懒加载） | `vllm_server.py` | vllm |
| `src.evaluation` | 基准评估：逐轮准确率、改进率 | `benchmarks.py` | deliberation, reward |
| `src.utils` | OmegaConf 配置合并、日志、WandB 集成、模型检测、随机种子 | `config.py`, `logging_utils.py`, `model_utils.py`, `seeding.py` | omegaconf, wandb |
| `src.society` | Diverse Actor-Critic Society：多Agent注册、MoE路由、API分类、数据分化、多Agent审议、交替训练、推理投票 | `agent_registry.py`, `router.py`, `data_classifier.py`, `multi_deliberation.py`, `diversity_split.py`, `society_trainer.py`, `inference_pipeline.py` | algorithms, anthropic (API) |

**注意**：`src.data.preprocessor` 和 `src.algorithms.reward` 都有各自的 `extract_answer` 实现，前者用于数据预处理阶段，后者用于奖励计算和评估阶段。两套实现的逻辑相似但返回格式不同（前者返回小写 "yes"/"no"，后者返回大写 "YES"/"NO"）。

### 训练架构（重构后）

训练系统已重构为分层模块，`alternating.py` 保留为兼容层：

| 文件 | 职责 |
|------|------|
| `model_manager.py` | VLLM 模型生命周期管理：创建/清理推理模型，消除重复的模型实例化代码 |
| `trainer.py` | 单 Agent 训练：轨迹生成 + DPO 训练，支持批量处理和 JSONL 轨迹缓存 |
| `scheduler.py` | 交替训练编排：迭代 best-response 优化，支持验证早停和轨迹复用 |
| `dpo_trainer.py` | DPO 训练底层实现，通过子进程调用 `_dpo_runner.py` |
| `_dpo_runner.py` | 隔离子进程 DPO 训练：在独立 CUDA 上下文中运行，确保 GPU 设备分配正确 |
| `lora_config.py` | LoRA 配置生成 |

## 配置系统

使用 OmegaConf 分层合并（不使用 Hydra CLI 入口），覆盖优先级从低到高：

```
configs/default.yaml       -- 全局默认（模型、推理、审议、训练、调度器参数）
configs/base.yaml          -- 基础覆盖（seed=42, 审议5轮, LoRA r=256, lr=5e-5）
configs/model/*.yaml       -- 模型配置（gemma2_2b / llama3_8b / mistral_7b / qwen2.5_7b）
configs/data/*.yaml        -- 数据集配置（boolq / mmlu / bbh / sciq / arc / math / gsm）
configs/train/*.yaml       -- 训练配置（dpo_actor / dpo_critic，结构相同仅 agent 字段不同）
configs/society/*.yaml     -- Society 模块配置（base / actors / critics / router / experiment_h100）
```

通过 `src.utils.config.load_config(model=..., dataset=..., train=...)` 加载并合并。

新增实验配置：
- `configs/verify.yaml` -- 3 样本验证（boolq, 1 iteration, 2 rounds）
- `configs/debug.yaml` / `configs/debug_3samples.yaml` -- 调试用配置
- `configs/experiment_gpu1.yaml` -- 单 H100 GPU 实验配置（含 cache_dir）
- `configs/experiment_qwen3_arc.yaml` -- Qwen3-4B + ARC-Challenge 实验

新增配置参数：
- `cache_dir`：实验级缓存目录
- `reuse_trajectories`：轨迹复用开关
- 各步骤独立的 GPU 设备分配（`gpu_device`）和推理参数（`dtype`, `gpu_memory_utilization`）

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

# === 原始 ACC-Collab 实验 ===

# 一键全流程
python scripts/06_full_pipeline.py --model_name google/gemma-2-2b-it --dataset boolq --output_dir experiments/gemma2_boolq --max_samples 100

# 统一训练入口（合并了原 03_train_critic 和 04_train_actor）
python scripts/train.py --config configs/verify.yaml --agent critic
python scripts/train.py --config configs/verify.yaml --agent actor

# 分步执行：01_生成轨迹 -> 02_构建偏好 -> train.py --agent critic -> train.py --agent actor -> 05_评估
```

Makefile 快捷方式：`make test` / `make lint` / `make clean`

### Diverse Actor-Critic Society 实验（07-12 脚本）

结合 ACC-Collab + Multiagent FT 的扩展实验。基座模型：Qwen2.5-7B-Instruct，单张 H100 80GB。

```bash
export PYTHONPATH=$(pwd)

# === 小规模验证（5样本，2轮审议）===

# Phase 1: Bootstrap 多样化数据
python scripts/07_bootstrap_actors.py --config configs/society/experiment_h100.yaml --max_samples 5

# Phase 2: 分类推理风格 + 错误类型（GLM4.5 API）
python scripts/08_classify_data.py --config configs/society/experiment_h100.yaml

# Phase 3: Actor 分化训练（3个 Actor 各自 DPO）
python scripts/09_diversify_actors.py --config configs/society/experiment_h100.yaml

# Phase 4: Critic 分化训练（4个 Critic 各自 DPO）
python scripts/10_diversify_critics.py --config configs/society/experiment_h100.yaml

# Phase 5: Society 交替训练（N×M 交替 DPO）
python scripts/11_society_train.py --config configs/society/experiment_h100.yaml --max_samples 5 --num_rounds 2

# Phase 6: 评估 + 消融实验（A1-A5）
python scripts/12_society_evaluate.py --config configs/society/experiment_h100.yaml --max_samples 5

# === 全量运行（200样本，5轮，预计17-22小时）===
python scripts/07_bootstrap_actors.py --config configs/society/experiment_h100.yaml
python scripts/08_classify_data.py --config configs/society/experiment_h100.yaml
python scripts/09_diversify_actors.py --config configs/society/experiment_h100.yaml
python scripts/10_diversify_critics.py --config configs/society/experiment_h100.yaml
python scripts/11_society_train.py --config configs/society/experiment_h100.yaml
python scripts/12_society_evaluate.py --config configs/society/experiment_h100.yaml
```

消融实验配置（A1-A5 自动运行）：
- **A1**: 1 Actor + 1 Critic（原始 ACC-Collab 基线）
- **A2**: 3 Actor + 1 Critic（Actor 多样性单独贡献）
- **A3**: 1 Actor + 4 Critic + Router（Critic 专长化单独贡献）
- **A4**: 3 Actor + 4 Critic，均匀权重（完整系统但无路由）
- **A5**: 3 Actor + 4 Critic + Router（完整系统）

预期：A5 > A1（完整系统优于单 Agent 基线）

API 分类需要设置环境变量：`export GLM_API_KEY=your_key`（或在 data_classifier.py 中配置默认值）。

## 管线脚本编号约定

脚本按执行顺序编号（01-06 为原始 ACC-Collab，07-12 为 Diverse Society 扩展），06 是一键全流程脚本。每个脚本通过 `sys.path.insert(0, ...)` 保证从项目根目录导入 `src`。分步脚本之间通过 JSON 文件和 HuggingFace Dataset 磁盘格式传递中间数据。

新增公共模块：
- `scripts/_utils.py` -- 日志设置、YAML 配置加载、配置合并等共享工具函数
- `scripts/train.py` -- 统一 DPO 训练入口，通过 `--agent` 区分 actor/critic

## Prompt 模板系统

6 种 PromptType 枚举值对应论文 Appendix C 的 6 类模板：
1. `SINGLE_SHOT` -- Actor 初始回答
2. `GUIDED_SINGLE_SHOT` -- Actor 引导向目标答案
3. `DELIBERATION_ACTOR` -- Actor 审议轮次回答
4. `GUIDED_DELIBERATION_ACTOR` -- Actor 引导审议
5. `DELIBERATION_CRITIC` -- Critic 反馈
6. `GUIDED_DELIBERATION_CRITIC` -- Critic 引导反馈

BoolQ 使用独立的 yes_no 模板；MMLU/BBH/SCIQ/ARC 共享多选模板（`BBH_TEMPLATES = MMLU_TEMPLATES`）。MATH 和 GSM 使用独立的数学模板，要求答案以 `\boxed{}` 格式输出。

## 测试策略

10 个测试文件覆盖所有核心模块，使用 `unittest.mock` 隔离外部依赖（HuggingFace 数据加载、vLLM 推理、GLM API 调用）。关键测试模式：
- `test_data.py` / `test_evaluation.py` 中的 `extract_answer` 参数化测试验证 zeta 函数对各种响应格式的鲁棒性
- `test_evaluation.py` 中的 `TestEvaluateBenchmark` 使用 mock 验证端到端评估流程
- `test_society.py` 中的 85 个测试覆盖 Society 模块全部组件（AgentRegistry、CriticRouter、DataClassifier、DiversitySplit、InferencePipeline、数学答案提取、消融配置）
- `test_training.py` 覆盖重构后的训练架构（trainer、scheduler、model_manager）
- `test_qwen_gsm.py` 测试 Qwen2.5 LoRA 配置和 MATH/GSM 答案提取
- `test_fixes_verification.py` 验证各类修复的正确性

## LoRA 目标模块

三种架构（llama3 / mistral / gemma2 / qwen2.5 / qwen3）共享相同的目标模块列表：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`。默认 rank=256, alpha=512。

## 模型类型检测

`detect_model_type()` 函数（`src/utils/model_utils.py`）通过模型名称字符串匹配判断架构类型：`"llama"` -> llama3，`"mistral"` -> mistral，`"gemma"` -> gemma2，`"qwen"` + `"2.5"` -> qwen2.5，`"qwen"` -> qwen3（默认）。Qwen2.5 检测必须在通用 qwen 之前。

## 硬件需求

- Gemma-2-2B：1x V100 (32GB) 可跑全流程
- Llama-3-8B 推理：1x V100/A100；训练需 1x A100/H800 (80GB)
- Qwen2.5-7B + Diverse Society：1x H100 (80GB)，全流程 200 样本约 17-22 小时
- Qwen3-4B + ARC：可使用 `configs/experiment_qwen3_arc.yaml` 配置

## 项目忽略规则

`experiments/logs/`, `wandb/`, `outputs/`, `checkpoints/`, `models/`, `data/cache/` 均被 `.gitignore` 排除，这些都是运行时产物目录。
