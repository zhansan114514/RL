# ACC-Collab 复现项目

**论文**: *ACC-Collab: An Actor-Critic Approach to Multi-Agent LLM Collaboration* (ICLR 2025)
**作者**: Estornell et al. (ByteDance Research)

联合训练 Actor-Critic 双 LLM 智能体团队，通过迭代审议协作解决推理任务。使用 Guided Collaborative Trajectories 生成偏好数据，DPO 进行优化。

---

## 项目结构

```
ACC-Collab/
├── configs/                        # 配置文件
│   ├── base.yaml                   # 全局默认配置
│   ├── model/                      # 模型配置 (gemma2_2b / llama3_8b / mistral_7b)
│   ├── data/                       # 数据集配置 (boolq / mmlu / bbh / sciq / arc)
│   └── train/                      # 训练配置 (dpo_actor / dpo_critic)
├── src/
│   ├── data/                       # 数据加载与预处理
│   │   ├── loader.py               # HuggingFace 数据集加载器 (5 个基准)
│   │   ├── preprocessor.py         # 格式化、答案提取、错误答案生成
│   │   └── utils.py                # 数据统计工具
│   ├── prompts/                    # Prompt 模板管理
│   │   ├── templates.py            # 6 类 prompt 模板 (论文 Section C)
│   │   └── formatter.py            # 变量注入与格式化
│   ├── deliberation/               # 审议引擎
│   │   ├── engine.py               # 自然审议循环 + 引导审议
│   │   └── rollouts.py             # One-step Monte Carlo roll-out 奖励估计
│   ├── reward/                     # 奖励计算
│   │   ├── accuracy.py             # 答案提取 (zeta 函数) + 准确率 + 置信区间
│   │   └── partial.py              # 轨迹奖励差值计算 + 偏好对筛选
│   ├── trajectory/                 # 轨迹生成与偏好数据构建
│   │   ├── generator.py            # Algorithm 1 完整实现
│   │   └── preference.py           # 偏好数据集构建与格式转换
│   ├── training/                   # DPO 训练管线
│   │   ├── dpo_trainer.py          # DPO 训练 (基于 trl)
│   │   ├── lora_config.py          # LoRA 配置 (rank=256)
│   │   └── alternating.py          # 交替训练调度 (actor/critic)
│   ├── inference/                  # 推理服务
│   │   └── vllm_server.py          # vLLM 推理封装
│   ├── evaluation/                 # 评估模块
│   │   └── benchmarks.py           # 基准测试运行器
│   └── utils/                      # 工具模块
│       ├── config.py               # OmegaConf 配置管理
│       └── logging_utils.py        # 日志 + WandB 实验追踪
├── tests/                          # 测试套件 (36 用例)
├── scripts/                        # 运行脚本 (分步 + 全流程)
├── pyproject.toml                  # 项目配置
└── Makefile                        # 常用命令快捷入口
```

---

## 环境搭建

### 1. 创建 conda 环境

```bash
conda create -n acc-collab python=3.10 -y
conda activate acc-collab
```

### 2. 安装项目

```bash
cd ACC-Collab
pip install -e ".[dev]"
```

核心依赖 (已在 `pyproject.toml` 中声明):

| 包 | 版本 | 用途 |
|---|---|---|
| torch | >= 2.1.0 | 深度学习框架 |
| transformers | >= 4.40.0 | 模型加载 |
| vllm | >= 0.4.0 | 高效推理 |
| trl | >= 0.9.0 | DPO 训练 |
| peft | >= 0.11.0 | LoRA 微调 |
| datasets | >= 2.19.0 | 数据加载 |
| scipy | >= 1.13.0 | 统计计算 |
| omegaconf | >= 2.3.0 | 配置管理 |
| wandb | >= 0.17.0 | 实验追踪 |

### 3. 硬件需求

| 阶段 | GPU 需求 | 说明 |
|------|----------|------|
| Gemma-2-2B 全流程 | 1x V100 (32GB) | 推理+训练均可用 |
| Llama-3-8B 推理 | 1x V100/A100 | vLLM 批量推理 |
| Llama-3-8B DPO 训练 | 1x A100/H800 (80GB) | LoRA rank=256 |

---

## 运行方式

### 方式 A: 全流程一键运行 (推荐使用 config 文件)

```bash
python scripts/06_full_pipeline.py \
    --config configs/config.yaml
```

说明:

```
scripts/01-06 现在都只接收一个参数: --config
所有运行参数统一写在 configs/config.yaml 中
```

configs/config.yaml 主要分组:

```
common: 全局通用参数 (model_name / dataset / seed / max_samples / use_wandb)
step01: 生成轨迹参数
step02: 偏好数据构建参数
step03: Critic 训练参数
step04: Actor 训练参数
step05: 评估参数
step06: 一键全流程参数
```

### 方式 B: 分步运行

```bash
# 步骤 1: 生成轨迹
python scripts/01_generate_trajectories.py \
    --config configs/config.yaml

# 步骤 2: 构建偏好数据集
python scripts/02_build_preferences.py \
    --config configs/config.yaml

# 步骤 3: 训练 Critic
python scripts/03_train_critic.py \
    --config configs/config.yaml

# 步骤 4: 训练 Actor
python scripts/04_train_actor.py \
    --config configs/config.yaml

# 步骤 5: 评估
python scripts/05_evaluate.py \
    --config configs/config.yaml
```

---

## 测试

### 运行全部测试

```bash
# 确保 PYTHONPATH 设置正确, 且在 acc-collab conda 环境中
export PYTHONPATH=$(pwd)
pytest tests/ -v
```

### 运行指定模块测试

```bash
# 数据模块
pytest tests/test_data.py -v

# Prompt 模板
pytest tests/test_prompts.py -v

# 审议引擎
pytest tests/test_deliberation.py -v

# 奖励与评估
pytest tests/test_evaluation.py -v

# 训练配置
pytest tests/test_training.py -v

# 轨迹与偏好数据
pytest tests/test_trajectory.py -v
```

### 测试覆盖范围

| 测试文件 | 用例数 | 覆盖内容 |
|---------|-------|---------|
| test_data.py | 14 | 答案提取 (Yes/No + MC)、归一化、数据加载 mock、格式验证 |
| test_prompts.py | 12 | 模板检索、6 类模板完整性、格式化渲染、变量注入 |
| test_evaluation.py | 36 | 答案提取、准确率、置信区间、逐轮准确率、改进率、evaluate_benchmark 集成 |
| test_deliberation.py | 5 | 自然审议轮次、引导审议 actor/critic、single-shot prompt |
| test_trajectory.py | 4 | 偏好对构建 (actor/critic)、min_delta 过滤、空数据 |
| test_training.py | 4 | LoRA 配置 (3 种架构)、alternating_train 签名验证 |

---

## 核心算法说明

### 算法流程 (Algorithm 1)

```
对每个训练样本 (x, y):
  1. 自然审议: actor + critic 迭代 T 轮, 得到轨迹 [(z_a^t, z_c^t)]
  2. 引导向正确答案 y: 得到 z_y
  3. 引导向错误答案 !y: 得到 z_not_y
  4. Monte Carlo roll-out 估计奖励:
     - v_natural = estimate_final_accuracy(z^t)
     - v_y = estimate_final_accuracy(z_y)
     - v_not_y = estimate_final_accuracy(z_not_y)
  5. 计算差值:
     - delta_y = v_y - v_natural
     - delta_not_y = v_natural - v_not_y
  6. 构建偏好对 (delta >= epsilon):
     - delta_y >= ε: chosen=z_y, rejected=z^t
     - delta_not_y >= ε: chosen=z^t, rejected=z_not_y
```

### 交替训练

```
ACC-Collab (1 轮):
  1. 固定 actor → 生成数据 → DPO 训练 critic
  2. 固定 critic → 生成数据 → DPO 训练 actor

ACC-Collab+ (2 轮):
  重复上述过程 2 次
```

---

## 复现目标

论文报告的 Llama-3 ACC-Collab 结果:

| 数据集 | 论文值 | 复现目标 (±2%) |
|--------|--------|---------------|
| BoolQ | .887±.005 | >= .869 |
| MMLU | .644±.01 | >= .631 |
| BBH | .593±.006 | >= .581 |
| SCIQ | .952±.0 | >= .933 |
| ARC | .881±.004 | >= .863 |

**推荐复现顺序**: Gemma-2-2B (快速验证) → Llama-3-8B (主要结果) → Mistral-7B (补充验证)

---

## Makefile 快捷命令

```bash
make setup     # 创建 conda 环境
make install   # 安装项目依赖
make test      # 运行全部测试
make lint      # 代码检查 (ruff)
make clean     # 清理缓存
```
