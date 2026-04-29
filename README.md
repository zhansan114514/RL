# ACC-Collab 复现项目

**论文**: *ACC-Collab: An Actor-Critic Approach to Multi-Agent LLM Collaboration* (ICLR 2025)

联合训练 Actor-Critic 双 LLM 智能体，通过迭代审议协作解决推理任务。使用 Guided Collaborative Trajectories 生成偏好数据，DPO 进行优化。

---

## 项目结构

```
src/
├── algorithms/          # 算法核心层（论文算法的权威实现）
│   ├── reward.py        # extract_answer (zeta 函数) + accuracy + reward_delta
│   ├── deliberation.py  # 自然审议 + 引导审议循环
│   └── trajectory.py    # Algorithm 1 轨迹生成与偏好对构建
├── data/                # 数据加载与预处理
│   ├── loader.py        # HuggingFace 数据集加载（BoolQ/MMLU/BBH/SCIQ/ARC）
│   └── preprocessor.py  # standardize_sample, generate_wrong_answer
├── prompts/             # Prompt 模板系统
│   ├── templates.py     # 6 类 PromptType 枚举与模板（论文 Appendix C）
│   └── formatter.py     # format_prompt 变量注入
├── training/            # 训练管线
│   ├── scheduler.py     # alternating_train 交替训练调度
│   ├── trainer.py       # train_agent 单 agent 轨迹生成 + DPO
│   ├── dpo_trainer.py   # train_dpo (基于 trl)
│   ├── model_manager.py # create_inference_model, create_model_pair, cleanup_models
│   └── lora_config.py   # get_lora_config (rank=256, alpha=512)
├── evaluation/          # 评估
│   └── benchmarks.py    # evaluate_benchmark 逐轮准确率 + Wilson 置信区间
├── inference/           # 推理服务
│   └── vllm_server.py   # VLLMInference (vLLM 封装)
├── trajectory/          # 偏好数据构建
│   └── preference.py    # build_preference_dataset, convert_to_hf_dataset
└── utils/               # 工具
    ├── config.py        # ConfigManager 单例 + configs/default.yaml
    ├── seeding.py       # fix_seed (random/numpy/torch)
    ├── logging_utils.py # ExperimentLogger + JSONL 持久化 + WandB
    └── model_utils.py   # detect_model_type
```

---

## 环境搭建

```bash
conda create -n acc-collab python=3.10 -y && conda activate acc-collab
pip install -e ".[dev]"
```

| 包 | 用途 |
|---|---|
| torch >= 2.1.0 | 深度学习框架 |
| vllm >= 0.4.0 | 高效推理 |
| trl >= 0.9.0 | DPO 训练 |
| peft >= 0.11.0 | LoRA 微调 |
| omegaconf >= 2.3.0 | 配置管理 |

| 阶段 | GPU 需求 |
|------|----------|
| Gemma-2-2B 全流程 | 1x V100 (32GB) |
| Llama-3-8B 训练 | 1x A100/H800 (80GB) |

---

## 运行方式

### 一键全流程

```bash
python scripts/06_full_pipeline.py --config configs/config.yaml
```

### 分步运行

```bash
python scripts/01_generate_trajectories.py --config configs/config.yaml
python scripts/02_build_preferences.py --config configs/config.yaml
python scripts/train.py --config configs/config.yaml --agent critic
python scripts/train.py --config configs/config.yaml --agent actor
python scripts/05_evaluate.py --config configs/config.yaml
```

### 统一训练入口

```bash
python scripts/train.py --config configs/config.yaml --agent {actor,critic}
```

---

## 配置系统

使用 ConfigManager 单例，启动时加载 `configs/default.yaml`，所有参数集中管理：

```yaml
# configs/default.yaml
inference:
  max_model_len: 4096
  gpu_memory_utilization: 0.45
deliberation:
  num_rounds: 5
  num_simulations: 5
training:
  lora_r: 256
  beta: 0.1
```

运行时通过 `ConfigManager.instance().get("inference.max_model_len")` 读取，无需硬编码。

---

## 测试

```bash
export PYTHONPATH=$(pwd)
pytest tests/ -v
make lint             # ruff check src/ scripts/ tests/
```

---

## 数据流

```
数据集加载 (data/loader)
  → 自然审议 + 引导审议 (algorithms/deliberation)
  → 轨迹生成与奖励差筛选 (algorithms/trajectory + algorithms/reward)
  → 偏好对构建 (trajectory/preference)
  → DPO 训练 (training/trainer + training/dpo_trainer)
  → 评估 (evaluation/benchmarks)
```

---

## 维护约定

- `scripts/train.py --agent {actor,critic}` 是唯一的单 Agent DPO 训练入口。
- `src.training.scheduler.alternating_train` 是 Actor-Critic 交替训练的公共 API。
- `src.algorithms.reward` 是答案抽取、归一化、准确率和奖励差的权威实现。
- 实验产物、模型权重、缓存和机器相关二进制补丁不进入源码树。

更详细的模块边界和长期维护规则见 `docs/architecture.md`。

---

## 复现目标

| 数据集 | 论文值 | 复现目标 (±2%) |
|--------|--------|---------------|
| BoolQ | .887±.005 | >= .869 |
| MMLU | .644±.01 | >= .631 |
| BBH | .593±.006 | >= .581 |
| SCIQ | .952±.0 | >= .933 |
| ARC | .881±.004 | >= .863 |
