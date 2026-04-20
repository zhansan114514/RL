# Diverse Actor-Critic Society 实验运行指南

## 概述

通过统一脚本 `scripts/run_society.py` 一键运行完整的 6 Phase 管线：

```
Phase 1: Bootstrap   → 多 Agent 轨迹生成
Phase 2: Classify    → 推理风格 + 错误类型分类 (GLM API)
Phase 3: Actor DPO   → 3 个 Actor LoRA 分化训练
Phase 4: Critic DPO  → 4 个 Critic LoRA 分化训练
Phase 5: Society     → N×M 交替训练
Phase 6: Evaluate    → A1-A5 消融实验
```

## 环境准备

```bash
# 激活 conda 环境
conda activate vllm

# 安装依赖（如未安装）
pip install trl peft omegaconf vllm

# 确认模型已缓存
ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/
```

## 配置文件

| 文件 | 适用环境 | GPU | dtype | 说明 |
|------|---------|-----|-------|------|
| `configs/society/experiment_v100.yaml` | 4x V100-32GB | device=4 | auto(float16) | 5 样本验证 |
| `configs/society/experiment_h100.yaml` | 1x H100-80GB | device=6 | bfloat16 | 完整实验 |

配置文件中的关键参数：

```yaml
common:
  model_name: "Qwen/Qwen2.5-7B-Instruct"   # 基座模型
  dataset: "math"                            # 数据集
  max_samples: 5                             # 样本数（全量运行设为 null 或删除）
  device: 4                                  # GPU 编号
  dtype: "auto"                              # V100 用 auto(float16), H100 用 bfloat16
  gpu_memory_utilization: 0.85               # 显存占用率
```

## 启动方式

### 1. 完整运行（一键全流程）

```bash
# V100 5 样本验证
CUDA_VISIBLE_DEVICES=4 python scripts/run_society.py \
    --config configs/society/experiment_v100.yaml

# H100 全量运行
CUDA_VISIBLE_DEVICES=6 python scripts/run_society.py \
    --config configs/society/experiment_h100.yaml
```

### 2. 指定 GPU

通过环境变量 `CUDA_VISIBLE_DEVICES` 控制：

```bash
# 使用 GPU 0
CUDA_VISIBLE_DEVICES=0 python scripts/run_society.py --config configs/society/experiment_v100.yaml

# 使用 GPU 4
CUDA_VISIBLE_DEVICES=4 python scripts/run_society.py --config configs/society/experiment_v100.yaml
```

> **注意**: 配置文件中的 `device` 参数也需要与 `CUDA_VISIBLE_DEVICES` 匹配。
> 当 `CUDA_VISIBLE_DEVICES=4` 时，程序看到的设备编号是 `0`，此时配置中 `device` 应设为 `0`。
> 或者不设 `CUDA_VISIBLE_DEVICES`，让配置中的 `device: 4` 直接生效。

### 3. 部分运行（指定 Phase 范围）

```bash
# 只运行 Phase 1-4（跳过训练和评估）
python scripts/run_society.py \
    --config configs/society/experiment_v100.yaml \
    --end_phase 4

# 从 Phase 3 恢复（前两步已完成）
python scripts/run_society.py \
    --config configs/society/experiment_v100.yaml \
    --start_phase 3
```

### 4. 跳过评估

```bash
python scripts/run_society.py \
    --config configs/society/experiment_v100.yaml \
    --no_eval
```

## 输出目录结构

所有中间数据和日志都在 `output/society/` 下：

```
output/society/
├── logs/
│   ├── pipeline.log          # 全流程日志
│   ├── run_meta.json         # 运行配置记录
│   └── run_summary.json      # 运行结果摘要
├── bootstrap/
│   └── trajectories.jsonl    # Phase 1: 启动轨迹
├── classified/
│   ├── classified_data.json  # Phase 2: 分类结果
│   └── splits.json           # 分类统计
├── actors/
│   ├── actor_algebraic/      # Phase 3: Actor LoRA
│   ├── actor_direct/
│   ├── actor_backtracking/
│   └── actor_registry.json
├── critics/
│   ├── critic_arithmetic/    # Phase 4: Critic LoRA
│   ├── critic_logic/
│   ├── critic_hallucination/
│   ├── critic_verification/
│   └── critic_registry.json
├── society/
│   ├── final_agent_registry.json  # Phase 5: 最终注册表
│   └── checkpoints/
└── eval/
    └── results.json          # Phase 6: 评估结果
```

## 预期结果

Phase 6 评估会自动运行 5 组消融实验：

| 配置 | 说明 |
|------|------|
| A1: 1 Actor + 1 Critic | 原始 ACC-Collab 基线 |
| A2: 3 Actors + 1 Critic | Actor 多样性贡献 |
| A3: 1 Actor + 4 Critics + Router | Critic 专长化贡献 |
| A4: 3 Actors + 4 Critics（无路由） | 完整系统但无路由 |
| A5: 3 Actors + 4 Critics + Router | **完整系统** |

结果示例（5 样本验证）：

```
配置                             初始准确率   最终准确率   绝对提升
A1_baseline                       0.8000      0.8000     +0.0000
A2_actor_diversity                0.8000      0.8000     +0.0000
A3_critic_specialization          0.8000      0.8000     +0.0000
A4_no_routing                     0.6000      0.6000     +0.0000
A5_full_system                    0.6000      0.8000     +0.2000
```

## 分步运行（调试用）

如果需要单独运行某个 Phase，可以直接调用对应脚本：

```bash
# Phase 1
python scripts/07_bootstrap_actors.py --config configs/society/experiment_v100.yaml

# Phase 2
python scripts/08_classify_data.py --config configs/society/experiment_v100.yaml

# Phase 3
python scripts/09_diversify_actors.py --config configs/society/experiment_v100.yaml

# Phase 4
python scripts/10_diversify_critics.py --config configs/society/experiment_v100.yaml

# Phase 5
python scripts/11_society_train.py --config configs/society/experiment_v100.yaml

# Phase 6
python scripts/12_society_evaluate.py --config configs/society/experiment_v100.yaml
```

## 常见问题

### Q: V100 上 dtype 该怎么设？
V100 不支持原生 BF16。配置中 `dtype: "auto"` 会自动回退到 float16。不要手动设为 `bfloat16`。

### Q: 如何清除之前的运行结果？
```bash
rm -rf output/society/bootstrap output/society/classified output/society/actors output/society/critics output/society/society output/society/eval output/society/logs
```

### Q: OOM（显存不足）怎么办？
1. 降低 `gpu_memory_utilization`（如 0.65）
2. 减小 `max_model_len`（如 2048）
3. 减小 `batch_size`（如 2）

### Q: Phase 2 分类 API 失败？
检查配置中的 `api_key` 和 `api_base` 是否有效。配置文件中已包含默认 key。如需使用自己的 key，修改配置文件的 `step02_classify` 部分。

### Q: 如何跑全量实验（200 样本）？
将配置文件中的 `max_samples: 5` 删除或设为 `null`，预计耗时 17-22 小时（H100）。
