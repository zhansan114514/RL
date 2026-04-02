# ACC-Collab 运行操作与实验配置指南

## 1. 环境准备

```bash
# 激活 conda 环境
conda activate acc-collab

# 设置 PYTHONPATH
export PYTHONPATH=$(pwd)

# NVML fix（PyTorch 2.10 + 旧驱动必需，只需执行一次）
bash scripts/setup_nvml_fix.sh

# 后续所有命令都需要 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH
```

## 2. 基本检查

```bash
# 运行全部测试（202 个）
pytest tests/ -v

# 代码检查
ruff check src/ tests/
```

## 3. 运行方式

### 方式 A：一键全流程（推荐）

```bash
python scripts/06_full_pipeline.py --config <配置文件>
```

### 方式 B：分步执行

```bash
# Step 1: 生成轨迹
python scripts/01_generate_trajectories.py --config configs/config.yaml

# Step 2: 构建偏好数据
python scripts/02_build_preferences.py --config configs/config.yaml

# Step 3: 训练 Critic（LoRA + DPO）
python scripts/03_train_critic.py --config configs/config.yaml

# Step 4: 训练 Actor（LoRA + DPO）
python scripts/04_train_actor.py --config configs/config.yaml

# Step 5: 评估
python scripts/05_evaluate.py --config configs/config.yaml
```

分步脚本之间通过 JSON 文件和 HuggingFace Dataset 磁盘格式传递中间数据。

## 4. GPU 选择

当前服务器 16x V100-SXM3-32GB。通过配置文件的 `actor_device` 和 `critic_device` 指定物理 GPU 编号：

- 两个模型（actor + critic）分别使用不同的 GPU
- DPO 训练使用子进程隔离 CUDA context，与 vLLM 推理分时复用同一 GPU
- 每个 Gemma-2-2B float32 实例约占 10 GiB 显存

## 5. 配置文件

配置文件位于 `configs/` 目录，使用 YAML 格式。`06_full_pipeline.py` 读取 `common` + `step06` 节。

### 已有配置文件

| 文件 | 用途 |
|------|------|
| `configs/verify.yaml` | 3 样本快速验证（已通过） |
| `configs/config.yaml` | 10 样本调试 |
| `configs/debug_3samples.yaml` | 3 样本调试 |
| `configs/default.yaml` | 全局默认值 |
| `configs/model/gemma2_2b.yaml` | Gemma-2-2B 模型配置 |
| `configs/data/boolq.yaml` | BoolQ 数据集配置 |

### 配置模板

```yaml
# common: 通用参数
common:
  model_name: "google/gemma-2-2b-it"   # 或本地路径
  dataset: "boolq"                       # boolq / mmlu / bbh / sciq / arc
  seed: 42
  max_samples: 100                       # null = 使用全部数据
  use_wandb: false

# step06: 全流程参数
step06:
  output_dir: "experiments/<实验名>"
  num_iterations: 1                      # 1=ACC-Collab, 2=ACC-Collab+
  num_rounds: 5                          # 审议轮数（论文默认 5）
  num_simulations: 5                     # MC roll-out 次数（论文默认 5）
  lora_r: 256                            # LoRA rank
  learning_rate: 5.0e-5
  batch_size: 4
  num_epochs: 1
  beta: 0.1                              # DPO beta
  reward_threshold: 0.0
  skip_training: false
  actor_device: 5                        # 物理GPU编号
  critic_device: 6                       # 物理GPU编号
  dtype: "float32"                       # V100 用 float32（Gemma 不支持 fp16）
  gpu_memory_utilization: 0.45           # vLLM 每实例显存占比
  max_model_len: 4096
```

## 6. 论文实验配置

### 6.1 论文 Table 2 复现：Gemma-2-2B + BoolQ

这是最基础的实验组合，1x V100-32GB 即可完成推理，2x V100-32GB 可完成全流程训练。

创建配置文件 `configs/experiments/gemma2_boolq_100.yaml`：

```yaml
common:
  model_name: "google/gemma-2-2b-it"
  dataset: "boolq"
  seed: 42
  max_samples: 100                       # 先用 100 样本验证，后续增加到全量
  use_wandb: false

step06:
  output_dir: "experiments/gemma2_boolq_100"
  num_iterations: 1                      # ACC-Collab（单轮交替训练）
  num_rounds: 5
  num_simulations: 5
  lora_r: 256
  lora_alpha: 512
  learning_rate: 5.0e-5
  batch_size: 4
  num_epochs: 1
  beta: 0.1
  reward_threshold: 0.0
  skip_training: false
  actor_device: 5
  critic_device: 6
  dtype: "float32"
  gpu_memory_utilization: 0.45
  max_model_len: 4096
```

运行：

```bash
python scripts/06_full_pipeline.py --config configs/experiments/gemma2_boolq_100.yaml
```

### 6.2 论文 Table 2 复现：全量 BoolQ（9427 样本）

将 `max_samples` 设为 `null`，预计耗时数小时至一天。

```yaml
common:
  model_name: "google/gemma-2-2b-it"
  dataset: "boolq"
  seed: 42
  max_samples: null                      # 全量 9427 样本
  use_wandb: true                        # 建议开启 WandB 跟踪

step06:
  output_dir: "experiments/gemma2_boolq_full"
  num_iterations: 1
  num_rounds: 5
  num_simulations: 5
  lora_r: 256
  lora_alpha: 512
  learning_rate: 5.0e-5
  batch_size: 4
  num_epochs: 1
  beta: 0.1
  reward_threshold: 0.0
  skip_training: false
  actor_device: 5
  critic_device: 6
  dtype: "float32"
  gpu_memory_utilization: 0.45
  max_model_len: 4096
```

### 6.3 ACC-Collab+（两轮交替训练）

将 `num_iterations` 改为 `2`，其余不变。两轮意味着：训练 Critic → 训练 Actor → 再训练 Critic → 再训练 Actor。

### 6.4 其他数据集

将 `dataset` 改为对应名称即可：

| 数据集 | HuggingFace ID | task_type | 说明 |
|--------|---------------|-----------|------|
| boolq | google/boolq | yes_no | 二元问答 |
| mmlu | cais/mmlu | multiple_choice | 多项选择（57 子集） |
| bbh | lukaemon/bbh | mixed | Big-Bench Hard |
| sciq | sciq | multiple_choice | 科学问答 |
| arc | allenai/ai2_arc | multiple_choice | ARC-Challenge |

### 6.5 更大模型（Llama-3-8B / Mistral-7B）

需要修改 `model_name` 和 `dtype`，且需要 A100/H800 (80GB) 进行训练。

```yaml
common:
  model_name: "meta-llama/Meta-Llama-3-8B-Instruct"
  # model_name: "mistralai/Mistral-7B-Instruct-v0.3"
```

## 7. 输出结构

运行后输出目录结构：

```
experiments/<实验名>/
├── critic_iter0/            # 合并后的 Critic 模型（vLLM 可直接加载）
├── critic_iter0_adapter/    # Critic LoRA 适配器
├── actor_iter0/             # 合并后的 Actor 模型
├── actor_iter0_adapter/     # Actor LoRA 适配器
└── results.json             # 评估结果
```

`results.json` 示例：

```json
{
  "dataset": "boolq",
  "num_samples": 3,
  "num_rounds": 2,
  "final_accuracy": 0.667,
  "ci_margin": 0.365,
  "initial_accuracy": 0.667,
  "per_round_accuracy": [0.667, 0.667],
  "improvement_rate": 0.0
}
```

## 8. 时间估算

基于 3 样本验证（Gemma-2-2B, BoolQ, V100-32GB, 2 轮审议）：

| 阶段 | 3 样本时间 | 100 样本估算 | 全量估算 |
|------|-----------|-------------|---------|
| 数据加载 | < 1s | < 1s | < 1s |
| 模型加载（vLLM x2） | ~30s | ~30s | ~30s |
| 轨迹生成（每样本） | ~30s | ~50min | ~80h |
| vLLM 清理 | ~2s | ~2s | ~2s |
| DPO 训练（子进程） | ~12s | ~5min | ~数小时 |
| 评估 | ~2min | ~30min | ~数小时 |
| **总计** | **~12min** | **~1.5h** | **~数天** |

## 9. 关键参数说明

| 参数 | 论文默认 | 说明 |
|------|---------|------|
| `num_iterations` | 1 | 交替训练轮数。1=ACC-Collab, 2=ACC-Collab+ |
| `num_rounds` | 5 | Actor-Critic 审议轮数 |
| `num_simulations` | 5 | MC roll-out 奖励估计的模拟次数 |
| `lora_r` | 256 | LoRA 秩 |
| `lora_alpha` | 512 | LoRA 缩放因子（论文 alpha/r = 2） |
| `beta` | 0.1 | DPO 偏好强度参数 |
| `learning_rate` | 5e-5 | 学习率 |
| `batch_size` | 4 | 训练批次大小 |
| `reward_threshold` | 0.0 | 偏好对的最小奖励差值 |
| `gpu_memory_utilization` | 0.45 | vLLM 每实例 GPU 显存占比 |
| `dtype` | float32 | V100 必须用 float32（Gemma 不支持 fp16/bf16） |

## 10. 已知限制

1. **V100 不支持 bf16**：Gemma-2 模型在 V100 上只能用 float32，推理和训练速度较慢
2. **CUDA context 隔离**：DPO 训练通过子进程执行，以避免与 vLLM 的 CUDA context 冲突
3. **Python 3.13**：当前环境使用 Python 3.13，部分依赖（如 pyproject.toml 要求 >=3.10）版本偏高
4. **FA2 不支持**：V100 (compute capability 7.0) 不支持 FlashAttention-2，使用 TRITON_ATTN 后端
