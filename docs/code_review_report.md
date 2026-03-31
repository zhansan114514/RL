# ACC-Collab 阶段三代码审查报告

**审查日期**: 2026-03-31
**审查范围**: 全流程代码（scripts/01-06, src/核心模块）
**审查团队**: Planner, Engineer, Reviewer, Critic, Tester
**测试状态**: 189个测试全部通过 ✅

---

## 问题汇总（按严重性排序）

### 🔴 高优先级问题

| # | 问题 | 文件位置 | 状态 |
|---|------|----------|------|
| 1 | preference_pairs 缺少 prompt 字段 | `src/trajectory/generator.py:152-173` | ✅ 已修复 |

#### 问题 #1: preference_pairs 缺少 prompt 字段 ✅

**位置**: `src/trajectory/generator.py:152-173`

**问题描述**:
- `generate_trajectories()` 返回的 preference_pairs 缺少 `actor_prompt` 和 `critic_prompt` 字段
- `src/trajectory/preference.py:37,46` 期望这些字段来构建 DPO 训练数据
- 导致偏好数据集的 prompt 字段为空

**影响**: DPO 训练无法正确进行，prompt 是训练的关键输入

**修复方案**:
1. 在遍历 `natural_trajectory` 时提取 `actor_prompt` 和 `critic_prompt`
2. 在构建 preference_pairs 时添加这两个字段

**修复状态**: ✅ 已完成
- 修改文件: `src/trajectory/generator.py`
- 测试验证: 189个测试全部通过

---

### 🟡 中优先级问题

| # | 问题 | 文件位置 | 状态 |
|---|------|----------|------|
| 2 | GPU 设备硬编码 | `scripts/06_full_pipeline.py:186-187` | ⏸️ 需配置化 |
| 3 | 重复模型加载可能导致显存碎片 | `src/training/alternating.py:96-158` | ⏸️ 需优化 |

#### 问题 #2: GPU 设备硬编码

**位置**: `scripts/06_full_pipeline.py:186-187`

**问题描述**:
```python
actor_model = VLLMInference(actor_path, gpu_memory_utilization=0.8, cuda_device=0)
critic_model = VLLMInference(critic_path, gpu_memory_utilization=0.8, cuda_device=1)
```

**影响**:
- GPU 0-1 已被大量占用时可能导致 OOM
- 缺乏灵活性，无法适应不同 GPU 资源环境

**修复方案**:
1. 通过配置文件指定 GPU 设备
2. 已有 `actor_device`/`critic_device` 参数支持，需在配置文件中设置

**配置示例**:
```yaml
# configs/debug_3samples.yaml
step06:
  actor_device: 14  # 使用空闲 GPU
  critic_device: 15
```

---

#### 问题 #3: 重复模型加载可能导致显存碎片

**位置**: `src/training/alternating.py:96-158`

**问题描述**:
每个 iteration 中，actor 和 critic 模型被多次加载和释放：
1. 生成 Critic 轨迹（加载 actor + critic）
2. 训练 Critic（释放后重新加载）
3. 生成 Actor 轨迹（加载 actor + critic）
4. 训练 Actor（释放后重新加载）
5. 验证（再次加载）

**影响**:
- 显存碎片化
- 效率较低
- 可能触发 OOM

**修复方案**:
1. 当前已有 `cleanup()` 调用和 `gc.collect()`
2. 可考虑复用模型实例或优化显存管理
3. 优先级：当前实现可用，后续优化

---

### 🟢 低优先级问题

| # | 问题 | 文件位置 | 状态 |
|---|------|----------|------|
| 4 | 配置系统与文档描述不符 | `configs/` | ⏸️ 暂缓 |
| 5 | vLLM CUDA 设备映射 | `src/inference/vllm_server.py:59-73` | ✅ 已修复 |
| 6 | 模型类型检测可能遗漏 | `src/utils/model_utils.py` | ⏸️ 暂缓 |

#### 问题 #4: 配置系统与文档描述不符

**位置**: `configs/`, `CLAUDE.md`

**问题描述**:
- CLAUDE.md 提到分层配置系统（base.yaml + model/*.yaml + data/*.yaml + train/*.yaml）
- 实际只使用单一 `config.yaml` 和 `debug_3samples.yaml`

**影响**: 文档与实现不一致

**修复方案**: 保持当前单一配置文件，或实现分层配置合并

---

#### 问题 #5: vLLM CUDA 设备映射 ✅

**位置**: `src/inference/vllm_server.py:59-73`

**问题描述**: 当 `CUDA_VISIBLE_DEVICES` 已设置时，逻辑设备到物理设备的映射

**状态**: ✅ 已在阶段3.0修复

---

#### 问题 #6: 模型类型检测可能遗漏

**位置**: `src/utils/model_utils.py`

**问题描述**: 只通过字符串匹配检测模型类型

**影响**: 非标准命名的模型会默认使用 llama3 配置

**修复方案**: 可通过加载 config.json 来准确检测，或手动指定 model_type

---

## 审查通过的模块 ✅

### Algorithm 1 实现 (`src/trajectory/generator.py`)
- ✅ 自然审议轨迹生成
- ✅ MC Roll-out 奖励估计（one-step）
- ✅ 引导向正确答案（z_y）
- ✅ 引导向错误答案（z_not_y）
- ✅ Delta 计算和偏好对构建

### Prompt 模板 (`src/prompts/templates.py`)
- ✅ 6种模板类型完整
- ✅ BoolQ 模板与论文 Appendix C 一致
- ✅ MMLU/BBH/SCIQ/ARC 模板结构正确

### Reward 计算 (`src/reward/partial.py`)
- ✅ `compute_reward_delta` 实现正确
- ✅ `select_preference_pairs` 符合 Algorithm 1

### MC Roll-out (`src/deliberation/rollouts.py`)
- ✅ `remaining_rounds=1` one-step roll-out
- ✅ 符合论文 "one-step roll-out heuristics"

### 交替训练 (`src/training/alternating.py`)
- ✅ 第2轮基于第1轮输出（路径传递正确）
- ✅ 验证集评估和 early stopping

### DPO 训练 (`src/training/dpo_trainer.py`)
- ✅ beta 参数正确传递
- ✅ bf16/fp16 自动检测（V100 使用 float16）

---

## 步骤1-5执行计划

### 步骤1: 极小规模验证（3样本）

**目标**: 验证数据流完整性

**命令**:
```bash
CUDA_VISIBLE_DEVICES=14,15 python scripts/06_full_pipeline.py \
  --config configs/debug_3samples.yaml
```

**验证点**:
- [ ] 数据加载正确
- [ ] 轨迹生成成功（偏好对数量>0）
- [ ] DPO 训练 loss 下降
- [ ] 评估结果完整

**预计耗时**: 30-60分钟

---

### 步骤2: 中等规模验证（10样本）

**目标**: 验证训练稳定性

**命令**:
```bash
CUDA_VISIBLE_DEVICES=14,15 python scripts/06_full_pipeline.py \
  --model_name google/gemma-2-2b-it \
  --dataset boolq \
  --output_dir experiments/debug_10samples \
  --max_samples 10 \
  --num_iterations 1 \
  --batch_size 2
```

**验证点**:
- [ ] MC Roll-out 奖励估计合理
- [ ] 偏好对 delta 分布合理
- [ ] 准确率有提升

**预计耗时**: 1-2小时

---

### 步骤3: 核心算法验证

**目标**: 验证 Algorithm 1 实现正确性

**验证点**:
- [x] one-step roll-out: remaining_rounds=1
- [x] 引导审议生成正确
- [x] delta 计算正确
- [x] 偏好对筛选正确

**状态**: ✅ 代码审查已通过

---

### 步骤4: 完整 BoolQ 复现（1轮）

**目标**: 论文级结果复现

**命令**:
```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/06_full_pipeline.py \
  --model_name google/gemma-2-2b-it \
  --dataset boolq \
  --output_dir experiments/boolq_acc_collab \
  --num_iterations 1 \
  --batch_size 4
```

**验证点**:
- [ ] 准确率达到论文趋势（~65% → ~70%）
- [ ] 改进率 ~5-10%

**预计耗时**: 1-2天

---

### 步骤5: ACC-Collab+（2轮）

**目标**: 验证2轮交替训练效果

**命令**:
```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/06_full_pipeline.py \
  --model_name google/gemma-2-2b-it \
  --dataset boolq \
  --output_dir experiments/boolq_acc_collab_plus \
  --num_iterations 2 \
  --batch_size 4
```

**验证点**:
- [ ] ACC-Collab+ >= ACC-Collab
- [ ] 第2轮路径传递正确

**预计耗时**: 2-3天

---

## GPU 资源使用策略

**V100 约束**:
- Gemma2 必须使用 float32
- 单模型 ~9.77 GiB
- 两模型共需 ~20 GiB

**推荐配置**:
| 规模 | GPU 配置 | gpu_memory_utilization |
|------|----------|------------------------|
| 3样本 | 单 GPU 共享 | 0.45 × 2 = 0.9 |
| 10样本 | 单 GPU 共享 | 0.45 × 2 = 0.9 |
| 完整 | 双 GPU 分离 | 0.8 × 2 = 1.6 |

---

## 关键配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| num_rounds | 5 | 审议轮数 |
| num_simulations | 5 | MC roll-out 次数 |
| reward_threshold | 0.0 | 最小 delta 阈值 |
| lora_r | 256 | LoRA rank |
| learning_rate | 5e-5 | 学习率 |
| batch_size | 2-4 | 批次大小 |
| beta | 0.1 | DPO beta 参数 |

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| GPU 资源不足 | 高 | 高 | 使用多 GPU 集群，分时调度 |
| 训练不稳定 | 中 | 中 | 调整超参数，增加验证监控 |
| 准确率不达标 | 中 | 高 | 检查 prompt 模板和答案抽取 |

---

## 总结

**代码审查结果**: ✅ 核心算法实现正确，高优先级问题已修复

**测试覆盖**: 189 个测试全部通过

**准备状态**: ✅ 可以开始阶段三实验验证

**下一步**: GPU 资源就绪后执行步骤1（3样本验证）
