# ACC-Collab 复现计划

## 项目概述

本项目旨在复现 ICLR 2025 论文《ACC-Collab: An Actor-Critic Approach to Multi-Agent LLM Collaboration》的核心方法和实验结果。

**核心思想**：联合训练 Actor 和 Critic 两个 LLM 智能体，通过迭代审议（Deliberation）协作解决推理任务。使用 Guided Collaborative Trajectories 生成偏好数据，再通过 DPO 进行优化。

## 关键问题状态（Critic 发现）

| 问题 | 严重性 | 状态 | 说明 |
|------|--------|------|------|
| 1. alternating_train 模型路径 bug | 🔴 高 | ✅ 已修复 | 第2轮训练现在基于第1轮输出 |
| 2. MC Roll-out 一致性 | 🟡 中 | ✅ 已修复 | 修改为真正的 one-step roll-out |
| 3. DPO beta 参数缺失 | 🟡 中 | ✅ 已添加 | beta=0.1 贯穿配置和训练 |
| 4. BBH 按类别划分 | 🟢 低 | ✅ 已实现 | 代码已按类别分层划分 |
| 5. GUIDED_DELIBERATION_ACTOR 模板 | 🟡 中 | ✅ 已修复 | BoolQ: 添加完整引导句，MMLU: 添加完整引导句 |
| 6. 验证集评估和 early stopping | 🟡 中 | ✅ 已实现 | alternating_train 支持验证集性能检测和提前停止 |
| 7. _detect_model_type 重复代码 | 🟢 低 | ✅ 已修复 | 提取到 src/utils/model_utils.py 统一管理 |

## 数据流管线

```
数据集加载 (src.data)
  -> 自然审议 + 引导审议 (src.deliberation)
  -> MC Roll-out 奖励估计 (src.deliberation.rollouts)
  -> 偏好对构建 (src.trajectory)
  -> DPO 训练 (src.training)
  -> 评估 (src.evaluation)
```

---

## 分阶段复现计划

### 阶段 1：核心 Bug 修复 ✅ 已完成

**目标**：修复阻碍正确复现的关键 bug

#### 已完成任务

| 任务 | 状态 | 负责人 | 验收 |
|------|------|--------|------|
| alternating_train 模型路径 bug | ✅ | Engineer | 测试通过 |
| MC Roll-out one-step 修复 | ✅ | Critic | 测试通过 |
| DPO beta 参数配置 | ✅ | Engineer | 测试通过 |
| BBH 按类别划分 | ✅ | Engineer | 测试通过 |
| 测试覆盖扩展 (90→189) | ✅ | Tester | 189个全部通过 |

**验收标准**：
- [x] 189 个测试全部通过
- [x] 所有核心 bug 已修复
- [x] 代码审查完成

**状态**：✅ 已完成（2026-03-31）

---

### 阶段 2：功能完善 ✅ 已完成

**目标**：确保所有功能组件完整且可用

#### 已完成任务

| 任务 | 状态 | 负责人 | 验收 |
|------|------|--------|------|
| Prompt 模板验证与修复 | ✅ | Critic | 与论文 Appendix C 一致 |
| 验证集评估和 early stopping | ✅ | Engineer | 功能已添加 |
| `_detect_model_type` 重复代码提取 | ✅ | Engineer | 提取到 src/utils/model_utils.py |

**验收标准**：
- [x] 189 个测试全部通过
- [x] 模板与论文 Appendix C 一致
- [x] 代码重复已消除

**状态**：✅ 已完成（2026-03-31）

---

#### 2.2 端到端小规模验证 ⏳ 待开始

**任务**：
- [ ] 使用 Gemma-2-2B 在 BoolQ 上运行 10 个样本
- [ ] 验证数据流：加载 -> 轨迹生成 -> 偏好构建 -> DPO训练 -> 评估
- [ ] 检查中间输出格式正确性

**命令**：
```bash
python scripts/06_full_pipeline.py \
  --model_name google/gemma-2-2b-it \
  --dataset boolq \
  --output_dir experiments/debug \
  --max_samples 10 \
  --num_iterations 1
```

**验收标准**：
- [ ] 端到端无报错完成
- [ ] 生成偏好对数量合理（>0）
- [ ] 输出结果文件完整（results.json）

**预计耗时**：1-2 小时

**负责人**：Implementer

---

#### 2.3 核心算法验证 ⏳ 待开始

**任务**：
- [ ] 验证 MC Roll-out 奖励估计的合理性
- [ ] 检查偏好对的 delta 分布（可视化）
- [ ] 验证 DPO 训练 loss 下降
- [ ] 检查 LoRA 权重正确应用

**验收标准**：
- [ ] MC Roll-out 估计的准确率与实际接近（误差 <10%）
- [ ] 偏好对 delta 分布合理（有正有负，均值 >0）
- [ ] DPO training loss 稳定下降
- [ ] 评估指标逐轮提升（或保持稳定）

**负责人**：Implementer + Analyst

---

#### 2.4 配置系统完善 ⏳ 待开始

**任务**：
- [ ] 确保所有超参数可通过配置文件设置
- [ ] 验证配置合并逻辑正确

**验收标准**：
- [ ] configs/base.yaml 和 configs/config.yaml 包含所有必要参数
- [ ] 命令行参数可覆盖配置文件

**负责人**：Engineer

---

### 阶段 3：实验验证 🔄 进行中

**目标**：完成论文实验复现，验证方法有效性

**开始时间**：2026-03-31

**环境**：
- 硬件：16x V100-SXM3-32GB
- 模型：google/gemma-2-2b-it
- 数据集：boolq

#### 3.0 运行时修复 ✅ 已完成（2026-03-31）

**三轮调试过程**：

| 轮次 | 问题 | 根因 | 修复 |
|------|------|------|------|
| 第1轮 | NVML 错误 | PyTorch 2.10+cu128 在 V100 上 device_map="auto" 触发 NVML 兼容性错误 | 改为 device_map={"": 0} |
| 第1轮 | bf16 不支持 | V100 不支持 bf16，DPOConfig 中 bf16=True | 自动检测硬件，V100 用 fp16 |
| 第1轮 | test_data 为空 | BoolQ 没有 test split | fallback 到 val_data |
| 第2轮 | GPU OOM | gpu_memory_utilization=0.3 太小（模型 float32 = 9.77GiB > 9.6GiB） | 调整为 0.45 |
| 第3轮 | GPU 2 不足 | 两模型分放 GPU 8(32GB) + GPU 2(8.5GB)，GPU 2 不够 | 改为共享单 GPU |
| 第3轮 | Gemma2 不支持 fp16 | vLLM 报错 "Gemma2 不支持 fp16 数值不稳定" | 保持 float32 + 单 GPU 共享 |
| 第3轮 | GPU 全被占用 | 等待 GPU 资源释放 | 等待中 |
| 第4轮 | Gemma2 不支持 float16 | vLLM 报错 "Gemma2 不支持 fp16 数值不稳定" | 保持 float32 |
| 第4轮 | GPU 2 不够放第二模型 | actor/critic 分放不同 GPU，GPU 2(8.5GB) 不够 | 改为单 GPU 共享 |
| 第4轮 | CUDA_VISIBLE_DEVICES 映射错误 | cuda_device=0 被直接设为 CUDA_VISIBLE_DEVICES="0"，而非物理 GPU 8 | 修复映射逻辑 |

**修复文件清单**：
1. `src/training/dpo_trainer.py` — bf16/fp16 自动检测 + device_map={"": 0} + dtype 一致性
2. `src/training/alternating.py` — actor_device/critic_device 参数 + GPU 内存清理 + 单 GPU 共享
3. `src/inference/vllm_server.py` — cuda_device 参数 + CUDA_VISIBLE_DEVICES 正确映射（逻辑→物理）
4. `scripts/06_full_pipeline.py` — test_data fallback + actor_device/critic_device 传递
5. `tests/test_training.py` — mock 函数兼容 **kwargs
6. `configs/debug_3samples.yaml` — actor_device=0, critic_device=0 配置

**关键决策**：
- V100 上 Gemma2 必须使用 float32（不支持 bf16 也不支持 fp16）
- 单 GPU 需 ≥20GB 空闲才能同时加载两个模型（9.77GiB × 2）
- `gpu_memory_utilization` 每个模型设 0.45，两个模型共需 0.9 × 32 = 28.8 GiB

**验收状态**：
- [x] 189 个测试全部通过（代码修改未破坏功能）
- [x] NVML 错误已解决（device_map 显式指定）
- [x] bf16/fp16 已解决（硬件自动检测）
- [x] GPU OOM 已解决（单 GPU 共享 + 0.45 利用率）
- [ ] 端到端运行成功（等待 GPU 资源）

#### 3.1 小规模验证 ⏳ 等待 GPU 资源

**阻塞原因**：当前所有 16 块 GPU 均被其他进程占用，无 ≥20GB 空闲 GPU 可用。

**恢复命令**（GPU 就绪后）：
```bash
# 需要至少 1x V100 (≥20GB 空闲)
# 使用 CUDA_VISIBLE_DEVICES=X 指定空闲 GPU
CUDA_VISIBLE_DEVICES=X python scripts/06_full_pipeline.py \
  --config configs/debug_3samples.yaml
```

**渐进式验证策略**：从 3样本 → 10样本 → 全量

| 子任务 | 样本数 | 状态 | 负责人 | GPU 需求 | 预计耗时 |
|--------|--------|------|--------|---------|----------|
| 3.1.1 极小规模验证 | 3 | ⏳ 等待GPU | Runner | 1x V100(≥20GB free) | 30-60分钟 |
| 3.1.2 中等规模验证 | 10 | ⏳ 等待3.1.1 | Implementer | 1x V100(≥20GB free) | 1-2小时 |
| 3.1.3 核心算法验证 | - | ⏳ 等待3.1.2 | Implementer + Analyst | 1x V100 | 1-2小时 |

**运行命令**（GPU 就绪后）：
```bash
# 单 GPU 模式（两个模型共享）
CUDA_VISIBLE_DEVICES=X python scripts/06_full_pipeline.py \
  --config configs/debug_3samples.yaml
  --max_samples 3 \
  --num_iterations 1
```

**验收标准**：
- [x] 任务列表已创建（5个任务）
- [x] 实验已启动（GPU 10）
- [ ] 3样本端到端无报错
- [ ] 10样本训练稳定
- [ ] 算法组件验证通过

---

#### 3.1 BoolQ 完整复现 ⏳ 等待小规模验证

**任务**：
- [ ] 使用完整 BoolQ 训练集
- [ ] 运行 ACC-Collab（1 轮交替训练）
- [ ] 运行 ACC-Collab+（2 轮交替训练）
- [ ] 对比基线（单次回答、无审议）

**命令**：
```bash
# ACC-Collab (1 轮)
python scripts/06_full_pipeline.py \
  --model_name google/gemma-2-2b-it \
  --dataset boolq \
  --output_dir experiments/boolq_acc_collab \
  --num_iterations 1

# ACC-Collab+ (2 轮)
python scripts/06_full_pipeline.py \
  --model_name google/gemma-2-2b-it \
  --dataset boolq \
  --output_dir experiments/boolq_acc_collab_plus \
  --num_iterations 2
```

**验收标准**：
- [ ] 准确率提升达到论文报告的趋势（审议后 > 单次）
- [ ] ACC-Collab+ >= ACC-Collab（2轮 >= 1轮）
- [ ] 结果可复现（不同运行结果接近）

**论文参考指标**（以 Gemma-2-2B 为参考，需根据实际调整）：
- BoolQ 单次：~65%
- BoolQ 审议后：~70%+
- 改进率：~5-10%

**预计耗时**：1-2 天

**负责人**：Implementer

---

#### 3.2 多数据集扩展验证

**任务**：
- [ ] MMLU 复现
- [ ] BBH 复现
- [ ] SCIQ 复现
- [ ] ARC 复现

**验收标准**：
- [ ] 每个数据集都能端到端运行
- [ ] 准确率提升趋势与论文一致
- [ ] 生成对比表格

**预计耗时**：2-3 天

**负责人**：Implementer

---

#### 3.3 消融实验

**任务**：
- [ ] 不同审议轮数（T=1,3,5,7）的影响
- [ ] 不同 MC 模拟次数的影响
- [ ] 不同 reward_threshold 的影响
- [ ] LoRA rank 的影响

**验收标准**：
- [ ] 生成消融实验报告
- [ ] 关键超参数敏感性分析图表

**预计耗时**：1-2 天

**负责人**：Implementer + Analyst

---

## 关键里程碑

| 里程碑 | 目标 | 预计完成时间 | 状态 |
|--------|------|--------------|------|
| M1: Bug 修复完成 | alternating_train、MC Roll-out、DPO beta 全部修复 | Day 1 | ✅ 已完成 |
| M2: 功能完善完成 | Prompt 模板、验证集评估、代码重复问题全部解决 | Day 2 | ✅ 已完成 |
| M3.0: 极小规模验证通过 | 3样本端到端运行成功 | Day 3 | 🔄 进行中 |
| M3.1: 中等规模验证通过 | 10样本端到端运行成功 | Day 3 | ⏳ 待开始 |
| M3.2: 核心算法验证通过 | MC Roll-out、DPO loss、偏好对分布验证 | Day 3 | ⏳ 待开始 |
| M4: BoolQ 复现完成 | 完整数据集运行，结果符合预期 | Day 5 | ⏳ 待开始 |
| M5: 多数据集验证完成 | 5个数据集全部复现 | Day 8 | ⏳ 待开始 |
| M6: 消融实验完成 | 超参数敏感性分析 | Day 10 | ⏳ 待开始 |

---

## 风险与缓解措施

| 风险 | 影响 | 缓解措施 | 状态 |
|------|------|----------|------|
| alternating_train bug | 2轮训练无效 | ✅ 已修复 | 已解决 |
| MC Roll-out 不一致 | 奖励估计偏差 | ✅ 已修复 | 已解决 |
| Prompt 模板不一致 | 引导审议效果差 | ✅ 已修复 | 已解决 |
| GPU 资源不足 | 无法运行大模型 | 先用 Gemma-2-2B 验证 | 可控 |
| 训练不稳定 | DPO 不收敛 | 调整 learning rate、batch size、beta | 可控 |

---

## 下一步行动（优先级排序）

### 立即执行（2026-03-31）

1. **Implementer**：执行极小规模验证（3样本，30-60分钟）
   ```bash
   python scripts/06_full_pipeline.py \
     --model_name google/gemma-2-2b-it \
     --dataset boolq \
     --output_dir experiments/debug_small \
     --max_samples 3 \
     --num_iterations 1
   ```

2. **Planner**：监控实验进度，更新任务状态

3. **Analyst**：准备实验结果记录模板和分析工具

### 待执行（依赖任务完成后）

4. **Implementer**：中等规模验证（10样本）
5. **Implementer + Analyst**：核心算法验证与分析

## 当前状态总结

**阶段 1 & 2 已完成**：
- ✅ 189 个测试全部通过
- ✅ 7 个关键问题全部修复
- ✅ Prompt 模板与论文一致
- ✅ 代码质量提升（消除重复）

**准备进入阶段 3：实验验证**

---

## 附录：已修复问题详情

### A1. alternating_train 模型路径 bug

**问题**：第2轮训练从原始模型重新开始，而非基于第1轮微调结果

**修复**：`src/training/alternating.py` 确保路径正确传递

### A2. MC Roll-out one-step 修复

**问题**：模拟所有剩余轮（如4轮），论文要求只模拟1轮

**修复**：`remaining_rounds=1`，符合论文 "one-step roll-out heuristics"

### A3. DPO beta 参数

**问题**：beta 参数未显式配置

**修复**：`configs/base.yaml` 添加 `beta: 0.1`，贯穿调用链

### A4. BBH 按类别划分

**问题**：全局划分，论文要求按每个 task_type 内部分别划分

**修复**：`src/data/loader.py` 实现按 task 分组的 stratified split

### A5. GUIDED_DELIBERATION_ACTOR 模板修复

**问题**：
- BoolQ 模板缺少 "take these answers and the passage into consideration"
- MMLU 模板引导句不完整

**影响**：引导审议效果可能低于论文

**修复**：
- BoolQ: 添加完整引导句
- MMLU: 添加完整引导句

**测试**：`tests/test_prompts.py` 相关测试

---

### A6. 验证集评估和 early stopping

**问题**：缺少训练过程中的验证集性能监控

**影响**：无法及时检测过拟合或训练问题

**修复**：`alternating_train` 添加验证集评估和 early stopping 机制

---

### A7. _detect_model_type 重复代码

**问题**：多个脚本中重复定义 `_detect_model_type` 函数

**影响**：代码重复，维护困难

**修复**：提取到 `src/utils/model_utils.py` 统一管理

---

### A8. 未明确指定的超参数

以下超参数在论文中未明确指定具体值，我们使用以下默认值：

| 参数 | 值 | 依据 | 备注 |
|------|-----|------|------|
| reward_threshold (ε) | 0.0 | 最大化数据利用 | 论文 Algorithm 1 定义了 ε，但未指定具体值。ε=0 意味着保留所有 delta≥0 的偏好对 |
| num_simulations | 5 | Monte Carlo 常见选择 | 论文说 "multiple times"，5 次平衡准确率和效率 |
| temperature | 0.7 | LLM 采样标准值 | 平衡创造性和确定性 |
| max_tokens | 256 | 常用生成长度 | 对于 yes/no 问题充足，多选题可能需要验证 |
| beta (DPO) | 0.1 | trl 库默认值 | DPO 原始论文推荐值 |

**关于 ε = 0.0 的特别说明**：

论文 Algorithm 1 定义了 reward threshold ε，但未指定具体数值。我们使用 ε=0.0 的理由：
1. 最大化数据利用率，不丢弃任何生成的偏好对
2. 论文 Eq. 5 说 "if neither value is above the threshold, then the example is thrown out"，但未指定阈值
3. 对于初期复现，使用 ε=0 是保守选择

如果后续发现训练不稳定，可以考虑：
- 增加 ε 到 0.1 或 0.2，过滤掉 delta 较小的低质量样本
- 这会减少训练数据量，但可能提高数据质量

---

*本计划将根据实际执行情况动态调整*

---

## 会话状态快照

**最后更新**：2026-03-31 09:10

**当前阶段**：阶段 3 - 实验验证

**运行中的实验**：
- GPU 10 上运行 3样本端到端测试
- 命令：`python scripts/06_full_pipeline.py --model_name google/gemma-2-2b-it --dataset boolq --output_dir experiments/debug_small --max_samples 3 --num_iterations 1`
- 启动时间：约 09:05

**任务列表**：
- #5: 小规模验证 (3样本) - 🔄 进行中
- #6: 中等规模验证 (10样本) - ⏳ 等待中
- #7: 核心算法验证 - ⏳ 等待中
- #9: 完整实验 (1轮) - ⏳ 等待中
- #8: ACC-Collab+ (2轮) - ⏳ 等待中

**下次恢复工作**：
1. 检查 experiments/debug_small/ 目录结果
2. 验证 3样本实验是否成功
3. 如成功，启动 10样本实验

