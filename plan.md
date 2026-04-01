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

#### 3.0.1 NVML patchelf 修复方案 ✅ 已完成（2026-04-01）

**问题**：PyTorch 2.10+cu128 在 NVIDIA 驱动 450.142.00 的 V100 上调用 `nvmlDeviceGetNvLinkRemoteDeviceType` 失败。

**最终方案**：使用 patchelf 将 stub .so 注入到 libnvidia-ml.so.1 副本中。

```bash
# 创建 stub 库
mkdir -p /tmp/nvml_fix
cat > /tmp/nvml_fix/nvml_stub.c << 'EOF'
int nvmlDeviceGetNvLinkRemoteDeviceType(void *device, unsigned int link, int *type) {
    if (type) *type = 0;
    return 0;
}
EOF
gcc -shared -fPIC -o /tmp/nvml_fix/libnvml_fix.so /tmp/nvml_fix/nvml_stub.c
cp /lib/x86_64-linux-gnu/libnvidia-ml.so.1 /tmp/nvml_fix/
patchelf --add-needed libnvml_fix.so /tmp/nvml_fix/libnvidia-ml.so.1

# 运行时设置
export LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH
```

**验证**：CUDA tensor 操作、模型加载、DPO 训练均通过。

#### 3.0.2 端到端管线调通 ✅ 已完成（2026-04-01）

**步骤 1-5 全部成功运行**（3样本 BoolQ，GPU 5）：

| 步骤 | 命令 | 结果 | 耗时 |
|------|------|------|------|
| Step 1: 生成轨迹 | `CUDA_VISIBLE_DEVICES=5 python scripts/01_generate_trajectories.py --config configs/config.yaml` | 22 个偏好对 | ~5 min |
| Step 2: 构建偏好 | `python scripts/02_build_preferences.py --config configs/config.yaml` | actor 22对 + critic 22对 | ~1s |
| Step 3: 训练 Critic | `CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH python scripts/03_train_critic.py --config configs/config.yaml` | loss=2.585, 6步 | ~60s |
| Step 4: 训练 Actor | 同上 + `04_train_actor.py` | loss=0.815, 6步 | ~60s |
| Step 5: 评估 | 同上 + `05_evaluate.py` | acc=33.3% (3样本) | ~4 min |

**训练参数调整**（适应 V100 32GB）：
- batch_size: 4 → 1（解决 GPU OOM）
- max_length: 2048 → 1024（减少 logits 内存占用）
- gradient_checkpointing: True（减少激活内存）
- max_model_len: 1024 → 4096（容纳审议上下文）

**代码修复清单**：
1. `src/training/dpo_trainer.py` — 添加 gradient_checkpointing 参数
2. `scripts/03_train_critic.py` — batch_size=1, max_length=1024
3. `scripts/04_train_actor.py` — batch_size=1, max_length=1024
4. `scripts/05_evaluate.py` — test_data fallback + max_model_len=4096
5. `configs/config.yaml` — 更新 step03/04 默认参数

#### 3.1 小规模验证 ✅ 已完成（3样本）

| 子任务 | 样本数 | 状态 | 结果 |
|--------|--------|------|------|
| 3.1.1 极小规模验证 | 3 | ✅ 完成 | 端到端通过，acc=33.3% |
| 3.1.2 中等规模验证 | 10 | ⏳ 下一步 | 待运行 |
| 3.1.3 核心算法验证 | - | ⏳ 下一步 | 待运行 |
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
| M3.0: 极小规模验证通过 | 3样本端到端运行成功 | Day 3 | ✅ 已完成 |
| M3.1: 中等规模验证通过 | 10样本端到端运行成功 | Day 3 | 🔄 进行中 |
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

### 立即执行（2026-04-01）

1. **Implementer**：运行中等规模验证（10样本 BoolQ）
   ```bash
   # 修改 configs/config.yaml 中 max_samples: 10
   # 步骤 1-2: 重新生成轨迹（步骤 3-5 可复用当前数据或重新生成）
   CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH \
     python scripts/01_generate_trajectories.py --config configs/config.yaml

   # 步骤 2-5 依次运行
   python scripts/02_build_preferences.py --config configs/config.yaml
   CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH \
     python scripts/03_train_critic.py --config configs/config.yaml
   CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH \
     python scripts/04_train_actor.py --config configs/config.yaml
   CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH \
     python scripts/05_evaluate.py --config configs/config.yaml
   ```

2. **Engineer**：确保 NVML fix 脚本和文档更新完成

### 待执行（依赖任务完成后）

3. **Implementer + Analyst**：核心算法验证与分析
4. **Implementer**：完整 BoolQ 复现（全量数据）

## 当前状态总结

**阶段 1 & 2 已完成**：
- ✅ 189 个测试全部通过
- ✅ 7 个关键问题全部修复
- ✅ Prompt 模板与论文一致
- ✅ 代码质量提升（消除重复）

**阶段 3 进展**：
- ✅ NVML 兼容性问题解决（patchelf 方案）
- ✅ 端到端管线调通（3样本 BoolQ）
- ✅ 步骤 1-5 全部运行成功
- 🔄 中等规模验证（10样本）待运行

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

**最后更新**：2026-04-01 07:40

**当前阶段**：阶段 3 - 实验验证

**已完成的验证**：
- ✅ 步骤 1: 生成轨迹（22个偏好对）
- ✅ 步骤 2: 构建偏好数据集（actor 22对 + critic 22对）
- ✅ 步骤 3: 训练 Critic（loss=2.585, 6步, ~60s）
- ✅ 步骤 4: 训练 Actor（loss=0.815, 6步, ~60s）
- ✅ 步骤 5: 评估（acc=33.3%, 3样本验证集）

**关键修复**：
1. NVML patchelf 方案：解决了 PyTorch 2.10 + 旧驱动不兼容问题
2. GPU OOM：batch_size=1, max_length=1024, gradient_checkpointing=True
3. max_model_len：从 1024 增加到 4096（容纳审议上下文）
4. test_data fallback：使用 validation split 替代 test split

**运行命令模板**：
```bash
# 所有训练/评估脚本使用此模板
CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH \
  python scripts/XX_xxx.py --config configs/config.yaml
```
- #6: 中等规模验证 (10样本) - ⏳ 等待中
- #7: 核心算法验证 - ⏳ 等待中
- #9: 完整实验 (1轮) - ⏳ 等待中
- #8: ACC-Collab+ (2轮) - ⏳ 等待中

**下次恢复工作**：
1. 检查 experiments/debug_small/ 目录结果
2. 验证 3样本实验是否成功
3. 如成功，启动 10样本实验

---

## 阶段三步骤 3 执行计划（详细版）

### 计划概述

**目标**：完成步骤 3（训练 Critic）和步骤 4（训练 Actor）的执行，并准备步骤 5（评估）

**关键阻塞**：
- 当前环境 PyTorch 版本（2.10.0+cu128）与 NVIDIA 驱动（450.142.00）不兼容
- 需要降级 PyTorch 到 2.3.1+cu118

**硬件资源**：
- GPU 5, 6, 15 完全空闲（32GB each）
- 单模型 float32 约 9.77GiB，两模型共需 ~20GiB

---

### 环境准备步骤

#### 步骤 0.1：降级 PyTorch（必须优先完成）

**问题**：PyTorch 2.10.0+cu128 调用 `nvmlDeviceGetNvLinkRemoteDeviceType`，该符号在驱动 450.142.00 中不存在。

**解决方案**：
```bash
# 卸载当前 PyTorch
pip uninstall torch trl transformers peft accelerate -y

# 安装兼容版本
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[dev]"

# 验证
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
# 预期输出：PyTorch: 2.3.1+cu118, CUDA: 11.8

# 验证 CUDA 可用性
python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print('CUDA 可用')"
```

**验收标准**：
- [ ] PyTorch 版本为 2.3.1+cu118
- [ ] `torch.cuda.is_available()` 返回 True
- [ ] 189 个测试仍然通过（确保兼容性）

**预计耗时**：5-10 分钟

---

### 步骤 3：训练 Critic

#### 3.1 输入验证

**输入位置**：`experiments/debug_3samples/preferences/critic_preferences`

**验证命令**：
```bash
python -c "
from datasets import load_from_disk
import os
path = 'experiments/debug_3samples/preferences/critic_preferences'
if not os.path.exists(path):
    print('ERROR: 偏好数据集不存在，请先运行步骤 1-2')
    exit(1)
ds = load_from_disk(path)
print(f'偏好对数量: {len(ds)}')
print(f'特征列: {ds.column_names}')
print(f'第一个样本:')
for k, v in ds[0].items():
    print(f'  {k}: {v[:100] if isinstance(v, str) else v}...')
"
```

**预期输出**：
- 偏好对数量：>0（3 样本应生成约 6-9 个 critic 偏好对）
- 特征列：prompt, chosen, rejected, round, delta, direction

**验收标准**：
- [ ] 偏好数据集存在且格式正确
- [ ] 偏好对数量合理

---

#### 3.2 执行命令

**命令**：
```bash
# 使用 GPU 5（完全空闲）
CUDA_VISIBLE_DEVICES=5 python scripts/03_train_critic.py \
  --config configs/debug_3samples.yaml
```

**参数说明**（从 configs/debug_3samples.yaml 继承）：
- `model_name`: google/gemma-2-2b-it
- `preference_dir`: experiments/debug_3samples/preferences
- `output_dir`: experiments/debug_3samples/critic
- `lora_r`: 256
- `learning_rate`: 5.0e-5
- `batch_size`: 2（debug 配置）
- `num_epochs`: 1
- `max_length`: 2048

**预计耗时**：
- 3 样本 × 2 batch_size × 1 epoch ≈ 2-5 分钟

---

#### 3.3 输出验证

**输出位置**：`experiments/debug_3samples/critic/`

**验证命令**：
```bash
# 检查目录结构
ls -la experiments/debug_3samples/critic/
# 应包含：config.json, adapter_model/, tokenizer files

# 检查模型文件
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
path = 'experiments/debug_3samples/critic'
if not os.path.exists(path):
    print('ERROR: Critic 模型目录不存在')
    exit(1)
print('加载 Critic 模型...')
model = AutoModelForCausalLM.from_pretrained(path, device_map='cpu')
tokenizer = AutoTokenizer.from_pretrained(path)
print(f'模型类型: {model.config.model_type}')
print(f'词汇表大小: {len(tokenizer)}')
print('Critic 模型验证通过')
"
```

**验收标准**：
- [ ] 输出目录存在，包含 config.json
- [ ] 模型可以加载（device_map='cpu' 测试）
- [ ] tokenizer 文件完整

---

### 步骤 4：训练 Actor

#### 4.1 输入验证

**输入位置**：`experiments/debug_3samples/preferences/actor_preferences`

**验证命令**：
```bash
python -c "
from datasets import load_from_disk
import os
path = 'experiments/debug_3samples/preferences/actor_preferences'
if not os.path.exists(path):
    print('ERROR: Actor 偏好数据集不存在')
    exit(1)
ds = load_from_disk(path)
print(f'Actor 偏好对数量: {len(ds)}')
"
```

---

#### 4.2 执行命令

**命令**：
```bash
# 使用 GPU 6（完全空闲）
CUDA_VISIBLE_DEVICES=6 python scripts/04_train_actor.py \
  --config configs/debug_3samples.yaml
```

**参数**：与 Critic 相同，仅输出目录不同

**预计耗时**：2-5 分钟

---

#### 4.3 输出验证

**输出位置**：`experiments/debug_3samples/actor/`

**验证命令**：同 Critic 验证

**验收标准**：
- [ ] 输出目录存在
- [ ] 模型可以加载

---

### 步骤间数据流验证

**检查点脚本**：
```bash
python -c "
import os
import json

# 检查完整数据链
steps = {
    'Step 1 (轨迹)': 'experiments/debug_3samples/trajectories/trajectory_pairs.json',
    'Step 2 (偏好-actor)': 'experiments/debug_3samples/preferences/actor_preferences',
    'Step 2 (偏好-critic)': 'experiments/debug_3samples/preferences/critic_preferences',
    'Step 3 (Critic)': 'experiments/debug_3samples/critic',
    'Step 4 (Actor)': 'experiments/debug_3samples/actor',
}

print('=== 数据链完整性检查 ===')
for step, path in steps.items():
    exists = os.path.exists(path)
    status = '✓' if exists else '✗'
    print(f'{status} {step}: {path}')
    if not exists:
        print(f'  WARNING: {step} 缺失，后续步骤可能失败')

# 验证轨迹文件内容
traj_path = steps['Step 1 (轨迹)']
if os.path.exists(traj_path):
    with open(traj_path) as f:
        pairs = json.load(f)
    print(f'\n轨迹对数量: {len(pairs)}')
    if pairs:
        print(f'示例 keys: {list(pairs[0].keys())}')
"
```

---

### 步骤 5：评估（准备工作）

#### 5.1 评估命令准备

**命令**：
```bash
# 使用 GPU 15（完全空闲）
CUDA_VISIBLE_DEVICES=15 python scripts/05_evaluate.py \
  --config configs/debug_3samples.yaml
```

**注意**：评估需要加载训练好的 Actor 和 Critic 模型

---

### 失败回滚策略

| 失败场景 | 检测方法 | 回滚动作 |
|---------|---------|---------|
| PyTorch 降级后测试失败 | `pytest tests/` 不通过 | 重新调整依赖版本，检查兼容性 |
| 偏好数据集缺失/格式错误 | 步骤 3 验证失败 | 回退到步骤 1-2，重新生成轨迹 |
| 训练 CUDA 错误 | 日志中出现 CUDA error | 检查 PyTorch 版本，确认降级成功 |
| 训练 OOM | 日志中出现 out of memory | 减少 batch_size 或 gpu_memory_utilization |
| 模型加载失败 | 步骤 3/4 验证失败 | 检查输出目录完整性，重新训练 |
| 评估加载模型失败 | 步骤 5 报错 | 确认步骤 3/4 输出正确 |

---

### 执行顺序总结

```
0. 环境准备（PyTorch 降级）
   ↓ 验证：pytest tests/ 通过
1. 验证步骤 1-2 输出
   ↓ 验证：偏好数据集存在
2. 执行步骤 3（训练 Critic）
   ↓ 验证：Critic 模型可加载
3. 执行步骤 4（训练 Actor）
   ↓ 验证：Actor 模型可加载
4. 数据链完整性检查
   ↓ 验证：所有步骤输出完整
5. 准备步骤 5（评估）
   ↓ 等待：用户确认后执行
```

---

### 预计总耗时

- 步骤 0（PyTorch 降级）：5-10 分钟
- 步骤 3（Critic 训练）：2-5 分钟
- 步骤 4（Actor 训练）：2-5 分钟
- 验证和检查：5 分钟
- **总计**：15-25 分钟（不含步骤 5 评估）

---

### 下一步行动

1. **立即执行**：降级 PyTorch 到 2.3.1+cu118
2. **验证**：运行测试确保兼容性
3. **执行**：按顺序运行步骤 3 → 步骤 4
4. **汇报**：向 team-lead 汇报结果并准备步骤 5

