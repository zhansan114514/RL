# ACC-Collab 阶段三代码审查完整报告

**审查日期**: 2026-03-31
**审查团队**: Planner, Engineer, Reviewer
**测试状态**: 189个测试全部通过 ✅

---

## 收到的审查报告

| 团队成员 | 状态 | 报告内容 |
|---------|------|----------|
| Engineer | ✅ 已收到 | 7个发现（配置、GPU、模型加载等） |
| Reviewer | ✅ 已收到 | 7个问题（高/中/低优先级） |
| Critic | ❌ 未收到 | - |
| Tester | ❌ 未收到 | - |

---

## 问题汇总（合并后按严重性排序）

### 🔴 高优先级问题

| # | 问题 | 来源 | 状态 |
|---|------|------|------|
| 1 | preference_pairs 缺少 prompt 字段 | Reviewer | ✅ 已修复 |
| 2 | DPO 训练未使用 NLL 正则化 | Reviewer | ⏸️ 需确认论文 |
| 3 | MC Roll-out 未使用 current_critic_response | Reviewer | ⏸️ 需修复 |
| 4 | logger 未定义（误报） | Reviewer | ✅ 实际已定义 |

### 🟡 中优先级问题

| # | 问题 | 来源 | 状态 |
|---|------|------|------|
| 5 | GPU 设备硬编码 | Engineer | ⏸️ 需配置化 |
| 6 | 重复模型加载导致显存碎片 | Engineer | ⏸️ 需优化 |
| 7 | 交替训练每轮重新生成轨迹效率低 | Reviewer | ⏸️ 需优化 |

### 🟢 低优先级问题

| # | 问题 | 来源 | 状态 |
|---|------|------|------|
| 8 | 配置系统与文档不符 | Engineer | ⏸️ 暂缓 |
| 9 | vLLM CUDA 设备映射 | Engineer | ✅ 已修复 |
| 10 | 模型类型检测可能遗漏 | Engineer | ⏸️ 暂缓 |
| 11 | wrong_answer 生成随机性 | Reviewer | ⏸️ 暂缓 |
| 12 | 测试覆盖率不足 | Reviewer | ⏸️ 暂缓 |

---

## 详细问题分析

### 问题1: preference_pairs 缺少 prompt 字段 ✅

**位置**: `src/trajectory/generator.py:152-173`

**描述**: `generate_trajectories()` 返回的 pairs 缺少 `actor_prompt` 和 `critic_prompt`

**修复**:
```python
# 在遍历 natural_trajectory 时提取 prompt
actor_prompt = round_data["actor_prompt"]
critic_prompt = round_data["critic_prompt"]

# 在构建 preference_pairs 时添加
preference_pairs.append({
    "actor_prompt": actor_prompt,
    "critic_prompt": critic_prompt,
    # ... 其他字段
})
```

**状态**: ✅ 已修复，测试通过

---

### 问题2: MC Roll-out 未使用 current_critic_response ⏸️

**位置**: `src/deliberation/rollouts.py:21-103`

**描述**: `estimate_final_accuracy()` 接收 `current_critic_response` 参数但从未使用

**影响**: one-step roll-out 的上下文不完整，可能影响奖励估计准确性

**修复方案**: 需要在模拟的 actor prompt 中包含当前的 critic 反馈

**状态**: ⏸️ 需要修复（优先级：高）

---

### 问题3: DPO 训练未使用 NLL 正则化 ⏸️

**位置**: `src/training/dpo_trainer.py:92-118`

**描述**: 代码注释声称实现 IRPO 的 NLL 正则化，但 `DPOTrainer` 未配置相关参数

**影响**: 可能与论文方法不一致

**需要确认**: ACC-Collab 论文 Eq.6 是否明确要求 NLL 正则化

**状态**: ⏸️ 需要查阅论文确认

---

### 问题4: logger 未定义（误报）✅

**位置**: `scripts/03_train_critic.py:98`, `scripts/04_train_actor.py:98`

**描述**: logger 变量未定义

**验证**: logger 在脚本第24行已定义 `logger = logging.getLogger(__name__)`

**状态**: ✅ 误报，无需修复

---

### 问题5: GPU 设备硬编码 ⏸️

**位置**: `scripts/06_full_pipeline.py:186-187`

**描述**: 硬编码 GPU 0 和 1

**修复方案**: 通过配置文件指定 `actor_device` 和 `critic_device`

**状态**: ⏸️ 已有参数支持，需配置化

---

### 问题6: 重复模型加载导致显存碎片 ⏸️

**位置**: `src/training/alternating.py:96-158`

**描述**: 每个 iteration 中模型多次加载和释放

**影响**: 效率低，可能 OOM

**状态**: ⏸️ 当前可用，后续优化

---

### 问题7: 交替训练每轮重新生成轨迹效率低 ⏸️

**位置**: `src/training/alternating.py:91-204`

**描述**: 每轮迭代重新运行完整轨迹生成

**影响**: 计算成本高

**状态**: ⏸️ 当前可用，后续优化

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
- [ ] 轨迹生成成功（偏好对>0）
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

**状态**: ✅ 代码审查通过（需修复问题2）

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

**预计耗时**: 2-3天

---

## 建议的修复优先级

### 立即修复（阻塞实验）:
1. ✅ 问题1: prompt 字段缺失 - 已修复
2. ⏸️ 问题2: MC Roll-out 未使用 current_critic_response - 需要修复

### 需要确认:
3. ⏸️ 问题3: DPO NLL 正则化 - 需查阅论文

### 后续优化:
4. ⏸️ 问题5: GPU 设备配置化
5. ⏸️ 问题6: 显存管理优化
6. ⏸️ 问题7: 轨迹生成缓存

---

## 总结

**核心问题**: 问题2（MC Roll-out）需要修复，可能影响奖励估计准确性

**建议**:
1. 先修复问题2，然后运行步骤1验证
2. 确认问题3（DPO NLL）是否必需
3. 其他问题可在后续迭代中优化

**准备状态**: ⚠️ 有1个高优先级问题需修复后才能开始实验
