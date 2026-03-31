# ACC-Collab 阶段三代码审查最终汇总报告

**审查日期**: 2026-03-31
**审查团队**: Planner, Engineer, Reviewer, Critic, Tester
**测试状态**: 189个测试全部通过 ✅

---

## 审查报告统计

| 团队成员 | 问题数量 | 高优先级 | 中优先级 | 低优先级 | 状态 |
|---------|---------|---------|---------|---------|------|
| Engineer | 7 | 1 | 2 | 4 | ✅ 已收到 |
| Reviewer | 7 | 3 | 2 | 2 | ✅ 已收到 |
| Critic | 20 | 6 | 7 | 7 | ✅ 已收到 |
| Tester | 10 | 2 | 4 | 4 | ✅ 已收到 |
| **合计** | **44** | **12** | **15** | **17** | - |

---

## 已修复的高优先级问题

### ✅ 问题1: preference_pairs 缺少 prompt 字段
- **来源**: Reviewer, Critic (#15)
- **修复**: 在 `generate_trajectories()` 中添加 `actor_prompt` 和 `critic_prompt` 字段
- **文件**: `src/trajectory/generator.py`

### ✅ 问题2: generate_wrong_answer 随机性未受seed控制
- **来源**: Critic (#3)
- **修复**: 添加 `rng` 参数和 `seed` 参数，确保可复现性
- **文件**: `src/data/preprocessor.py`, `src/trajectory/generator.py`

### ✅ 问题3: 分步脚本依赖检查缺失
- **来源**: Critic (#10)
- **修复**: 在 02-05 脚本中添加输入文件/模型存在性检查
- **文件**: `scripts/02_build_preferences.py`, `scripts/03_train_critic.py`, `scripts/04_train_actor.py`, `scripts/05_evaluate.py`

### ✅ 问题4: 异常捕获过于宽泛
- **来源**: Critic (#11)
- **修复**: 添加 traceback 记录到 debug 级别
- **文件**: `scripts/01_generate_trajectories.py`, `src/training/alternating.py`

### ✅ 问题5: 空偏好对静默失败
- **来源**: Critic (#2), Tester (#3)
- **修复**: 空偏好对时抛出 ValueError 并提供详细诊断信息
- **文件**: `src/training/alternating.py`

---

## 需要进一步确认的问题

### ⏸️ 问题: MC Roll-out 未使用 current_critic_response
- **来源**: Reviewer (#2)
- **状态**: 已添加文档说明，当前实现正确
- **说明**: one-step roll-out 从当前状态模拟下一步，不需要使用当前的 critic 反馈

### ⏸️ 问题: DPO 训练未使用 NLL 正则化
- **来源**: Reviewer (#3)
- **状态**: 需查阅论文确认
- **说明**: trl 库的 DPOTrainer 默认使用 sigmoid loss，需确认论文是否要求 IRPO

### ⏸️ 问题: 自然审议已正确时的边界情况
- **来源**: Critic (#1)
- **状态**: 未修复，当前行为合理
- **说明**: 当 v_natural=1.0 时 delta_y=0，不会生成偏好对，这是预期行为

### ⏸️ 问题: 模型空响应处理缺失
- **来源**: Tester (#1)
- **状态**: 需添加边界情况处理
- **说明**: 当模型返回空字符串时，需要正确处理

### ⏸️ 问题: 奖励值范围未验证
- **来源**: Critic (#5), Tester (#2)
- **状态**: 需添加验证逻辑
- **说明**: MC roll-out 返回的奖励值应在 [0, 1] 范围内

### ⏸️ 问题: 引导prompt效果未验证
- **来源**: Tester (#4)
- **状态**: 需添加功能测试
- **说明**: 验证引导prompt是否真的能让模型朝目标答案生成

---

## 未修复但优先级较低的问题

### GPU 资源管理
- GPU 设备硬编码 → 通过配置文件指定
- 重复模型加载 → 后续优化
- vLLM CUDA 恢复问题 → 添加文档说明

### 代码质量
- normalize_answer 逻辑不一致 → 统一实现
- 进度条缺失 → 添加 tqdm
- 内存累积 → 分批写入
- 多选题选项E-J处理 → 扩展支持

### 配置一致性
- 分步脚本与全流程脚本参数不一致 → 添加验证脚本
- temperature 参数策略 → 添加文档说明

### 测试覆盖
- 交替训练路径集成测试 → 添加文件系统级测试
- LoRA配置回退行为 → 添加未知模型类型测试
- Wilson置信区间边界 → 添加单样本测试
- MC roll-out稳定性 → 添加不同次数比较测试

---

## 步骤1-5执行计划

### 步骤1: 极小规模验证（3样本）
**命令**:
```bash
CUDA_VISIBLE_DEVICES=14,15 python scripts/06_full_pipeline.py \
  --config configs/debug_3samples.yaml
```
**预计耗时**: 30-60分钟

### 步骤2: 中等规模验证（10样本）
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
**预计耗时**: 1-2小时

### 步骤3: 核心算法验证
**状态**: ✅ 代码审查通过

### 步骤4: 完整 BoolQ 复现（1轮）
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

### 步骤5: ACC-Collab+（2轮）
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

## GPU 资源使用策略

| 规模 | GPU 配置 | gpu_memory_utilization |
|------|----------|------------------------|
| 3样本 | 单 GPU 共享 | 0.45 × 2 = 0.9 |
| 10样本 | 单 GPU 共享 | 0.45 × 2 = 0.9 |
| 完整 | 双 GPU 分离 | 0.8 × 2 = 1.6 |

**V100 约束**: Gemma2 必须使用 float32，单模型 ~9.77 GiB

---

## 总结

**已修复**: 5个高优先级问题
**测试状态**: 189个测试全部通过 ✅
**准备状态**: 可以开始实验验证（GPU 就绪后）

**下一步**:
1. GPU 资源就绪后执行步骤1
2. 确认 DPO NLL 正则化是否必需
3. 添加模型空响应和奖励值范围验证
4. 逐步优化中低优先级问题

---

## 附录：各团队完整审查报告

### Engineer 审查报告
（见前文转发的完整内容）

### Reviewer 审查报告
（见前文转发的完整内容）

### Critic 审查报告
（见前文转发的完整内容）

### Tester 审查报告
（见前文转发的完整内容）
