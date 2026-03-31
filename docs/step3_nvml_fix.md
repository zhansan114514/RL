# 步骤3 NVML错误修复记录

**修复时间**: 2026-03-31
**问题**: V100上训练Critic时遇到NVML错误

---

## 问题描述

训练Critic时遇到NVML错误，这是V100上DataParallel相关的已知问题。

---

## 修复措施

### 1. dpo_trainer.py 添加 import os
**文件**: `src/training/dpo_trainer.py`
**修改**: 添加 `import os` 确保文件操作可用

### 2. 脚本03/04添加CUDA_VISIBLE_DEVICES设置
**文件**: `scripts/03_train_critic.py`, `scripts/04_train_actor.py`
**修改**: 在运行时设置 `CUDA_VISIBLE_DEVICES=0`

---

## 当前状态

| 步骤 | 状态 | 结果 |
|------|------|------|
| 1. 生成轨迹 | ✅ 完成 | 3样本 → 22偏好对 |
| 2. 构建偏好 | ✅ 完成 | actor 22对 + critic 22对 |
| 3. 训练 Critic | 🔄 修复后重新运行 | - |
| 4. 训练 Actor | ⏳ 等待中 | - |
| 5. 评估 | ⏳ 等待中 | - |

---

## 下一步监控

### 步骤3重新运行（当前）
- **预计时间**: 10-15分钟
- **验证点**:
  - [ ] 训练启动成功
  - [ ] loss 下降
  - [ ] 模型保存成功

### 如果步骤3成功
→ 立即执行步骤4（训练Actor）

### 如果步骤3再次失败
→ 检查错误日志 → 调整参数 → 重试

---

## 可能的后续问题

### 如果CUDA问题持续
**应对策略**:
1. 使用单GPU模式（已设置）
2. 减少 batch_size 到 1
3. 检查GPU内存使用情况

### 如果训练时间过长
**应对策略**:
1. 减少 gradient_accumulation_steps
2. 减少 max_length
3. 减少 num_epochs（当前已是1）

---

## 准备下一步

步骤3成功后，步骤4（训练Actor）预计需要：
- 时间：10-15分钟
- GPU：1x V100 (≥20GB)
- 输入：actor_preferences 数据集

步骤5（评估）预计需要：
- 时间：5-10分钟
- GPU：2x V100 (各≥10GB)
- 输入：训练后的 actor + critic 模型
