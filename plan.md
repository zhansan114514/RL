# ACC-Collab 复现计划

## 当前状态

**阶段 1-2 已完成**：核心 bug 修复 + 功能完善 + 架构重构
**阶段 3 进行中**：实验验证

---

## 已完成

### 架构重构 (2026-04-01)

| 改动 | 说明 |
|------|------|
| 统一 extract_answer → src/algorithms/reward.py | 消除两套实现的大小写不一致 |
| ConfigManager 单例 + configs/default.yaml | 集中硬编码参数 |
| 拆分 alternating_train → model_manager + trainer + scheduler | 205 行 God function → 3 个模块 |
| src/algorithms/ 层 | reward, deliberation, rollout, trajectory |
| scripts/train.py | 合并 03/04 脚本 |
| seeding.py | fix_seed 确保可复现 |
| ExperimentLogger JSONL | 无 wandb 也能回溯 metrics |
| 函数命名清晰化 | get_prompt_template, convert_to_hf_dataset, log_dataset_summary |

### 核心问题修复

| # | 问题 | 状态 |
|---|------|------|
| 1 | alternating_train 模型路径 bug | ✅ |
| 2 | MC Roll-out 改为 one-step | ✅ |
| 3 | DPO beta 参数贯穿配置 | ✅ |
| 4 | BBH 按类别分层划分 | ✅ |
| 5 | GUIDED_DELIBERATION_ACTOR 模板 | ✅ |
| 6 | 验证集评估 + early stopping | ✅ |
| 7 | _detect_model_type 去重 | ✅ |

---

## 待完成

### 3.1 中等规模验证 (10 样本 BoolQ)

```bash
CUDA_VISIBLE_DEVICES=5 LD_LIBRARY_PATH=/tmp/nvml_fix:$LD_LIBRARY_PATH \
  python scripts/06_full_pipeline.py --config configs/config.yaml
```

- [ ] 10 样本端到端无报错
- [ ] 偏好对数量 >0
- [ ] 结果文件完整

### 3.2 完整 BoolQ 复现

- [ ] ACC-Collab (1 轮): 全量 BoolQ
- [ ] ACC-Collab+ (2 轮): 全量 BoolQ

### 3.3 多数据集扩展

- [ ] MMLU / BBH / SCIQ / ARC

### 3.4 消融实验

- [ ] 审议轮数 T=1,3,5,7
- [ ] MC 模拟次数
- [ ] reward_threshold
- [ ] LoRA rank

---

## 运行环境

- 硬件: 16x V100-SXM3-32GB
- PyTorch: 2.10.0+cu128（需 NVML fix）
- 运行前执行: `bash scripts/setup_nvml_fix.sh`

---

*189 tests passing | 最后更新 2026-04-01*
