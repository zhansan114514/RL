# Diverse Actor-Critic Society — 实现完成报告

> 结合 ACC-Collab（Actor-Critic 协作）与 Multiagent FT（多样化思维链）的实验复现。
> 基座模型：Qwen2.5-7B-Instruct | 硬件：单张 H100 80GB | 每个 Agent 独立 LoRA (r=256)

---

## 架构

```
3 Actors (推理风格分化)                    4 Critics (错误类型专长)
├── actor_algebraic  (代数推理)            ├── critic_arithmetic     (算术错误)
├── actor_direct     (直接计算)            ├── critic_logic         (逻辑错误)
└── actor_backtracking (反推验证)          ├── critic_hallucination (幻觉错误)
                                          └── critic_verification  (验证错误)
         │                                         │
         └──────── MoE Critic Router ──────────────┘
              softmax(confidence) → Top-K=2 → 加权拼接反馈
```

推理流程：所有 Actor 并行生成候选 → 所有 Critic 评估 → Router 组合反馈 → Actor 修正 → 多数投票共识

---

## 新增/修改文件清单

### 修改的现有文件（6个）

| 文件 | 修改内容 |
|------|----------|
| `src/training/lora_config.py` | `MODEL_TARGET_MODULES` 添加 `"qwen2.5"` 键 |
| `src/utils/model_utils.py` | `detect_model_type()` 添加 `"qwen2.5"` 分支（在 `"qwen3"` 之前） |
| `src/data/loader.py` | `DATASET_REGISTRY` 添加 MATH (`hendrycks/math`) 和 GSM8K (`openai/gsm8k`) |
| `src/data/preprocessor.py` | 添加 `math` task_type 处理、`\boxed{}` / `####` 解析、`generate_wrong_answer(task_type="math")` |
| `src/prompts/templates.py` | 添加 `MATH_TEMPLATES` 和 `GSM_TEMPLATES`（各 6 种 PromptType），注册到 `DATASET_TEMPLATES` |
| `src/algorithms/reward.py` | 添加 `_extract_math()` 函数、`extract_answer()` 支持 `task_type="math"` |

### 新建 src/society/ 模块（8个文件）

| 文件 | 核心类/函数 | 说明 |
|------|-------------|------|
| `__init__.py` | 公共 API 导出 | 模块入口 |
| `agent_registry.py` | `AgentRegistry`, `AgentConfig`, `ReasoningStyle`, `ErrorType` | Agent 身份管理，JSON 持久化，`create_default()` 创建 3A+4C |
| `router.py` | `CriticRouter`, `parse_confidence`, `build_critic_feedback` | MoE 置信度路由，softmax + Top-K，无训练参数 |
| `data_classifier.py` | `classify_reasoning_style`, `classify_error_type`, `DataClassifier` | GLM4.5 API 分类 + 本地 JSON 缓存 + 启发式降级 |
| `multi_deliberation.py` | `multi_agent_deliberate_single_gpu` | 单 GPU 顺序审议，原子磁盘持久化，崩溃恢复 |
| `diversity_split.py` | `DiversitySplit` | 按推理风格/错误类型切分数据，可选均衡化 |
| `society_trainer.py` | `society_alternating_train` | N×M 交替训练调度，断点续传 |
| `inference_pipeline.py` | `society_inference`, `run_ablation` | 3 种投票策略 + A1-A5 消融实验配置 |

### 新建配置文件（8个）

| 文件 | 说明 |
|------|------|
| `configs/data/math.yaml` | MATH 数据集 (hendrycks/math, task_type=math) |
| `configs/data/gsm.yaml` | GSM8K 数据集 (openai/gsm8k, task_type=math) |
| `configs/model/qwen2.5_7b.yaml` | Qwen2.5-7B-Instruct 模型配置 |
| `configs/society/base.yaml` | Society 基础配置（3A+4C, LoRA r=256, 审议5轮） |
| `configs/society/actors.yaml` | 3 个 Actor 定义（algebraic, direct, backtracking） |
| `configs/society/critics.yaml` | 4 个 Critic 定义（arithmetic, logic, hallucination, verification） |
| `configs/society/router.yaml` | MoE Router 配置（top_k=2, temperature=1.0） |
| `configs/society/experiment_h100.yaml` | 单 H100 完整实验配置（6 个 step） |

### 新建脚本（6个）

| 脚本 | Phase | 说明 |
|------|-------|------|
| `scripts/07_bootstrap_actors.py` | Phase 1 | Bootstrap 数据生成：N=5 独立响应 + M=2 轮辩论 |
| `scripts/08_classify_data.py` | Phase 2 | 数据分类：GLM4.5 API 推理风格 + 错误类型 |
| `scripts/09_diversify_actors.py` | Phase 3 | Actor 分化训练：按风格子集 DPO 训练 3 个 Actor |
| `scripts/10_diversify_critics.py` | Phase 4 | Critic 分化训练：按错误类型子集 DPO 训练 4 个 Critic |
| `scripts/11_society_train.py` | Phase 5 | Society 交替训练：N×M 交替 DPO |
| `scripts/12_society_evaluate.py` | Phase 6 | 评估 + 消融实验：A1-A5 + Wilson 95% CI |

### 新建测试（2个文件）

| 文件 | 说明 |
|------|------|
| `tests/test_society.py` | 85 个测试用例覆盖全部 society 模块 |
| `tests/test_qwen_gsm.py` | Qwen2.5 LoRA 配置 + MATH/GSM 答案提取测试 |

---

## 关键设计决策

1. **ReasoningStyle vs ThinkingStyle**: 实验计划指定 3 种推理风格（ALGEBRAIC, DIRECT, BACKTRACKING），而非 Multiagent FT 原文的 6 种通用风格。这样每个 Actor 的差异更显著。

2. **ErrorType 4 分类**: Critic 按错误类型专长分化（ARITHMETIC, LOGIC, HALLUCINATION, VERIFICATION），而非通用思维风格。错误类型来自实验计划的分类 Prompt。

3. **MoE Router 无训练参数**: Critic 输出的 `[Confidence: X]` 是天然的门控信号，无需额外训练路由网络。正则匹配 `\ [Confidence:\s*([0-9.]+)\]`，失败时降级为均匀权重。

4. **数据级分化**: Multiagent FT 论文（Figure 3）表明数据级分化足以保持多样性，不需要显式多样性 Loss。

5. **API 分类 vs 启发式**: 推理风格和错误类型需要语义理解，GLM4.5 API 准确率远高于正则匹配。降级策略确保 API 不可用时管线不中断。

6. **单 GPU 管理**: 7 个 Agent 各自独立 LoRA，推理时顺序加载/卸载，中间结果持久化到磁盘。原子写入（tmp + rename）保证崩溃恢复。

---

## 测试结果

```
tests/test_society.py:    85 passed
tests/test_qwen_gsm.py:   tests passed
tests/ (全部):             363 passed, 0 failed, 0 errors
```

---

## 消融实验配置

| 编号 | 配置 | 验证什么 |
|------|------|----------|
| A1 | 1 Actor + 1 Critic | 原始 ACC-Collab 基线 |
| A2 | 3 Actor + 1 Critic | Actor 多样性单独贡献 |
| A3 | 1 Actor + 4 Critic + Router | Critic 专长化单独贡献 |
| A4 | 3 Actor + 4 Critic, 均匀权重 | 完整系统但无路由 |
| A5 | 3 Actor + 4 Critic + Router | 完整系统 |

预期：A5 > A1（完整系统优于单 Agent 基线）
