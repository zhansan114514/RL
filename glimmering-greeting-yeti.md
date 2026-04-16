# Diverse Actor-Critic Society: 实验改进方案

## Context

**问题**: ACC-Collab 训练一个 Actor + 一个 Critic 协作解决推理任务，但单个 Actor 容易陷入单一推理模式，单个 Critic 纠错能力有限。Multiagent FT 通过数据级分化让多个 Agent 保持多样性，但没有 Actor-Critic 的精细引导能力。

**目标**: 结合两篇论文的优势，构建 "Diverse Actor-Critic Society"——3 个风格各异的 Actor + 4 个专长不同的 Critic，通过 MoE 路由机制在审议时动态激活，训练出既保持多样化思维链、又具备精细协作能力的社会。

**硬件约束**: 单张 H100 80GB，Qwen2.5-7B-Instruct 基座模型，每个 Agent 独立 LoRA。

**数据集**: 新增 MATH (Hendrycks et al.) 和 GSM8K (Cobbe et al.) 支持，与 Multiagent FT 论文对齐。

**分类方法**: 使用强模型 API（如 Qwen2.5-72B / GPT-4）做 prompt-based 分类，准确率高。

---

## 系统架构总览

```
                         ┌──────────────────────────────────┐
                         │    Qwen2.5-7B-Instruct (共享底座)     │
                         └──────────┬───────────────────────┘
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
    ┌─────────▼──────┐   ┌─────────▼──────┐   ┌─────────▼──────┐
    │  Actor 1       │   │  Actor 2       │   │  Actor 3       │
    │  代数推理      │   │  直接计算      │   │  反推验证      │
    │  LoRA-A1       │   │  LoRA-A2       │   │  LoRA-A3       │
    └────────┬───────┘   └────────┬───────┘   └────────┬───────┘
             │                    │                     │
             └────────────┬───────┴─────────────────────┘
                          │ 所有 Actor 候选响应
                          ▼
              ┌───────────────────────┐
              │   MoE Critic Router   │
              │   (置信度加权激活)     │
              └───┬───┬───┬───┬───────┘
                  │   │   │   │
     ┌────────────▼┐ ┌▼───┐ ┌▼───────┐ ┌──────────▼┐
     │ Critic 1    │ │C2  │ │C3      │ │Critic 4   │
     │ 算术错误    │ │逻辑│ │幻觉    │ │验证错误    │
     │ LoRA-C1     │ │错误│ │错误    │ │LoRA-C4     │
     └─────────────┘ │LoRA│ │LoRA-C3 │ └────────────┘
                     │-C2 │ └────────┘
                     └─────┘
```

### 推理阶段数据流

```
1. 所有 Actor 并行生成候选 (单GPU顺序执行)
2. 每个 Critic 输出 "置信度 + 专项反馈"
3. Router: softmax 置信度 → Top-K 激活 → 加权拼接反馈
4. Actor 接收路由反馈 → 下一轮修正
5. 最终: 所有 Actor 投票 → 共识答案
```

---

## 训练管线

### Phase 0: 基础设施 (1-2天)

#### 0.1 添加 Qwen2.5 + MATH/GSM 支持

**修改文件:**
- `src/training/lora_config.py` — 在 `MODEL_TARGET_MODULES` 中添加 `"qwen2.5"` 键（目标模块与其他架构相同：q/k/v/o/gate/up/down_proj）
- `src/utils/model_utils.py` — 在 `detect_model_type()` 中添加 `"qwen2.5"` 分支（需放在 `"qwen3"` 之前）

**新增 MATH/GSM 数据集支持:**

- `src/data/loader.py` — 在 registry 中添加 MATH 和 GSM8K 的 HuggingFace 数据集配置:
  ```python
  # MATH: hendrycks/math (MATH subset, level 1-3 for difficulty)
  # GSM8K: openai/gsm8k (main configuration)
  ```
- `src/data/preprocessor.py` — 添加 MATH/GSM 的 `standardize_sample()` 处理:
  - MATH/GSM 属于 `task_type="math"`，答案格式为 `\boxed{...}`
  - 需要实现 `extract_math_answer()` 解析 `\boxed{}` 格式
- `src/prompts/templates.py` — 添加 `MATH_TEMPLATES` 和 `GSM_TEMPLATES`，6 种 PromptType 的数学版本:
  ```python
  MATH_TEMPLATES = {
      PromptType.SINGLE_SHOT: (
          "You will be given a math problem. "
          "Solve it step by step and provide your final answer "
          "in the form \\boxed{answer}.\n"
          "Problem: {question}"
      ),
      # ... 其他 5 种模板
  }
  GSM_TEMPLATES = MATH_TEMPLATES  # GSM 共享 MATH 模板结构
  ```
- `src/algorithms/reward.py` — 扩展 `extract_answer()` 支持 `task_type="math"`，解析 `\boxed{}` 格式
- `configs/data/math.yaml` — MATH 数据集配置
- `configs/data/gsm.yaml` — GSM8K 数据集配置

**MATH/GSM 参考实现**: Multiagent FT 仓库的 `grader.py` 和 `math_normalize.py` 提供了答案解析和归一化的成熟实现，可直接复用。

#### 0.2 创建 Society 模块

**新建目录: `src/society/`**

```
src/society/
├── __init__.py              # 导出公共 API
├── agent_registry.py        # Agent 身份、LoRA路径、元数据管理
├── router.py                # MoE 置信度路由器
├── multi_deliberation.py    # 多Actor多Critic审议引擎
├── data_classifier.py       # 推理风格 + 错误类型分类器
├── diversity_split.py       # 数据级分化：按风格/错误类型切分
├── society_trainer.py       # 多Agent交替训练调度器
└── inference_pipeline.py    # 生产推理管线
```

**新建配置: `configs/society/`**

```
configs/society/
├── base.yaml                # 全局默认（7个Agent，top_k=2，LoRA r=256）
├── actors.yaml              # 3个Actor定义（风格标签、prompt后缀）
├── critics.yaml             # 4个Critic定义（错误类型标签、置信度prompt）
├── router.yaml              # 路由器配置（top_k、温度、置信度正则）
└── experiment_h100.yaml     # 单H100实验配置
```

---

### Phase 1: Bootstrap 数据生成 (2-3天)

**核心思想**: 沿用 Multiagent FT 的方法——用同一基座模型的高温采样模拟多Agent辩论，生成多样化初始轨迹。

**新建脚本: `scripts/07_bootstrap_actors.py`**

执行流程:
1. 加载 Qwen2.5-7B-Instruct 基座模型（单 VLLMInference，无 LoRA）
2. 对每个训练样本，用 temperature=0.8 + 不同 seed 生成 N=5 条独立响应
3. 模拟 M=2 轮辩论：每轮每个 Agent 看到其他 Agent 的响应后生成新响应
4. 多数投票确定共识答案
5. 保存所有轨迹（含每轮每Agent的响应、共识答案、正确性标记）

**复用**:
- `src/inference/vllm_server.py::VLLMInference` — 推理引擎
- `src/algorithms/reward.py::extract_answer()` — 答案抽取
- `src/data/loader.py::load_dataset()` — 数据加载

输出: `cache/society/bootstrap/trajectories.jsonl`

---

### Phase 2: 数据分类 (1-2天)

**核心思想**: 使用强模型 API 对轨迹进行 prompt-based 分类，确保推理风格和错误类型的标注准确。

**新建脚本: `scripts/08_classify_data.py`**
**新建模块: `src/society/data_classifier.py`**

#### API 分类接口设计

```python
def classify_via_api(
    response: str,
    question: str,
    correct_answer: str,
    classification_type: str,  # "style" or "error_type"
    api_config: dict,          # {provider, model, api_key, ...}
) -> str:
    """
    使用强模型 API 进行分类。

    支持: Anthropic (Claude)
    """
```

### API使用

使用glm4.5来进行分类。API如下：
url：https://open.bigmodel.cn/api/anthropic
API_KEY:bcf988da32f64948a82fd7dda3b9b3d3.mVYoCk3Wi5ZrcsUM

#### 推理风格分类 Prompt (针对正确响应)

```
Given a math problem and a correct solution, classify the reasoning style:

Problem: {question}
Solution: {response}
Correct Answer: {answer}

Classify into exactly one category:
- ALGEBRAIC: Uses symbolic manipulation, equations, variables (e.g., "let x =", solving systems)
- DIRECT: Direct step-by-step numerical computation without symbolic setup
- BACKTRACKING: Starts with an attempt, verifies it, then revises if needed

Respond with only the category name.
```

#### 错误类型分类 Prompt (针对错误响应)

```
Given a math problem, an incorrect solution, and the correct answer,
classify the primary error type:

Problem: {question}
Incorrect Solution: {response}
Extracted Answer: {extracted_answer}
Correct Answer: {correct_answer}

Classify into exactly one category:
- ARITHMETIC: Correct reasoning approach but numerical calculation mistake
- LOGIC: Flawed reasoning chain, wrong formula, or logical fallacy
- HALLUCINATION: Fabricated numbers, wrong theorem, or unsupported claims
- VERIFICATION: Attempted self-check but failed to catch the error

Respond with only the category name.
```

#### API 调用策略

- **批处理**: 每次请求包含 10-20 个样本（减少 API 调用次数）
- **缓存**: 分类结果缓存到本地 JSON，避免重复调用
- **降级策略**: 如 API 不可用
- **成本控制**: 预计 500 样本 x 5 响应 = 2500 次分类

---

### Phase 3: Actor 分化训练 (2-3天)

**核心思想**: Multiagent FT 的数据级分化——每个 Actor 只在自己的风格子集上训练，多样性自然涌现。

**新建脚本: `scripts/09_diversify_actors.py`**

对 3 个 Actor 中的每一个:
1. 加载基座模型作为 VLLMInference
2. 用该 Actor 的数据子集构建 DPO 偏好对:
   - chosen: 正确的、属于该风格的响应
   - rejected: 同一样本的其他错误响应或不同风格的响应
3. 清理 vLLM，运行 DPO 训练（复用 `train_dpo()`）
4. 合并 LoRA → 基座模型
5. 保存到 `cache/society/actors/{agent_id}/`

**复用**:
- `src/training/dpo_trainer.py::train_dpo()` — DPO 训练子进程
- `src/training/lora_config.py::get_lora_config()` — LoRA 配置
- `src/trajectory/preference.py::build_preference_dataset()` — 偏好数据构建

**关键配置**:
```yaml
lora_r: 256
lora_alpha: 512
learning_rate: 5e-5
beta: 0.1
num_epochs: 1
```

---

### Phase 4: Critic 分化训练 (2-3天)

**核心思想**: 每个 Critic 只在包含其专长错误类型的纠错轨迹上训练。

**新建脚本: `scripts/10_diversify_critics.py`**

对 4 个 Critic 中的每一个:
1. 加载第一个已分化的 Actor + 基座模型作为 VLLMInference
2. 运行审议（复用 `deliberate()`）生成轨迹
3. 对错误响应分类错误类型
4. 筛选属于该 Critic 错误类型的轨迹
5. 构建 DPO 偏好对:
   - chosen: Actor 犯错 + Critic 正确引导修正的反馈
   - rejected: Actor 犯错 + Critic 无效/误导的反馈
6. 清理 vLLM，运行 DPO 训练
7. 保存到 `cache/society/critics/{agent_id}/`

**Critic 置信度 Prompt**: 每个 Critic 的 prompt 后缀要求输出 `[Confidence: X]`，为后续 MoE 路由提供信号。

**复用**:
- `src/algorithms/deliberation.py::deliberate()` — 审议引擎
- `src/algorithms/trajectory.py::generate_trajectories()` — Algorithm 1
- `src/training/dpo_trainer.py::train_dpo()` — DPO 训练

---

### Phase 5: Society 交替训练 (3-4天)

**核心思想**: 将 ACC-Collab 的交替训练从 1 Actor + 1 Critic 泛化为 N Actor + M Critic。

**新建模块:**
- `src/society/router.py` — MoE 置信度路由器
- `src/society/multi_deliberation.py` — 多Agent审议引擎
- `src/society/society_trainer.py` — 多Agent交替训练调度器

**新建脚本: `scripts/11_society_train.py`**

#### MoE 路由器设计 (无训练参数)

```
输入: 所有 Critic 的 (置信度, 反馈文本)
处理:
  1. 对置信度做 softmax(temperature=1.0)
  2. 选择 Top-K (默认 K=2) 个 Critic
  3. 按权重拼接反馈
输出: 加权反馈文本 + 路由权重
```

置信度解析: 正则匹配 `\[Confidence:\s*([0-9.]+)\]`，失败时降级为均匀权重。

#### 多Agent审议引擎 (单GPU顺序执行)

```
multi_deliberate_single_gpu():
  for round t in range(num_rounds):
    # Step 1: 所有 Actor 生成响应 (顺序加载/卸载)
    for actor in actors:
      load(actor_lora) → generate_response() → unload()
      persist to cache/society/deliberation/{sample_id}/round_{t}/

    # Step 2: 所有 Critic 评估每个 Actor 的响应
    for actor_response in all_actor_responses:
      for critic in critics:
        load(critic_lora) → generate_feedback() → unload()
        persist feedback + confidence

    # Step 3: Router 组合反馈
    route(all_feedbacks) → routed_feedback per actor

    # Step 4: 多数投票
    majority_vote(all_actor_answers) → consensus
```

**崩溃恢复**: 每个模型调用后持久化到磁盘（JSON），重启时跳过已完成的步骤。每个文件原子写入（先写临时文件再 rename）。

#### 交替训练调度

```
society_alternating_train():
  for iteration in range(num_iterations):
    # Phase A: 训练所有 Critic (固定所有 Actor)
    for critic_c in critics:
      1. 加载一个 Actor (参考) + critic_c → 生成轨迹
      2. 分类错误类型 → 筛选 critic_c 的专长数据
      3. 清理模型 → DPO 训练 critic_c

    # Phase B: 训练所有 Actor (固定所有 Critic)
    for actor_a in actors:
      1. 加载 actor_a + 一个 Critic (参考) → 生成轨迹
      2. 分类推理风格 → 筛选 actor_a 的风格数据
      3. 清理模型 → DPO 训练 actor_a
```

**复用**:
- `src/training/scheduler.py::alternating_train()` — 训练模式参考
- `src/training/trainer.py::generate_trajectory_data()` — 轨迹生成
- `src/training/model_manager.py::create_inference_model()` — 单模型加载
- `src/training/model_manager.py::cleanup_models()` — GPU 内存释放

---

### Phase 6: 评估 (1-2天)

**新建模块: `src/society/inference_pipeline.py`**
**新建脚本: `scripts/12_society_evaluate.py`**

#### 评估策略

| 策略 | 描述 |
|------|------|
| `all_actors_majority` | 所有 Actor 生成，多数投票 |
| `best_actor` | 只用最高置信度的 Actor |
| `weighted` | 按 Actor 置信度加权 |

#### 消融实验 (关键验证)

| 编号 | 配置 | 验证什么 |
|------|------|----------|
| A1 | 1 Actor + 1 Critic (原始 ACC-Collab) | 基线 |
| A2 | 3 Actor + 1 Critic | Actor 多样性单独贡献 |
| A3 | 1 Actor + 4 Critic + Router | Critic 专长化单独贡献 |
| A4 | 3 Actor + 4 Critic，均匀权重 | 完整社会但无路由 |
| A5 | 3 Actor + 4 Critic + Router | 完整系统 |

#### 指标

- 逐轮准确率 (初始 → 最终)
- 改进率 (initial_wrong → final_correct)
- 共识准确率 (多数投票 vs 单Agent)
- 多样性指标: Embedding 相似度、共识分歧度
- Wilson 95% 置信区间

---

## 关键设计决策

### 1. 为什么不用显式多样性 Loss？

Multiagent FT 论文表明数据级分化已足以保持多样性（Figure 3 显示多样性指标在多轮微调中保持甚至提升）。添加显式 Loss 会增加超参敏感性和实现复杂度，且两篇论文均未使用。

### 2. 为什么用强模型 API 分类？

- **准确率高**: 数学推理风格和错误类型的判断需要语义理解，正则匹配不够可靠
- **成本可控**: ~2500 次 API 调用约 $1-3（GPT-4o-mini），远低于训练成本
- **可降级**: API 不可用时自动降级为启发式正则，保证管线不中断

### 3. 为什么 Router 不用可训练参数？

- 避免额外的训练管线和超参
- Critic 输出的置信度已是天然的门控信号
- 如初步实验表明均匀权重已足够好，Router 可简化或移除

### 4. 单 GPU 如何管理 7 个 Agent？

- 每个 Agent 独立 LoRA，训练后合并到基座
- 推理时顺序加载/卸载（~30-60秒/次切换）
- 中间结果持久化到磁盘，支持崩溃恢复
- 预计全流程 ~17-22 小时（200 样本，5 轮审议，1 轮迭代）

---

## 需要修改的现有文件 (最小化)

| 文件 | 修改内容 |
|------|----------|
| `src/training/lora_config.py:29` | `MODEL_TARGET_MODULES` 添加 `"qwen2.5"` 键 |
| `src/utils/model_utils.py` | `detect_model_type()` 添加 `"qwen2.5"` 分支 |
| `src/data/loader.py` | registry 添加 MATH (`hendrycks/math`) 和 GSM8K (`openai/gsm8k`) |
| `src/data/preprocessor.py` | 添加 `task_type="math"` 的 `standardize_sample()` 处理，实现 `\boxed{}` 解析 |
| `src/prompts/templates.py` | 添加 `MATH_TEMPLATES` 和 `GSM_TEMPLATES`（6 种 PromptType），注册到 `DATASET_TEMPLATES` |
| `src/algorithms/reward.py` | 扩展 `extract_answer()` 支持 `task_type="math"`，解析 `\boxed{}` |
| `configs/data/math.yaml` | 新建 MATH 数据集配置 |
| `configs/data/gsm.yaml` | 新建 GSM8K 数据集配置 |

其余所有代码通过新建 `src/society/` 模块和 `scripts/07-12` 实现，不修改现有功能。

---

## 完整文件清单

### 新建文件

```
src/society/__init__.py
src/society/agent_registry.py        # AgentConfig, AgentRegistry, ReasoningStyle, ErrorType
src/society/router.py                # CriticRouter, CriticFeedback, RoutedFeedback
src/society/multi_deliberation.py    # multi_deliberate(), multi_deliberate_single_gpu()
src/society/data_classifier.py       # classify_reasoning_style(), classify_error_type()
src/society/diversity_split.py       # split_by_reasoning_style(), split_by_error_type()
src/society/society_trainer.py       # society_alternating_train()
src/society/inference_pipeline.py    # society_inference()

configs/society/base.yaml
configs/society/actors.yaml
configs/society/critics.yaml
configs/society/router.yaml
configs/society/experiment_h100.yaml

scripts/07_bootstrap_actors.py
scripts/08_classify_data.py
scripts/09_diversify_actors.py
scripts/10_diversify_critics.py
scripts/11_society_train.py
scripts/12_society_evaluate.py

tests/test_society.py                # 单元测试
tests/test_society_integration.py    # 集成测试
```

### 修改文件

```
src/training/lora_config.py          # +1 行：添加 "qwen2.5" 键
src/utils/model_utils.py             # +3 行：添加 qwen2.5 检测
src/data/loader.py                   # +MATH/GSM registry entries
src/data/preprocessor.py             # +math task_type support
src/prompts/templates.py             # +MATH_TEMPLATES, GSM_TEMPLATES
src/algorithms/reward.py             # +math answer extraction
configs/data/math.yaml               # 新建
configs/data/gsm.yaml                # 新建
```

---

## 验证计划

### 每阶段验证

| 阶段完成 | 验证命令 |
|----------|----------|
| Phase 0 | `pytest tests/test_society.py::TestAgentRegistry -v`；加载 Qwen2.5 LoRA 配置成功 |
| Phase 1 | `cache/society/bootstrap/` 有数据，3 种风格均有样本 |
| Phase 2 | 分类后每个子集 > 20 个样本，分布不过度偏斜 |
| Phase 3 | 3 个 Actor LoRA 存在，对同一 prompt 生成不同风格响应 |
| Phase 4 | 4 个 Critic LoRA 存在，反馈聚焦于各自专长领域 |
| Phase 5 | 训练 loss 下降，Registry YAML 更新 |
| Phase 6 | A5 > A1 (完整系统优于单 Agent 基线) |

### 端到端测试

```bash
# 小规模验证（5 样本，2 轮审议）
python scripts/07_bootstrap_actors.py --config configs/society/experiment_h100.yaml --max_samples 5
python scripts/08_classify_data.py --config configs/society/experiment_h100.yaml
python scripts/09_diversify_actors.py --config configs/society/experiment_h100.yaml
python scripts/10_diversify_critics.py --config configs/society/experiment_h100.yaml
python scripts/11_society_train.py --config configs/society/experiment_h100.yaml --max_samples 5 --num_rounds 2
python scripts/12_society_evaluate.py --config configs/society/experiment_h100.yaml --max_samples 5

# 全量运行
python scripts/12_society_evaluate.py --config configs/society/experiment_h100.yaml
```

---

每个 Phase 完成后立即运行对应验证，发现问题及时修复。在实验出结果分析的时候，可以print打印出几个结果，看看训练之后actor的回答。
