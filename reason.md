# Phase 2 分类失败原因分析

## 背景

本次实验使用 `configs/society/experiment_large.yaml` 的 1000 样本配置，Phase 1 已生成 Actor SFT 候选数据，Phase 2 负责对这些候选回答进行正确性检查和 reasoning style 分类。Phase 2 最终运行完成，并生成了以下产物：

- `output/society_mmlu_large_n1000_style3_qwen3/classified/classified_data.json`
- `output/society_mmlu_large_n1000_style3_qwen3/classified/classification_report.json`
- `output/society_mmlu_large_n1000_style3_qwen3/classified/actor_sft_candidate_report.json`
- `output/society_mmlu_large_n1000_style3_qwen3/confirm/phase02_actor_sft_data_quality.json`

但 Phase 2 的质量确认脚本判定为 `fail`，因此不能直接进入 Phase 3 Actor SFT 训练。

## Phase 2 流程

Phase 2 的入口是 `scripts/08_classify_data.py`，核心流程如下：

1. 读取 Phase 1 生成的 `bootstrap/trajectories.jsonl`，本次共有 1000 条轨迹。
2. 每条轨迹包含 12 个候选回答：3 种 reasoning style（direct、evidence、elimination）乘以 4 个 temperature。
3. 对每个候选回答先做本地答案正确性检查。
4. 只有答案正确的回答才调用 GLM API 进行 reasoning style 分类。
5. 每个候选回答最终会被标注：
   - 是否答案正确；
   - 是否可作为 Actor SFT 训练样本；
   - API 分类得到的 `primary_style`；
   - 分类 style 是否与生成时的 `prompted_style` 一致；
   - 是否通过 `accepted_for_actor_sft` 门槛。

`accepted_for_actor_sft` 需要同时满足：答案正确、回答可训练、分类 style 与 prompted style 一致、分类置信度达到阈值。

## 运行层面的分类失败

本次 Phase 2 的 API 分类统计如下：

- 分类尝试数：9660
- 分类失败数：13
- 分类失败率：0.0013457556935817805
- 配置阈值：0.011

因此，API 层面的失败率低于阈值，不是 Phase 2 质量失败的主要原因。

API 层面的失败主要来自 GLM 服务：

- `400 code=1301`：GLM 内容过滤。被过滤的样本本身是正常 MMLU 题，例如 Buddhism、Cold War 等题目，但 GLM 判定输入可能敏感。
- `429 code=1302`：GLM 速率限制，脚本会自动重试。
- `500` 和 timeout：GLM 服务端偶发错误或长尾请求，脚本会自动重试。
- 早期还出现过 GLM 返回 fenced JSON 且 `evidence` 字段里含未转义双引号，导致 JSON 解析失败；该解析问题已经在 `src/society/data_classifier.py` 中修复，并添加了回归测试。

## 质量门失败现象

Phase 2 确认脚本 `confirm/02_check_actor_sft_data.py` 的结果显示，失败点集中在 style 数据质量，而不是 API 可用性。

关键指标如下：

| style | generated | correct | style_matched | usable |
| --- | ---: | ---: | ---: | ---: |
| direct | 4000 | 3328 | 36 | 19 |
| evidence | 4000 | 3191 | 704 | 678 |
| elimination | 4000 | 3141 | 2823 | 2661 |

失败的质量门：

- `usable_examples_direct`: 19，阈值是 >= 256
- `subject_coverage_direct`: 6，阈值是 >= 55
- `subject_coverage_evidence`: 53，阈值是 >= 55
- `style_imbalance_ratio`: 140.0526，阈值是 <= 1.5

其中最严重的是 direct 风格。虽然 direct 生成了 4000 个候选，且正确回答有 3328 个，但最终可用于 Actor SFT 的 direct 样本只有 19 个。

## 根因分析

根因是 Phase 1 的 Actor 生成提示和 Phase 2 的 style 分类规则不一致。

Phase 2 的分类器在 `src/society/data_classifier.py` 中将三种风格定义为：

- direct：简短、最小解释，不系统引用证据，不比较选项；
- evidence：明确引用事实、定义、概念、题干线索或领域知识；
- elimination：比较选项、排除错误选项，或解释为什么一个选项优于其他选项。

分类器还包含高精度规则，例如：

- 如果回答包含 `Direct reason:` 且没有比较选项，则判为 direct；
- 如果包含 `Key evidence:` 或 `Application:`，则倾向判为 evidence；
- 如果包含 `Option analysis:`、`Elimination:` 或明显排除选项，则判为 elimination。

但 Phase 1 的 Actor prompt 在 `src/prompts/actor_prompts.py` 中对 direct 的约束较弱，只要求：

- 使用最短充分推理；
- 避免不必要讨论；
- 聚焦最直接路线。

同时统一的最终回答指令仍要求模型自然地给出 reasoning。Qwen3-14B 在 MMLU 多选题上即使收到 direct prompt，也倾向生成较长解释，常常引用事实、定义或比较选项。于是这些本来由 direct Actor 生成的正确回答，在 Phase 2 被分类为 evidence 或 elimination。

实际混淆矩阵显示：

- direct prompted 且正确的回答中：
  - 1644 个被判为 evidence；
  - 1645 个被判为 elimination；
  - 只有 36 个被判为 direct；
  - 最终通过置信度与可训练门槛后只剩 19 个 usable direct 样本。

这说明问题不是 direct Actor 完全答不对，而是 direct Actor 的输出形式没有稳定落在分类器定义的 direct 风格范围内。

## 结论

Phase 2 的脚本执行成功，API 分类失败率也达标，但 Phase 2 的数据质量不符合后续 Actor SFT 训练预期。

主要失败原因是风格数据严重不平衡，尤其 direct 风格几乎没有足够可用样本。其根因是 Phase 1 生成提示和 Phase 2 分类规则之间存在契约不一致：生成阶段没有强制 direct/evidence/elimination 输出可被分类器稳定识别的结构或锚点。

因此，不应直接进入 Phase 3。应先修复 Phase 1 的风格生成提示，使三种 Actor 输出和 Phase 2 分类器规则一致，然后重新运行 Phase 1 和 Phase 2，并再次执行 Phase 2 质量确认。

## 建议修复方向

1. 调整 `src/prompts/actor_prompts.py` 中三种 Actor 风格提示：
   - direct 明确要求使用 `Direct reason:`，且只给 1 到 2 句简短理由；
   - evidence 明确要求使用 `Key evidence:` 或 `Application:`；
   - elimination 明确要求使用 `Option analysis:` 或 `Elimination:`。
2. 重新运行 Phase 1，生成新的 bootstrap 轨迹。
3. 重新运行 Phase 2，让分类器基于新的风格锚点重新分类。
4. 再次执行 `confirm/02_check_actor_sft_data.py`，确认：
   - 每个 style 的 usable 样本数 >= 256；
   - 每个 style 的 subject coverage >= 55；
   - style imbalance ratio <= 1.5；
   - API classification failure rate <= 0.011。

