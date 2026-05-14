# Critic SFT Rewrite Plan

## Goal

Rewrite Critic training from DPO preference learning to supervised fine-tuning
based on the Multiagent Finetuning paper's critic-data construction.

Compatibility with old Critic DPO data, pair caches, and config keys is not
required. The old Critic DPO interface can be deleted.

The new Critic training target is:

```text
task
actor_i current answer
summary of other actors' answers
critic_i role / critique angle
    -> critic_i feedback

metadata:
actor_i revised answer
whether the final answer is correct
case type: correction or keep
```

## Paper Basis

Use the Multiagent Finetuning paper's Critic construction:

- Generation agents produce first-round answers.
- Later debate rounds use summaries of other agents' responses.
- Critic agents evaluate other agents' outputs and produce improved responses.
- Critic finetuning data is built from two sets:
  - `DC-`: initial answer differs from the final majority answer, but final
    answer matches it. This teaches correction.
  - `DC+`: initial answer already matches the final majority answer and still
    matches it at the end. This teaches preservation.
- The paper uses standard finetuning on these trajectories, not DPO.

For this repository, because MMLU labels are available, replace the paper's
majority-voted `y_hat` filter with ground-truth correctness for stricter data:

- `DC-`: actor starts wrong, critic feedback helps the actor revise to correct.
- `DC+`: actor starts correct, at least one other actor is wrong, and critic
  feedback keeps the target actor correct despite conflicting peer context.

## New Critic SFT Dataset

Each full training case should be stored with rich metadata:

```json
{
  "sample_id": "mmlu_000001",
  "critic_skill": "reasoning",
  "case_type": "correction",
  "sample": {
    "question": "...",
    "choices": ["...", "...", "...", "..."],
    "answer": "C",
    "task_type": "multiple_choice",
    "subject": "..."
  },
  "actor_name": "actor_direct",
  "actor_initial_response": "...",
  "actor_initial_answer": "B",
  "other_actor_summary": "...",
  "critic_role": "reasoning",
  "critic_feedback_target": "...",
  "actor_revised_response": "...",
  "actor_revised_answer": "C",
  "final_answer_correct": true,
  "error_profile": {
    "primary": "reasoning",
    "scores": {
      "reasoning": 0.9,
      "knowledge": 0.2,
      "grounding": 0.3,
      "verification": 0.4
    }
  }
}
```

The actual HuggingFace SFT dataset should only contain:

```json
{
  "prompt": "...",
  "response": "critic_feedback_target"
}
```

Do not include `chosen` or `rejected` in the SFT rows.

## Prompt Format

Add a Critic SFT prompt builder, preferably in `src/prompts/critic_prompts.py`
or `src/prompts/prompt_builder.py`.

The prompt should include:

```text
You are Critic-{skill}.
{skill-specific critique instruction}

Your role is to help the actor improve or preserve its answer.
Use the other actors' summary as peer context, but judge the target actor's
answer independently.

Task:
...

Target actor:
actor_direct

Target actor current response:
...

Summary of other actors' responses:
...

Now provide feedback to the target actor.

At the end, write:
Judgement:
Answer correct: yes/no/uncertain
Suggested answer: A/B/C/D/Yes/No, a math result, or unknown
Confidence: 0.0-1.0
```

The response target should preserve the existing parseable judgement block via
`render_critic_judgement(...)`, because inference still parses Critic feedback.

## Case Type: DC- Correction

This corresponds to the paper's:

```text
y1,n != y_hat and yM,n == y_hat
```

Repository version:

1. `actor_i` first-round answer is wrong.
2. Build a summary of the other actors' first-round answers.
3. Route the wrong `actor_i` answer through the existing error-profile
   classifier.
4. Assign the case to one or more Critic skills:
   - `reasoning`
   - `knowledge`
   - `grounding`
   - `verification`
5. Build the target Critic feedback:
   - says the target actor answer is wrong;
   - identifies the error using the target Critic skill;
   - uses useful peer-summary evidence when available;
   - points toward the correct answer;
   - ends with `Judgement: Answer correct: no`.
6. Feed this Critic feedback into the actor revision prompt.
7. Keep the case only if the revised actor answer is correct.

This teaches the Critic to correct an initially wrong actor answer.

## Case Type: DC+ Keep

This corresponds to the paper's:

```text
y1,n == y_hat and yM,n == y_hat
```

Repository version:

1. `actor_i` first-round answer is correct.
2. Require at least one other actor to be wrong, so the summary contains
   conflicting or misleading peer context. If all other actors are also
   correct, do not use the sample for `DC+` training.
3. Build a summary of other actors' first-round answers.
4. Assign the case to Critic skills using one of:
   - the dominant error profile from wrong peer responses;
   - balanced round-robin fallback across Critic skills only when wrong peer
     responses exist but their profiles are ambiguous or unclassifiable.
5. Build the target Critic feedback:
   - says the target actor answer is correct;
   - explains why it should be kept;
   - warns against wrong peer claims in the summary;
   - ends with `Judgement: Answer correct: yes`.
6. Feed this feedback into the actor revision prompt.
7. Keep the case only if the revised actor answer remains correct.

This teaches the Critic not to over-correct correct actors.

## Other Actor Summary

Add a deterministic summary builder first. Do not require an additional LLM
summary call unless later experiments need it.

Recommended format:

```text
Other actors:
- actor_evidence: answer C. Key reason: ...
- actor_elimination: answer B. Key reason: ...

Agreement:
- correct-looking answer C appears in actor_evidence.
- conflicting answer B appears in actor_elimination.
```

Rules:

- Include each other actor's extracted answer.
- Include a short clipped response body.
- Include disagreement information.
- Do not reveal the gold answer directly.
- For `DC+`, include wrong peer answers. A keep case without a wrong peer is
  not trainable under this plan.

## Data Generation Flow

Replace `scripts/10_diversify_critics.py` with a Critic SFT training stage.
Renaming to `scripts/10_train_critics_sft.py` is recommended.

New flow:

1. Load classified data and source samples.
2. Load Actor SFT LoRA paths from `actor_registry.json`.
3. Run first-round generation for all actors on selected source samples.
4. Extract each actor answer and correctness.
5. For each `(sample, actor_i)`:
   - build summary of all other actors;
   - create a `DC-` correction candidate if `actor_i` is wrong;
   - create a `DC+` keep candidate only if `actor_i` is correct and at least
     one other actor is wrong.
6. For `DC-`, classify the target actor's wrong response into error-profile
   skills.
7. For `DC+`, route by wrong peer error profiles when possible. If wrong peer
   profiles are ambiguous, use round-robin skill balancing while preserving the
   wrong-peer conflict requirement.
8. Render Critic feedback targets.
9. Run actor revision with target feedback.
10. Keep only candidates where revised answer is correct.
11. Build per-skill SFT datasets.
12. Train one Critic LoRA per skill with `train_sft`.
13. Write datasets, metrics, and `critic_registry.json`.

## Training Mix

Use a two-bucket mix per Critic:

```yaml
correction_weight: 0.65
keep_weight: 0.35
```

Sampling rules:

- `correction` examples come from `DC-`.
- `keep` examples come from `DC+`.
- Every active Critic needs at least `min_examples_per_critic`.
- Cap each skill at `target_examples_per_critic`.
- Deduplicate by `(question, actor_name, actor_initial_response, case_type)`.
- Keep per-skill metrics for:
  - correction count;
  - keep count;
  - subject coverage;
  - actor distribution;
  - initial wrong -> revised correct rate;
  - initial correct -> revised correct rate;
  - assigned error-profile distribution.

## Delete Old Critic DPO Interfaces

Delete these old DPO-specific functions from `scripts/10_diversify_critics.py`
or the replacement script:

- `select_critic_preference_pairs`
- `build_structured_critic_pairs`
- `summarize_structured_critic_pairs`
- `_render_chosen_judgement`
- `_render_rejected_judgement`
- `_negative_kind`
- `_critic_pair_quotas`
- `train_critic_dpo`

Also delete or stop routing Critic training through legacy DPO paths in:

- `src/society/society_trainer.py`
  - `_generate_critic_pairs_pairwise`
  - Critic calls to `_run_dpo_training`
  - Critic `preference_pairs.json` cache handling
- `src/society/pair_generation.py` when functions are only used for Critic DPO
  pair construction.
- `src/society/critic_pair_quality.py` if no remaining non-Critic-SFT caller
  uses those helpers.
- `scripts/11_society_train.py` if its Critic phase still assumes DPO pairs.

Delete or stop using these DPO artifacts:

- `preference_pairs`
- `chosen`
- `rejected`
- `pairs_{critic_skill}_adaptive.json`
- `raw_critic_pool.json` if it only exists for pair generation
- `routed_critic_pool.json` if it only exists for pair generation

Delete Critic DPO config keys:

- `beta`
- `pair_mix`
- `reward_threshold`
- `target_pairs_per_critic`
- `max_pairs_per_critic`
- `min_unique_pairs_per_critic`
- `max_critic_pair_duplicate_rate`
- `allow_synthetic_critique`

Keep the error-profile classifier, but repurpose it for SFT routing instead of
DPO pair construction.

## New Config

Replace `step04_diversify_critics` with:

```yaml
step04_train_critics_sft:
  input_dir: "output/society_mmlu_style3_qwen3/classified"
  actor_base_dir: "output/society_mmlu_style3_qwen3/actors"
  output_dir: "output/society_mmlu_style3_qwen3/critics"
  api_base: "https://api.labforge.top"
  api_model: "gpt5.5"
  request_timeout: 60
  retry_delay: 5
  max_retries: 5
  max_classification_workers: 4
  strict_classification: true
  max_classification_failure_rate: 0.0

  critic_skills: ["reasoning", "knowledge", "grounding", "verification"]
  max_source_samples: 500
  target_examples_per_critic: 512
  min_examples_per_critic: 64
  correction_weight: 0.65
  keep_weight: 0.35
  min_keep_conflict_ratio: 0.5

  actor_max_tokens: 192
  critic_feedback_max_tokens: 192
  revision_max_tokens: 192
  summary_max_chars_per_actor: 500

  lora_r: 64
  lora_alpha: 128
  learning_rate: 5.0e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 1
  max_length: 2048
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0
```

## Output Files

Write:

```text
output/.../critics/sft_critic_reasoning.json
output/.../critics/sft_critic_knowledge.json
output/.../critics/sft_critic_grounding.json
output/.../critics/sft_critic_verification.json

output/.../critics/sft_critic_reasoning_metrics.json
output/.../critics/sft_critic_knowledge_metrics.json
output/.../critics/sft_critic_grounding_metrics.json
output/.../critics/sft_critic_verification_metrics.json

output/.../critics/critic_sft_case_report.json
output/.../critics/critic_registry.json
```

Registry format:

```json
{
  "schema_version": 5,
  "training_method": "sft",
  "critics": {
    "reasoning": {
      "critic_skill": "reasoning",
      "model_path": "output/.../critic_reasoning_adapter",
      "base_model": "Qwen/Qwen3-14B",
      "status": "trained_specialist",
      "participates": true,
      "base_model_only": false,
      "metrics": {}
    }
  },
  "metadata": {
    "training_method": "sft",
    "case_types": ["correction", "keep"],
    "correction_weight": 0.65,
    "keep_weight": 0.35
  }
}
```

## Suggested Code Structure

New helper functions:

- `load_actor_lora_paths(actor_dir)`
- `build_actor_first_round_pool(...)`
- `build_other_actor_summary(actor_name, actor_outputs, sample)`
- `build_correction_case(...)`
- `build_keep_case(...)`
- `render_critic_sft_feedback(...)`
- `render_critic_sft_prompt(...)`
- `verify_case_with_actor_revision(...)`
- `select_critic_sft_examples_for_skill(...)`
- `examples_to_critic_sft_dataset(...)`
- `train_critic_sft(...)`
- `write_critic_registry(...)`

Prefer keeping these helpers in `scripts/10_train_critics_sft.py` until they
stabilize. Move reusable prompt helpers into `src/prompts`.

## Tests

Replace old Critic DPO tests with SFT tests:

- `test_build_correction_case_requires_initial_wrong_final_correct`
- `test_build_keep_case_requires_initial_correct_final_correct`
- `test_keep_case_prefers_wrong_peer_conflict`
- `test_critic_sft_rows_have_prompt_response_no_chosen_rejected`
- `test_critic_sft_prompt_contains_task_actor_summary_and_role`
- `test_critic_sft_feedback_has_parseable_judgement_block`
- `test_critic_sft_mix_respects_correction_keep_ratio`
- `test_train_critic_sft_calls_train_sft`
- `test_critic_registry_records_training_method_sft`
- `test_config_removes_critic_dpo_keys`

Update pipeline tests so Phase 4 is named Critic SFT training rather than
Critic diversification with DPO.

## Acceptance Criteria

The rewrite is complete when:

1. No Critic training path builds `chosen/rejected` pairs.
2. Critic training calls `train_sft`, not `train_dpo`.
3. `scripts/10_train_critics_sft.py` writes per-skill SFT JSON files with only
   `prompt`, `response`, and optional `metadata`.
4. `critic_registry.json` has `training_method: "sft"`.
5. Every active Critic has both `correction` and `keep` examples unless a metric
   explicitly marks a low-data fallback.
6. `DC-` cases require initial wrong and revised correct.
7. `DC+` cases require initial correct, at least one wrong peer answer, and
   revised correct.
8. Critic feedback targets remain parseable by the existing Critic parser.
9. Configs no longer contain Critic DPO-only keys.
10. Tests pass for the new SFT data flow.

## Notes

- Do not preserve old DPO compatibility.
- Do not train Critic on synthetic negative pairs.
- Do not leak gold answers into the prompt. The gold answer can be used only
  for filtering and for rendering the supervised feedback target.
- Keep the Critic skill specialization. The SFT data should still train
  different Critic roles, not a single generic Critic.
