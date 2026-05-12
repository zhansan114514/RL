# CLAUDE.md

Guidance for coding agents working in this repository.

## Project State

The repository is now the Diverse Actor-Critic Society experiment codebase.
The old standalone 1 Actor + 1 Critic pipeline and its tests were removed
during the destructive restructure. Keep new work inside the Society pipeline
and its current module boundaries.

## Supported Pipeline

The supported one-command entrypoint is:

```bash
python scripts/13_society_pipeline.py --config configs/society/experiment_mmlu.yaml
```

The supported phase scripts are:

```text
scripts/07_bootstrap_actors.py
scripts/08_classify_data.py
scripts/09_diversify_actors.py
scripts/10_diversify_critics.py
scripts/11_society_train.py
scripts/12_society_evaluate.py
scripts/13_society_pipeline.py
```

Valid tracked experiment configs are:

```text
configs/society/experiment_mmlu.yaml
configs/society/experiment_large.yaml
configs/society/trend/mmlu_smoke_n50.yaml
configs/society/trend/mmlu_trend_n300.yaml
```

Provider credentials and machine-local overrides belong in `configs/local.yaml`
or the path specified by `ACC_CONFIG_LOCAL`.

## Architecture

The main data flow is:

```text
dataset loading (src.data)
  -> style-prompted bootstrap data (scripts/07)
  -> API classification (src.society.data_classifier, scripts/08)
  -> actor diversification (scripts/09)
  -> critic diversification (scripts/10)
  -> society alternating training (src.society.society_trainer, scripts/11)
  -> society evaluation and ablations (scripts/12)
```

Module responsibilities:

| Module | Role |
| --- | --- |
| `src.society` | Agent registry, Critic router, classification, diversity splits, multi-agent deliberation, Society training and inference |
| `src.prompts` | Natural Actor/Critic prompt builders and guided prompt helpers |
| `src.parsing` | Actor answer extraction and Critic judgement parsing |
| `src.data` | Dataset loading, MMLU handling, preprocessing, split sampling |
| `src.algorithms` | Natural deliberation and guided trajectory utilities reused by Society training |
| `src.training` | `train_dpo`, DPO subprocess runner, LoRA target module config |
| `src.evaluation` | Answer resolution, mixed-task accuracy, Society metrics, style diagnostics |
| `src.inference` | vLLM wrapper |
| `src.utils` | Config manager, model type detection, seeding, runtime library setup |

`src.algorithms` is now limited to shared reward and answer-normalization
helpers. Society training data generation lives in
`src.society.pair_generation`.

## Configuration

Use `ConfigManager.initialize(config_path=...)` and
`cfg.step(step_key, defaults=...)`. Experiment values come from `common:` and
`stepNN_*:` sections. Do not add broad CLI flags for knobs already represented
in YAML.

Merge priority:

```text
configs/default.yaml
selected experiment YAML
configs/local.yaml or ACC_CONFIG_LOCAL
explicit overrides
```

## Testing

Use:

```bash
export PYTHONPATH=$(pwd)
pytest tests/ -v
ruff check src/ scripts/ tests/
```

Current tests cover configuration, data preprocessing/loading, answer parsing,
prompt construction, Society router/deliberation, Society phase checkpointing,
evaluation modes, Qwen LoRA settings, and Society actor data flow.
