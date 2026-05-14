# Diverse Actor-Critic Society

This repository contains the current Society experiment pipeline: multiple
style-specialized Actors, multiple skill-specialized Critics, confidence-based
Critic routing, LoRA DPO training, and A1-A5 ablation evaluation.

The previous standalone 1 Actor + 1 Critic experiment scripts and tests have
been removed. The supported experiment entrypoints are the Society phase
scripts under `scripts/07_*.py` through `scripts/13_society_pipeline.py`.

## Current Structure

```text
src/
├── algorithms/          # Natural deliberation and guided trajectory utilities reused by Society
├── data/                # Dataset loading, MMLU support, preprocessing, sampling
├── evaluation/          # Answer resolution, metrics, and style diagnostics
├── inference/           # vLLM wrapper
├── parsing/             # Actor answer and Critic judgement parsers
├── prompts/             # Natural Actor/Critic prompt builders
├── society/             # Agent registry, router, classifier, split logic, training, inference
├── training/            # SFT/DPO runners, LoRA config, train_sft/train_dpo
└── utils/               # Config, seeding, runtime environment helpers
```

## Setup

```bash
conda create -n society-rl python=3.10 -y
conda activate society-rl
pip install -e ".[dev]"
```

For API-based classification, set provider credentials through
`configs/local.yaml`, `ACC_CONFIG_LOCAL`, or environment variables such as
`GLM_API_KEY`.

## Running Experiments

Use the checked-in MMLU configs:

```bash
python scripts/13_society_pipeline.py --config configs/society/experiment_mmlu.yaml
python scripts/13_society_pipeline.py --config configs/society/trend/mmlu_smoke_n50.yaml
python scripts/13_society_pipeline.py --config configs/society/trend/mmlu_trend_n300.yaml

# Run only selected phases
python scripts/13_society_pipeline.py --config configs/society/experiment_mmlu.yaml --only 5 6
```

Individual phases remain available for debugging:

```bash
python scripts/07_bootstrap_actors.py --config configs/society/experiment_mmlu.yaml
python scripts/08_classify_data.py --config configs/society/experiment_mmlu.yaml
python scripts/09_train_actors_sft.py --config configs/society/experiment_mmlu.yaml
python scripts/10_diversify_critics.py --config configs/society/experiment_mmlu.yaml
python scripts/11_society_train.py --config configs/society/experiment_mmlu.yaml
python scripts/12_society_evaluate.py --config configs/society/experiment_mmlu.yaml
```

## Configuration

`ConfigManager` loads and merges:

1. `configs/default.yaml`
2. the selected experiment YAML
3. `configs/local.yaml` or `ACC_CONFIG_LOCAL`
4. explicit OmegaConf overrides, where used

Runtime knobs such as sample counts, devices, model names, API provider, LoRA
rank, and output directories should live in YAML under `common:` and the
`stepNN_*:` sections.

## Tests

```bash
export PYTHONPATH=$(pwd)
pytest tests/ -v
ruff check src/ scripts/ tests/
```

The current tests cover configuration, data loading, parsing, prompt
construction, Society routing/deliberation, phase checkpointing, evaluation
modes, Qwen LoRA settings, and Society training data flow.
