# Society Phase Confirmation Scripts

These scripts are independent checks to run after each Society phase. They do
not advance the training pipeline; they read phase artifacts, run small
validation probes where needed, and write JSON reports under
`<common.cache_dir>/confirm/` by default.

Run from the repository root:

```bash
python confirm/01_check_bootstrap_quality.py --config configs/society/experiment_mmlu.yaml
python confirm/02_check_actor_sft_data.py --config configs/society/experiment_mmlu.yaml
python confirm/03_evaluate_actors_sft.py --config configs/society/experiment_mmlu.yaml --max-samples 32
python confirm/04_evaluate_critics_sft.py --config configs/society/experiment_mmlu.yaml --max-wrong-cases 32 --max-correct-cases 16
python confirm/05_check_society_training.py --config configs/society/experiment_mmlu.yaml
python confirm/06_check_final_evaluation.py --config configs/society/experiment_mmlu.yaml
```

Notes:

- `03_evaluate_actors_sft.py` loads Actor LoRAs and runs a small validation
  set. It checks answer parse rate, accuracy, and optional API-based style
  match. Use `--no-style-classification` to skip API style checks.
- `04_evaluate_critics_sft.py` loads Critic LoRAs and checks judgement parsing,
  answer-correct judgement accuracy, suggested-answer accuracy on wrong Actor
  responses, and optional API-based specialty routing. Use
  `--no-error-profile-classification` to skip API specialty checks.
- Thresholds are CLI flags, so smoke runs can be loose while main runs stay
  strict.
- `06_check_final_evaluation.py` checks final `results.json`, including A5
  presence and configurable A5-vs-A1/A4 improvement gates.
