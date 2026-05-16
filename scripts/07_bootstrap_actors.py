"""
Generate first-round Actor SFT candidate data.

For each selected MMLU sample, this script generates responses for every
configured Actor reasoning style at every configured temperature.  The output
remains trajectories.jsonl so the classification phase can preserve the
existing sample-centric shape, but metadata is schema v4 and describes SFT
candidate generation instead of Actor DPO bootstrap data.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.prompts.control_tokens import ensure_no_think, strip_no_think
from src.prompts.actor_prompts import ACTOR_PROMPT_VERSION
from src.society.agent_registry import ReasoningStyle
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen3-14B",
    "dataset": "mmlu",
    "cache_dir": "output/society",
    "output_dir": "output/society/bootstrap",
    "source_splits": ["dev", "validation"],
    "subject_balanced": True,
    "min_subjects": 57,
    "max_samples_per_subject": 20,
    "max_samples": None,
    "reasoning_styles": ["direct", "evidence", "elimination"],
    "temperatures": [0.4, 0.7, 1.0, 1.2],
    "generations_per_temperature": 1,
    "top_p": 0.9,
    "max_tokens": 256,
    "batch_size": 8,
    "seed": 42,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 4096,
    "device": 0,
    "sampling": None,
    "mmlu_load_mode": "by_subject",
}


def _as_list(value: Any, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _temperatures(args: Any) -> list[float]:
    values = getattr(args, "temperatures", None)
    if values is None:
        values = [getattr(args, "temperature", 0.8)]
    temps = [float(v) for v in _as_list(values, [0.8])]
    if not temps:
        raise ValueError("At least one generation temperature is required")
    return temps


def _source_splits(args: Any) -> list[str]:
    return [str(s) for s in _as_list(getattr(args, "source_splits", None), ["train"])]


def expected_bootstrap_metadata(args: Any, styles: list[ReasoningStyle]) -> dict[str, Any]:
    """Metadata that must match before resuming a candidate generation file."""

    def arg(name: str) -> Any:
        return getattr(args, name, STEP_DEFAULTS.get(name))

    return {
        "schema_version": 4,
        "generation_mode": "actor_sft_candidates",
        "actor_prompt_version": ACTOR_PROMPT_VERSION,
        "model_name": str(arg("model_name")),
        "dataset": arg("dataset"),
        "seed": int(arg("seed")),
        "source_splits": _source_splits(args),
        "subject_balanced": bool(arg("subject_balanced")),
        "min_subjects": int(arg("min_subjects")),
        "max_samples_per_subject": arg("max_samples_per_subject"),
        "max_samples": arg("max_samples"),
        "reasoning_styles": [style.value for style in styles],
        "temperatures": _temperatures(args),
        "generations_per_temperature": int(arg("generations_per_temperature")),
        "top_p": float(arg("top_p")),
        "max_tokens": int(arg("max_tokens")),
        "sampling": arg("sampling"),
        "mmlu_load_mode": arg("mmlu_load_mode"),
    }


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate first-round Actor SFT candidate data",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/society/experiment_mmlu.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    return cfg.step(
        "step01_bootstrap",
        defaults=STEP_DEFAULTS,
    ).to_namespace()


def build_style_prompt(
    sample: dict[str, Any],
    dataset_name: str,
    style: ReasoningStyle,
    temperature: float,
    generation_index: int,
) -> str:
    from src.prompts.prompt_builder import build_simple_actor_prompt

    base_prompt = strip_no_think(build_simple_actor_prompt(
        sample,
        dataset_name,
        style=style,
        no_think=True,
    ))
    prompt = (
        f"This is independent SFT candidate generation attempt {generation_index + 1} "
        f"at temperature {temperature:g}. Follow the exact visible output contract "
        "for the requested Actor style. Do not add extra paragraphs, headings, "
        "or alternative analyses.\n\n"
        f"{base_prompt}"
    )
    return ensure_no_think(prompt)


def coerce_generation_results(results: Any, expected: int) -> list[str]:
    if isinstance(results, str):
        results = [results]
    results = [r if isinstance(r, str) else str(r) for r in results]
    if len(results) != expected:
        raise ValueError(f"Generated {len(results)} responses for {expected} prompts")
    return results


def compute_consensus(responses: list[dict[str, Any]], task_type: str) -> tuple[str, float]:
    answers = [str(r.get("answer") or "") for r in responses if r.get("answer")]
    if not answers:
        return "", 0.0

    if task_type == "math":
        from src.algorithms.reward import math_answers_equal

        groups: dict[str, list[str]] = {}
        for answer in answers:
            matched = None
            for key in groups:
                if math_answers_equal(answer, key):
                    matched = key
                    break
            if matched:
                groups[matched].append(answer)
            else:
                groups[answer] = [answer]
        best = max(groups, key=lambda key: len(groups[key]))
        return best, len(groups[best]) / len(answers)

    common = Counter(answers).most_common(1)[0]
    return common[0], common[1] / len(answers)


def _sample_key(sample: dict[str, Any], split_name: str, split_idx: int) -> str:
    source_split = str(sample.get("source_split") or split_name)
    source_index = sample.get("source_index", split_idx)
    subject = str(sample.get("subject") or sample.get("category") or "unknown")
    return f"{source_split}:{subject}:{source_index}"


def _cap_total_samples(
    samples: list[tuple[int, dict[str, Any]]],
    max_samples: int | None,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng = random.Random(seed)
    selected = list(samples)
    rng.shuffle(selected)
    return sorted(selected[:max_samples], key=lambda item: item[0])


def _cap_subject_balanced_total(
    samples: list[tuple[int, dict[str, Any]]],
    max_samples: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    """Cap total examples by round-robin subject selection to preserve coverage."""
    if max_samples is None or max_samples <= 0 or len(samples) <= max_samples:
        return samples

    by_subject: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in samples:
        subject = str(item[1].get("subject") or item[1].get("category") or "unknown")
        by_subject[subject].append(item)

    selected: list[tuple[int, dict[str, Any]]] = []
    subjects = sorted(by_subject)
    cursor = 0
    while len(selected) < max_samples:
        progressed = False
        for subject in subjects:
            items = by_subject[subject]
            if cursor < len(items):
                selected.append(items[cursor])
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
        cursor += 1
    return sorted(selected, key=lambda item: item[0])


def select_subject_balanced_samples(
    data: dict[str, list[dict[str, Any]]],
    source_splits: list[str],
    *,
    subject_balanced: bool,
    min_subjects: int,
    max_samples_per_subject: int | None,
    max_samples: int | None,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Select samples from configured splits, enforcing subject coverage."""

    selected_pool: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    ordinal = 0
    for split_name in source_splits:
        split_samples = data.get(split_name)
        if split_samples is None:
            raise ValueError(
                f"Configured source split '{split_name}' is not available. "
                f"Available splits: {sorted(data)}"
            )
        for split_idx, sample in enumerate(split_samples):
            key = _sample_key(sample, split_name, split_idx)
            if key in seen:
                continue
            seen.add(key)
            sample = dict(sample)
            sample.setdefault("source_split", split_name)
            sample.setdefault("source_index", split_idx)
            selected_pool.append((ordinal, sample))
            ordinal += 1

    if not selected_pool:
        raise ValueError(f"No samples found for source_splits={source_splits}")

    if not subject_balanced:
        samples = _cap_total_samples(selected_pool, max_samples, seed)
        subject_count = len({
            s.get("subject") or s.get("category") or "unknown"
            for _, s in samples
        })
        if subject_count < min_subjects:
            raise ValueError(
                f"Subject coverage {subject_count} is below min_subjects={min_subjects}"
            )
        return samples

    by_subject: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for item in selected_pool:
        subject = str(item[1].get("subject") or item[1].get("category") or "unknown")
        by_subject[subject].append(item)

    available_subjects = sorted(subject for subject, items in by_subject.items() if items)
    if len(available_subjects) < min_subjects:
        raise ValueError(
            f"Subject coverage {len(available_subjects)} is below min_subjects={min_subjects}; "
            f"source_splits={source_splits}"
        )

    rng = random.Random(seed)
    capped: list[tuple[int, dict[str, Any]]] = []
    per_subject_cap = (
        int(max_samples_per_subject)
        if max_samples_per_subject is not None and int(max_samples_per_subject) > 0
        else None
    )
    for subject in available_subjects:
        items = list(by_subject[subject])
        rng.shuffle(items)
        if per_subject_cap is not None:
            items = items[:per_subject_cap]
        capped.extend(items)

    capped = _cap_subject_balanced_total(capped, max_samples)
    final_subjects = {
        str(sample.get("subject") or sample.get("category") or "unknown")
        for _, sample in capped
    }
    if len(final_subjects) < min_subjects:
        raise ValueError(
            f"Subject coverage after caps is {len(final_subjects)}, below "
            f"min_subjects={min_subjects}. Increase max_samples or per-subject caps."
        )

    return sorted(capped, key=lambda item: item[0])


def load_samples(args: Any) -> list[tuple[int, dict[str, Any]]]:
    from src.data.loader import load_dataset

    data = load_dataset(
        args.dataset,
        seed=args.seed,
        sampling=getattr(args, "sampling", None),
        mmlu_load_mode=getattr(args, "mmlu_load_mode", "by_subject"),
    )
    return select_subject_balanced_samples(
        data,
        _source_splits(args),
        subject_balanced=bool(getattr(args, "subject_balanced", True)),
        min_subjects=int(getattr(args, "min_subjects", 0) or 0),
        max_samples_per_subject=getattr(args, "max_samples_per_subject", None),
        max_samples=getattr(args, "max_samples", None),
        seed=int(getattr(args, "seed", 42)),
    )


def existing_sample_ids(
    output_file: Path,
    args: Any,
    styles: list[ReasoningStyle],
) -> set[str]:
    if not output_file.exists():
        return set()

    expected_metadata = expected_bootstrap_metadata(args, styles)
    sample_ids: set[str] = set()
    with open(output_file) as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Existing bootstrap file is not valid JSONL at {output_file}:{line_no}. "
                    "Remove it before rerunning phase 07."
                ) from e

            sample_id = record.get("sample_id")
            if not sample_id:
                raise RuntimeError(
                    f"Existing bootstrap record is missing sample_id at {output_file}:{line_no}. "
                    "Remove the file before rerunning phase 07."
                )

            metadata = record.get("metadata") or {}
            mismatches = {
                key: {"expected": value, "found": metadata.get(key)}
                for key, value in expected_metadata.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise RuntimeError(
                    "Existing bootstrap output does not match the current "
                    f"actor-sft candidate configuration at {output_file}:{line_no}: "
                    f"{mismatches}. Remove stale trajectories.jsonl before rerunning."
                )

            sample_ids.add(sample_id)
    return sample_ids


def _response_id(
    sample_id: str,
    style: ReasoningStyle,
    temperature: float,
    generation_index: int,
) -> str:
    temp_text = f"{temperature:g}".replace("-", "m").replace(".", "p")
    return f"{sample_id}_{style.value}_t{temp_text}_g{generation_index}"


def generate_batch(
    model: Any,
    batch_entries: list[tuple[int, dict[str, Any]]],
    args: Any,
    styles: list[ReasoningStyle],
) -> list[dict[str, Any]]:
    from src.algorithms.reward import extract_answer

    temperatures = _temperatures(args)
    generations_per_temperature = int(getattr(args, "generations_per_temperature", 1))
    if generations_per_temperature <= 0:
        raise ValueError("generations_per_temperature must be positive")

    responses_by_sample: list[list[dict[str, Any]]] = [[] for _ in batch_entries]
    prompt_items: list[tuple[str, int, ReasoningStyle, float, int]] = []
    for local_idx, (_, sample) in enumerate(batch_entries):
        for style in styles:
            for temperature in temperatures:
                for generation_index in range(generations_per_temperature):
                    prompt_items.append((
                        build_style_prompt(
                            sample,
                            args.dataset,
                            style,
                            temperature,
                            generation_index,
                        ),
                        local_idx,
                        style,
                        temperature,
                        generation_index,
                    ))

    from vllm import SamplingParams

    prompts = [item[0] for item in prompt_items]
    sampling_params = []
    for _, local_idx, style, temperature, generation_index in prompt_items:
        seed = (
            int(args.seed)
            + batch_entries[local_idx][0] * 1000
            + temperatures.index(temperature) * 100_000
            + styles.index(style) * 10_000
            + generation_index
        )
        sampling_params.append(
            SamplingParams(
                max_tokens=int(args.max_tokens),
                temperature=temperature,
                top_p=float(getattr(args, "top_p", 0.9)),
                seed=seed,
            )
        )

    generated = model.generate_with_sampling_params(prompts, sampling_params)
    generated = coerce_generation_results(generated, len(prompts))
    generated_items = [
        (local_idx, style, temperature, generation_index, response_text)
        for (_, local_idx, style, temperature, generation_index), response_text
        in zip(prompt_items, generated)
    ]

    for local_idx, style, temperature, generation_index, response_text in generated_items:
        global_idx, sample = batch_entries[local_idx]
        sample_id = f"{args.dataset}_{global_idx}"
        agent_name = f"actor_{style.value}"
        answer = extract_answer(
            response_text,
            sample.get("task_type", "multiple_choice"),
        )
        responses_by_sample[local_idx].append({
            "agent_id": styles.index(style),
            "round": 0,
            "response": response_text,
            "answer": answer,
            "sample_id": sample_id,
            "source_split": sample.get("source_split", ""),
            "subject": sample.get("subject", sample.get("category", "")),
            "response_id": _response_id(
                sample_id,
                style,
                temperature,
                generation_index,
            ),
            "prompted_style": style.value,
            "temperature": temperature,
            "generation_index": generation_index,
            "agent_name": agent_name,
        })

    records: list[dict[str, Any]] = []
    for local_idx, (global_idx, sample) in enumerate(batch_entries):
        sample_id = f"{args.dataset}_{global_idx}"
        responses = responses_by_sample[local_idx]
        consensus, confidence = compute_consensus(
            responses,
            sample.get("task_type", "multiple_choice"),
        )
        records.append({
            "sample_id": sample_id,
            "source_split": sample.get("source_split", ""),
            "subject": sample.get("subject", sample.get("category", "")),
            "sample": sample,
            "initial_responses": responses,
            "debate_rounds": [],
            "consensus_answer": consensus,
            "confidence": confidence,
            "metadata": expected_bootstrap_metadata(args, styles),
        })
    return records


def build_generation_report(
    records: list[dict[str, Any]],
    args: Any,
    styles: list[ReasoningStyle],
) -> dict[str, Any]:
    by_subject = Counter(
        str(record.get("subject") or record.get("sample", {}).get("subject") or "unknown")
        for record in records
    )
    by_style: Counter[str] = Counter()
    for record in records:
        for resp in record.get("initial_responses", []):
            by_style[str(resp.get("prompted_style") or "unknown")] += 1

    temperatures = _temperatures(args)
    responses_per_sample = (
        len(styles)
        * len(temperatures)
        * int(getattr(args, "generations_per_temperature", 1))
    )
    return {
        "schema_version": 4,
        "generation_mode": "actor_sft_candidates",
        "num_samples": len(records),
        "subject_coverage": len(by_subject),
        "source_splits": _source_splits(args),
        "styles": [style.value for style in styles],
        "temperatures": temperatures,
        "generations_per_temperature": int(
            getattr(args, "generations_per_temperature", 1)
        ),
        "responses_per_sample": responses_per_sample,
        "by_subject": dict(sorted(by_subject.items())),
        "by_style": dict(sorted(by_style.items())),
    }


def load_existing_records(output_file: Path) -> list[dict[str, Any]]:
    if not output_file.exists():
        return []
    records = []
    with open(output_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "trajectories.jsonl"

    styles = [ReasoningStyle(style) for style in args.reasoning_styles]
    temperatures = _temperatures(args)
    batch_size = max(1, int(args.batch_size))

    logger.info("=" * 60)
    logger.info("Generate Actor SFT Candidate Data")
    logger.info("  Model: %s", args.model_name)
    logger.info("  Dataset: %s", args.dataset)
    logger.info("  Source splits: %s", _source_splits(args))
    logger.info("  Styles: %s", [style.value for style in styles])
    logger.info("  Temperatures: %s", temperatures)
    logger.info("  Generations per temperature: %s", args.generations_per_temperature)
    logger.info("  Output dir: %s", output_dir)
    logger.info("=" * 60)

    logger.info("[Step 1] Loading dataset...")
    samples = load_samples(args)
    logger.info("  Selected %s samples", len(samples))
    logger.info(
        "  Subject coverage: %s",
        len({
            str(sample.get("subject") or sample.get("category") or "unknown")
            for _, sample in samples
        }),
    )

    completed = existing_sample_ids(output_file, args, styles)
    pending = [
        (idx, sample)
        for idx, sample in samples
        if f"{args.dataset}_{idx}" not in completed
    ]
    logger.info("  Existing trajectories: %s", len(completed))
    logger.info("  Pending samples: %s", len(pending))

    if pending:
        logger.info("[Step 2] Loading model...")
        from src.inference.vllm_server import VLLMInference

        model = VLLMInference(
            args.model_name,
            cuda_device=args.device,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

        try:
            logger.info("[Step 3] Generating style/temperature candidates...")
            for batch_start in range(0, len(pending), batch_size):
                batch_entries = pending[batch_start:batch_start + batch_size]
                if not batch_entries:
                    continue
                first = batch_start + 1
                last = batch_start + len(batch_entries)
                prompt_count = (
                    len(batch_entries)
                    * len(styles)
                    * len(temperatures)
                    * int(args.generations_per_temperature)
                )
                logger.info(
                    "  Progress: pending %s-%s/%s (%s prompts)",
                    first,
                    last,
                    len(pending),
                    prompt_count,
                )
                records = generate_batch(model, batch_entries, args, styles)
                with open(output_file, "a") as f:
                    for record in records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            model.cleanup()
    else:
        logger.info("[Step 2] No pending samples; reusing existing trajectories")

    logger.info("[Step 4] Writing generation report...")
    records = load_existing_records(output_file)
    report = build_generation_report(records, args, styles)
    report_file = output_dir / "generation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Actor SFT candidate generation complete")
    logger.info("  Trajectories: %s", output_file)
    logger.info("  Report: %s", report_file)


if __name__ == "__main__":
    main()
