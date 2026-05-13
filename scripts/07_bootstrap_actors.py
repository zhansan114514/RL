"""
Generate style-prompted Actor bootstrap data.

For each training sample, this script generates responses for every configured
Actor reasoning style before classification.  The output shape is
trajectories.jsonl so scripts/08_classify_data.py can reuse the same
schema-v3 labeling path.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.prompts.control_tokens import ensure_no_think, strip_no_think
from src.society.agent_registry import ReasoningStyle
from src.utils.config import ConfigManager

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen3-14B",
    "dataset": "mmlu",
    "cache_dir": "output/society",
    "output_dir": "output/society/bootstrap",
    "reasoning_styles": ["direct", "evidence", "elimination"],
    "generations_per_style": 4,
    "temperature": 0.8,
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

def expected_bootstrap_metadata(args: Any, styles: list[ReasoningStyle]) -> dict[str, Any]:
    """Metadata that must match before resuming a style-prompted bootstrap file."""

    def arg(name: str) -> Any:
        return getattr(args, name, STEP_DEFAULTS.get(name))

    return {
        "schema_version": 3,
        "generation_mode": "style_prompted",
        "model_name": str(arg("model_name")),
        "dataset": arg("dataset"),
        "seed": int(arg("seed")),
        "reasoning_styles": [style.value for style in styles],
        "generations_per_style": int(arg("generations_per_style")),
        "temperature": float(arg("temperature")),
        "top_p": float(arg("top_p")),
        "max_tokens": int(arg("max_tokens")),
        "sampling": arg("sampling"),
        "mmlu_load_mode": arg("mmlu_load_mode"),
    }


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate style-prompted Actor bootstrap data",
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
        f"This is independent generation attempt {generation_index + 1}. "
        "Produce a complete response in the requested style.\n\n"
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


def load_samples(args: Any) -> list[dict[str, Any]]:
    from src.data.loader import load_dataset

    data = load_dataset(
        args.dataset,
        seed=args.seed,
        sampling=getattr(args, "sampling", None),
        mmlu_load_mode=getattr(args, "mmlu_load_mode", "by_subject"),
    )
    samples = data.get("train", []) or data.get("test", [])
    if not samples:
        raise ValueError(f"No data loaded for dataset={args.dataset}")
    return samples


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
                    "Existing bootstrap output does not match the current style-prompted "
                    f"phase-07 configuration at {output_file}:{line_no}: {mismatches}. "
                    "Remove the stale trajectories.jsonl before rerunning."
                )

            sample_ids.add(sample_id)
    return sample_ids


def generate_batch(
    model: Any,
    batch_entries: list[tuple[int, dict[str, Any]]],
    args: Any,
    styles: list[ReasoningStyle],
) -> list[dict[str, Any]]:
    from src.algorithms.reward import extract_answer

    prompts: list[str] = []
    meta: list[tuple[int, ReasoningStyle, int]] = []
    for local_idx, (_, sample) in enumerate(batch_entries):
        for style in styles:
            for generation_index in range(int(args.generations_per_style)):
                prompts.append(
                    build_style_prompt(
                        sample,
                        args.dataset,
                        style,
                        generation_index,
                    )
                )
                meta.append((local_idx, style, generation_index))

    seed = int(args.seed) + batch_entries[0][0] * 1000
    generated = model.generate(
        prompts,
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=float(getattr(args, "top_p", 0.9)),
        seed=seed,
    )
    generated = coerce_generation_results(generated, len(prompts))

    responses_by_sample: list[list[dict[str, Any]]] = [
        [] for _ in batch_entries
    ]
    for (local_idx, style, generation_index), response_text in zip(meta, generated):
        global_idx, sample = batch_entries[local_idx]
        sample_id = f"{args.dataset}_{global_idx}"
        agent_name = f"actor_{style.value}"
        response_id = f"{sample_id}_{style.value}_{generation_index}"
        answer = extract_answer(response_text, sample.get("task_type", "multiple_choice"))
        responses_by_sample[local_idx].append({
            "agent_id": styles.index(style),
            "round": 0,
            "response": response_text,
            "answer": answer,
            "sample_id": sample_id,
            "response_id": response_id,
            "prompted_style": style.value,
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
            "sample": sample,
            "initial_responses": responses,
            "debate_rounds": [],
            "consensus_answer": consensus,
            "confidence": confidence,
            "metadata": expected_bootstrap_metadata(args, styles),
        })
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "trajectories.jsonl"

    styles = [ReasoningStyle(style) for style in args.reasoning_styles]
    batch_size = max(1, int(args.batch_size))

    logger.info("=" * 60)
    logger.info("Generate Style-Prompted Actor Data")
    logger.info("  Model: %s", args.model_name)
    logger.info("  Dataset: %s", args.dataset)
    logger.info("  Styles: %s", [style.value for style in styles])
    logger.info("  Generations per style: %s", args.generations_per_style)
    logger.info("  Output dir: %s", output_dir)
    logger.info("=" * 60)

    logger.info("[Step 1] Loading dataset...")
    samples = load_samples(args)
    logger.info("  Loaded %s samples", len(samples))

    completed = existing_sample_ids(output_file, args, styles)
    pending = [
        (idx, sample)
        for idx, sample in enumerate(samples)
        if f"{args.dataset}_{idx}" not in completed
    ]
    logger.info("  Existing trajectories: %s", len(completed))
    logger.info("  Pending samples: %s", len(pending))

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
        logger.info("[Step 3] Generating style-prompted responses...")
        for batch_start in range(0, len(pending), batch_size):
            batch_entries = pending[batch_start:batch_start + batch_size]
            if not batch_entries:
                continue
            first = batch_start + 1
            last = batch_start + len(batch_entries)
            prompt_count = (
                len(batch_entries) * len(styles) * int(args.generations_per_style)
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

    logger.info("[Step 4] Saved trajectories to %s", output_file)
    logger.info("Style-prompted bootstrap complete")


if __name__ == "__main__":
    main()
