"""Confirm Phase 3 Actor SFT LoRAs with a small held-out evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from _common import (
    gate,
    get_api_key,
    init_config,
    overall_status,
    print_summary,
    rate,
    read_json,
    resolve_output_path,
    sample_items,
    write_json,
)
from src.evaluation.answer_resolution import answers_match
from src.parsing.answer_extractor import extract_answer
from src.prompts.prompt_builder import build_simple_actor_prompt
from src.society.agent_registry import ReasoningStyle, resolve_reasoning_style
from src.society.data_classifier import ClassificationError, classify_reasoning_style
from src.society.multi_deliberation import _load_lora_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 Actor SFT adapters")
    parser.add_argument("--config", default="configs/society/experiment_mmlu.yaml")
    parser.add_argument("--actor-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--style-check-samples", type=int, default=16)
    parser.add_argument("--no-style-classification", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--min-parse-rate", type=float, default=0.90)
    parser.add_argument("--min-style-match-rate", type=float, default=0.60)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    return parser.parse_args()


def _load_eval_samples(step_cfg: dict[str, Any], max_samples: int, seed: int) -> list[dict[str, Any]]:
    from src.data.loader import load_dataset

    data = load_dataset(
        step_cfg.get("dataset", "mmlu"),
        seed=seed,
        sampling=step_cfg.get("sampling"),
        mmlu_load_mode=step_cfg.get("mmlu_load_mode", "by_subject"),
    )
    samples = data.get("validation") or data.get("dev") or []
    if not samples:
        raise RuntimeError("No validation/dev samples available for Actor confirmation")
    return sample_items(list(samples), max_samples, seed)


def _classify_actor_styles(
    *,
    actor_style: str,
    rows: list[dict[str, Any]],
    max_items: int,
    api_key: str,
    api_base: str,
    api_model: str,
    request_timeout: int | float,
    max_retries: int,
    retry_delay: int | float,
    cache_dir: str,
) -> dict[str, Any]:
    candidates = [row for row in rows if row.get("response")]
    selected = candidates[: max(0, max_items)]
    if not selected:
        return {"attempted": 0, "matched": 0, "failures": 0, "match_rate": None}

    matched = 0
    failures = 0
    classified = []
    for row in selected:
        sample = row["sample"]
        try:
            result = classify_reasoning_style(
                response=row["response"],
                question=sample.get("question", ""),
                correct_answer=sample.get("answer", ""),
                use_api=True,
                cache_dir=cache_dir,
                api_key=api_key,
                api_base=api_base,
                api_model=api_model,
                request_timeout=request_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            style = result.primary_style.value
            matched += int(style == actor_style)
            classified.append({
                "sample_id": row.get("sample_id", ""),
                "classified_style": style,
                "confidence": result.confidence,
                "matched": style == actor_style,
            })
        except ClassificationError as exc:
            failures += 1
            classified.append({
                "sample_id": row.get("sample_id", ""),
                "error": str(exc),
                "matched": False,
            })

    attempted = len(selected)
    successful = attempted - failures
    return {
        "attempted": attempted,
        "successful": successful,
        "matched": matched,
        "failures": failures,
        "failure_rate": rate(failures, attempted),
        "match_rate": rate(matched, successful) if successful else None,
        "items": classified,
    }


def main() -> None:
    args = parse_args()
    cfg = init_config(args.config)
    step3 = cfg.step("step03_train_actors_sft").to_dict()
    step2 = cfg.step("step02_classify").to_dict()
    actor_dir = Path(args.actor_dir or step3.get("output_dir", "output/society/actors"))
    registry_path = actor_dir / "actor_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Actor registry not found: {registry_path}")

    registry = read_json(registry_path)
    actors = registry.get("actors", {}) or {}
    if not actors:
        raise RuntimeError(f"No actors found in {registry_path}")

    seed = int(step3.get("seed", 42))
    samples = _load_eval_samples(step3, args.max_samples, seed)

    from src.inference.vllm_server import VLLMInference

    device = args.device if args.device is not None else int(step3.get("device", 0))
    engine = VLLMInference(
        step3.get("model_name", "Qwen/Qwen3-14B"),
        cuda_device=device,
        dtype=step3.get("dtype", "bfloat16"),
        gpu_memory_utilization=float(step3.get("gpu_memory_utilization", 0.70)),
        max_model_len=int(step3.get("max_model_len", 4096)),
        enable_lora=True,
        max_loras=max(1, len(actors)),
        max_lora_rank=int(step3.get("max_lora_rank", step3.get("lora_r", 256))),
    )

    actor_reports: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    try:
        for raw_style, info in sorted(actors.items()):
            style: ReasoningStyle = resolve_reasoning_style(raw_style)
            lora_path = str(info.get("model_path") or "")
            lora_req = _load_lora_adapter(engine, lora_path)
            prompts = [
                build_simple_actor_prompt(
                    sample,
                    step3.get("dataset", "mmlu"),
                    style=style,
                    no_think=True,
                )
                for sample in samples
            ]
            responses: list[str] = []
            for start in range(0, len(prompts), max(1, args.batch_size)):
                batch = prompts[start:start + args.batch_size]
                responses.extend(engine.generate_with_lora(
                    batch,
                    lora_req,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                ))

            rows = []
            source_counts: Counter[str] = Counter()
            correct = 0
            for idx, (sample, response) in enumerate(zip(samples, responses)):
                task_type = sample.get("task_type", "multiple_choice")
                extracted = extract_answer(response, task_type)
                source_counts[extracted.source] += 1
                is_correct = answers_match(extracted.answer, sample.get("answer", ""), task_type)
                correct += int(is_correct)
                row = {
                    "actor": f"actor_{style.value}",
                    "style": style.value,
                    "sample_id": str(sample.get("sample_id") or idx),
                    "sample": sample,
                    "response": response,
                    "answer": extracted.answer,
                    "answer_source": extracted.source,
                    "parse_confidence": extracted.confidence,
                    "is_correct": is_correct,
                }
                rows.append(row)
                all_rows.append(row)

            style_check = {"skipped": True}
            if not args.no_style_classification:
                api_key = args.api_key or get_api_key(step2)
                if api_key:
                    style_check = _classify_actor_styles(
                        actor_style=style.value,
                        rows=rows,
                        max_items=args.style_check_samples,
                        api_key=api_key,
                        api_base=str(step2.get("api_base", "")),
                        api_model=str(step2.get("api_model", "")),
                        request_timeout=step2.get("request_timeout", 60),
                        max_retries=int(step2.get("max_retries", 5)),
                        retry_delay=step2.get("retry_delay", 5),
                        cache_dir=str(actor_dir / "confirm_style_cache"),
                    )
                else:
                    style_check = {"skipped": True, "reason": "missing_api_key"}

            total = len(rows)
            actor_reports[style.value] = {
                "num_samples": total,
                "accuracy": rate(correct, total),
                "parse_rate": rate(total - source_counts["none"], total),
                "answer_source_counts": dict(source_counts),
                "style_check": style_check,
                "examples": [
                    {
                        "sample_id": row["sample_id"],
                        "answer": row["answer"],
                        "gold": row["sample"].get("answer", ""),
                        "is_correct": row["is_correct"],
                        "answer_source": row["answer_source"],
                        "response": row["response"][:800],
                    }
                    for row in rows[:5]
                ],
            }
    finally:
        engine.cleanup()

    gates = []
    for style, report in actor_reports.items():
        gates.append(gate(
            f"actor_{style}_parse_rate",
            report["parse_rate"] >= args.min_parse_rate,
            report["parse_rate"],
            f">= {args.min_parse_rate}",
        ))
        gates.append(gate(
            f"actor_{style}_accuracy",
            report["accuracy"] >= args.min_accuracy,
            report["accuracy"],
            f">= {args.min_accuracy}",
        ))
        style_check = report.get("style_check", {})
        match_rate = style_check.get("match_rate")
        if match_rate is None:
            gates.append(gate(f"actor_{style}_style_match_rate", None, style_check, "skipped or no successful classifications"))
        else:
            gates.append(gate(
                f"actor_{style}_style_match_rate",
                float(match_rate) >= args.min_style_match_rate,
                match_rate,
                f">= {args.min_style_match_rate}",
            ))

    report = {
        "phase": "phase03_actor_sft_eval",
        "status": overall_status(gates),
        "actor_dir": str(actor_dir),
        "num_eval_samples": len(samples),
        "actor_reports": actor_reports,
        "gates": gates,
    }
    output_path = resolve_output_path(args.output, cfg, "phase03_actor_sft_eval.json")
    write_json(output_path, report)
    print_summary(output_path, gates)


if __name__ == "__main__":
    main()

