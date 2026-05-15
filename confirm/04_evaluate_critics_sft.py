"""Confirm Phase 4 Critic SFT LoRAs with judgement and specialty probes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from src.parsing.critic_parser import parse_critic_response
from src.prompts.prompt_builder import build_simple_critic_prompt
from src.society.agent_registry import CriticSkill, resolve_critic_skill
from src.society.data_classifier import ClassificationError, classify_error_profile
from src.society.multi_deliberation import _load_lora_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 Critic SFT adapters")
    parser.add_argument("--config", default="configs/society/experiment_mmlu.yaml")
    parser.add_argument("--classified-dir", default=None)
    parser.add_argument("--critic-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-wrong-cases", type=int, default=32)
    parser.add_argument("--max-correct-cases", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--no-error-profile-classification", action="store_true")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--min-judgement-parse-rate", type=float, default=0.80)
    parser.add_argument("--min-judgement-accuracy", type=float, default=0.50)
    parser.add_argument("--min-suggested-answer-accuracy", type=float, default=0.20)
    parser.add_argument("--min-specialty-top1-rate", type=float, default=0.25)
    return parser.parse_args()


def _build_cases(
    classified: dict[str, Any],
    *,
    max_wrong: int,
    max_correct: int,
    seed: int,
) -> list[dict[str, Any]]:
    correct_cases: list[dict[str, Any]] = []
    wrong_cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in classified.get("results", []):
        sample = result.get("sample", {}) or {}
        sample_id = str(result.get("sample_id") or "")
        for label in result.get("per_response_labels", []):
            response = str(label.get("response") or "").strip()
            if not response:
                continue
            response_id = str(label.get("response_id") or f"{sample_id}:{len(seen)}")
            if response_id in seen:
                continue
            seen.add(response_id)
            case = {
                "case_id": response_id,
                "sample_id": sample_id,
                "sample": sample,
                "actor_response": response,
                "actor_answer": label.get("answer", ""),
                "expected_answer_correct": "yes" if label.get("is_correct") else "no",
                "is_actor_correct": bool(label.get("is_correct")),
            }
            if case["is_actor_correct"]:
                correct_cases.append(case)
            else:
                wrong_cases.append(case)
    return (
        sample_items(wrong_cases, max_wrong, seed)
        + sample_items(correct_cases, max_correct, seed + 1)
    )


def _profile_wrong_cases(
    cases: list[dict[str, Any]],
    *,
    dataset_name: str,
    api_key: str,
    api_base: str,
    api_model: str,
    request_timeout: int | float,
    max_retries: int,
    retry_delay: int | float,
    cache_dir: str,
) -> dict[str, Any]:
    profiled = 0
    failures = 0
    counts: Counter[str] = Counter()
    for case in cases:
        if case.get("is_actor_correct"):
            continue
        sample = case["sample"]
        try:
            result = classify_error_profile(
                response=case["actor_response"],
                question=sample.get("question", ""),
                extracted_answer=case.get("actor_answer", ""),
                correct_answer=sample.get("answer", ""),
                choices=sample.get("choices", ""),
                dataset_name=dataset_name,
                task_type=sample.get("task_type", ""),
                subject=sample.get("subject", sample.get("category", "")),
                use_api=True,
                cache_dir=cache_dir,
                api_key=api_key,
                api_base=api_base,
                api_model=api_model,
                request_timeout=request_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            case["error_profile"] = {
                "primary": result.primary,
                "secondary": result.secondary,
                "confidence": result.confidence,
                "scores": result.scores,
                "evidence": result.evidence,
            }
            if result.primary in {skill.value for skill in CriticSkill}:
                case["target_skill"] = result.primary
                counts[result.primary] += 1
            profiled += 1
        except ClassificationError as exc:
            case["error_profile_error"] = str(exc)
            failures += 1
    return {
        "profiled_wrong_cases": profiled,
        "profile_failures": failures,
        "target_skill_counts": dict(counts),
    }


def _route_score(parsed: Any) -> float:
    if parsed.confidence is not None:
        return float(parsed.confidence)
    return 0.35 if parsed.usable_for_feedback else 0.0


def main() -> None:
    args = parse_args()
    cfg = init_config(args.config)
    step4 = cfg.step("step04_diversify_critics").to_dict()
    step5 = cfg.step("step05_train_society").to_dict()
    classified_dir = Path(args.classified_dir or step4.get("input_dir", "output/society/classified"))
    critic_dir = Path(args.critic_dir or step4.get("output_dir", "output/society/critics"))
    classified_path = classified_dir / "classified_data.json"
    registry_path = critic_dir / "critic_registry.json"
    if not classified_path.exists():
        raise FileNotFoundError(f"Classified data not found: {classified_path}")
    if not registry_path.exists():
        raise FileNotFoundError(f"Critic registry not found: {registry_path}")

    classified = read_json(classified_path)
    registry = read_json(registry_path)
    critics = registry.get("critics", {}) or {}
    if not critics:
        raise RuntimeError(f"No critics found in {registry_path}")

    seed = int(step4.get("seed", 42))
    cases = _build_cases(
        classified,
        max_wrong=args.max_wrong_cases,
        max_correct=args.max_correct_cases,
        seed=seed,
    )
    if not cases:
        raise RuntimeError("No correct/wrong Actor response cases available for Critic confirmation")

    profile_metrics = {"skipped": True}
    if not args.no_error_profile_classification:
        api_key = args.api_key or get_api_key(step5)
        if api_key:
            profile_metrics = _profile_wrong_cases(
                cases,
                dataset_name=str(step4.get("dataset", "mmlu")),
                api_key=api_key,
                api_base=str(step5.get("api_base", "")),
                api_model=str(step5.get("api_model", "")),
                request_timeout=step5.get("request_timeout", 60),
                max_retries=int(step5.get("max_retries", 5)),
                retry_delay=step5.get("retry_delay", 5),
                cache_dir=str(critic_dir / "confirm_error_profile_cache"),
            )
        else:
            profile_metrics = {"skipped": True, "reason": "missing_api_key"}

    from src.inference.vllm_server import VLLMInference

    lora_count = sum(1 for info in critics.values() if info.get("model_path"))
    device = args.device if args.device is not None else int(step4.get("device", 0))
    engine = VLLMInference(
        step4.get("model_name", "Qwen/Qwen3-14B"),
        cuda_device=device,
        dtype=step4.get("dtype", "bfloat16"),
        gpu_memory_utilization=float(step4.get("gpu_memory_utilization", 0.70)),
        max_model_len=int(step4.get("max_model_len", 4096)),
        enable_lora=lora_count > 0,
        max_loras=max(1, lora_count),
        max_lora_rank=int(step4.get("max_lora_rank", step4.get("lora_r", 256))),
    )

    critic_reports: dict[str, Any] = {}
    case_scores: dict[str, dict[str, float]] = defaultdict(dict)
    try:
        for raw_skill, info in sorted(critics.items()):
            skill = resolve_critic_skill(raw_skill)
            lora_path = str(info.get("model_path") or "")
            lora_req = _load_lora_adapter(engine, lora_path) if lora_path else None
            prompts = [
                build_simple_critic_prompt(
                    case["sample"],
                    step4.get("dataset", "mmlu"),
                    actor_response=case["actor_response"],
                    skill=skill,
                    no_think=True,
                )
                for case in cases
            ]
            responses: list[str] = []
            for start in range(0, len(prompts), max(1, args.batch_size)):
                batch = prompts[start:start + args.batch_size]
                if lora_req is not None:
                    responses.extend(engine.generate_with_lora(
                        batch,
                        lora_req,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    ))
                else:
                    responses.extend(engine.generate(
                        batch,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    ))

            parsed_rows = []
            parse_answer_correct = 0
            parse_confidence = 0
            judgement_correct = 0
            suggested_total = 0
            suggested_correct = 0
            parse_errors: Counter[str] = Counter()
            for case, response in zip(cases, responses):
                task_type = case["sample"].get("task_type", "multiple_choice")
                parsed = parse_critic_response(response, task_type=task_type)
                parse_errors.update(parsed.parse_errors)
                parse_answer_correct += int(parsed.has_answer_correct)
                parse_confidence += int(parsed.has_confidence)
                expected = case["expected_answer_correct"]
                judgement_ok = parsed.answer_correct == expected
                judgement_correct += int(judgement_ok)
                if not case["is_actor_correct"]:
                    suggested_total += 1
                    suggested_correct += int(answers_match(
                        parsed.suggested_answer,
                        case["sample"].get("answer", ""),
                        task_type,
                    ))
                score = _route_score(parsed)
                case_scores[case["case_id"]][skill.value] = score
                parsed_rows.append({
                    "case_id": case["case_id"],
                    "expected_answer_correct": expected,
                    "answer_correct": parsed.answer_correct,
                    "judgement_correct": judgement_ok,
                    "suggested_answer": parsed.suggested_answer,
                    "gold": case["sample"].get("answer", ""),
                    "confidence": parsed.confidence,
                    "route_score": score,
                    "parse_errors": parsed.parse_errors,
                    "response": response[:800],
                })

            total = len(parsed_rows)
            critic_reports[skill.value] = {
                "num_cases": total,
                "judgement_parse_rate": rate(parse_answer_correct, total),
                "confidence_parse_rate": rate(parse_confidence, total),
                "judgement_accuracy": rate(judgement_correct, total),
                "suggested_answer_accuracy_on_wrong": rate(suggested_correct, suggested_total),
                "suggested_answer_wrong_case_count": suggested_total,
                "parse_error_counts": dict(parse_errors),
                "examples": parsed_rows[:5],
            }
    finally:
        engine.cleanup()

    profiled_cases = [
        case for case in cases
        if case.get("target_skill") in {str(skill) for skill in critic_reports}
    ]
    specialty_top1 = 0
    specialty_margin_sum = 0.0
    specialty_by_skill: dict[str, Counter[str]] = defaultdict(Counter)
    for case in profiled_cases:
        target = str(case.get("target_skill"))
        scores = case_scores.get(case["case_id"], {})
        if not scores:
            continue
        top_skill = max(scores.items(), key=lambda item: item[1])[0]
        target_score = scores.get(target, 0.0)
        other_scores = [score for skill, score in scores.items() if skill != target]
        other_mean = sum(other_scores) / len(other_scores) if other_scores else 0.0
        specialty_top1 += int(top_skill == target)
        specialty_margin_sum += target_score - other_mean
        specialty_by_skill[target]["cases"] += 1
        specialty_by_skill[target]["top1"] += int(top_skill == target)

    all_totals = [r["num_cases"] for r in critic_reports.values()]
    total_cases = sum(all_totals)
    overall_parse = rate(
        sum(r["judgement_parse_rate"] * r["num_cases"] for r in critic_reports.values()),
        total_cases,
    )
    overall_judgement = rate(
        sum(r["judgement_accuracy"] * r["num_cases"] for r in critic_reports.values()),
        total_cases,
    )
    wrong_total = sum(r["suggested_answer_wrong_case_count"] for r in critic_reports.values())
    overall_suggested = rate(
        sum(
            r["suggested_answer_accuracy_on_wrong"] * r["suggested_answer_wrong_case_count"]
            for r in critic_reports.values()
        ),
        wrong_total,
    )
    specialty_top1_rate = rate(specialty_top1, len(profiled_cases))
    specialty_margin_mean = rate(specialty_margin_sum, len(profiled_cases))

    gates = [
        gate("critic_count", len(critic_reports) > 0, len(critic_reports), "> 0"),
        gate(
            "overall_judgement_parse_rate",
            overall_parse >= args.min_judgement_parse_rate,
            overall_parse,
            f">= {args.min_judgement_parse_rate}",
        ),
        gate(
            "overall_judgement_accuracy",
            overall_judgement >= args.min_judgement_accuracy,
            overall_judgement,
            f">= {args.min_judgement_accuracy}",
        ),
        gate(
            "overall_suggested_answer_accuracy_on_wrong",
            overall_suggested >= args.min_suggested_answer_accuracy,
            overall_suggested,
            f">= {args.min_suggested_answer_accuracy}",
        ),
    ]
    if profiled_cases:
        gates.append(gate(
            "specialty_top1_rate",
            specialty_top1_rate >= args.min_specialty_top1_rate,
            specialty_top1_rate,
            f">= {args.min_specialty_top1_rate}",
            "target error-profile Critic should often have the highest route score",
        ))
    else:
        gates.append(gate("specialty_top1_rate", None, profile_metrics, "skipped or no profiled wrong cases"))

    report = {
        "phase": "phase04_critic_sft_eval",
        "status": overall_status(gates),
        "critic_dir": str(critic_dir),
        "classified_dir": str(classified_dir),
        "num_cases": len(cases),
        "num_wrong_cases": sum(1 for case in cases if not case["is_actor_correct"]),
        "num_correct_cases": sum(1 for case in cases if case["is_actor_correct"]),
        "profile_metrics": profile_metrics,
        "overall": {
            "judgement_parse_rate": overall_parse,
            "judgement_accuracy": overall_judgement,
            "suggested_answer_accuracy_on_wrong": overall_suggested,
            "specialty_top1_rate": specialty_top1_rate if profiled_cases else None,
            "specialty_margin_mean": specialty_margin_mean if profiled_cases else None,
            "specialty_by_skill": {
                skill: {
                    "cases": counts["cases"],
                    "top1_rate": rate(counts["top1"], counts["cases"]),
                }
                for skill, counts in sorted(specialty_by_skill.items())
            },
        },
        "critic_reports": critic_reports,
        "gates": gates,
    }
    output_path = resolve_output_path(args.output, cfg, "phase04_critic_sft_eval.json")
    write_json(output_path, report)
    print_summary(output_path, gates)


if __name__ == "__main__":
    main()

