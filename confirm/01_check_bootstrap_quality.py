"""Confirm Phase 1 bootstrap candidate quality."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from _common import (
    gate,
    init_config,
    overall_status,
    print_summary,
    rate,
    read_jsonl,
    resolve_output_path,
    write_json,
)
from src.evaluation.answer_resolution import answers_match


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 1 bootstrap quality")
    parser.add_argument("--config", default="configs/society/experiment_mmlu.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-parse-rate", type=float, default=0.90)
    parser.add_argument("--min-subject-coverage", type=int, default=None)
    return parser.parse_args()


def _response_records(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for traj in trajectories:
        sample = traj.get("sample", {}) or {}
        gold = sample.get("answer", "")
        task_type = sample.get("task_type", "multiple_choice")
        for resp in traj.get("initial_responses", []):
            row = dict(resp)
            row["sample"] = sample
            row["sample_id"] = traj.get("sample_id", "")
            row["subject"] = (
                row.get("subject")
                or traj.get("subject")
                or sample.get("subject")
                or sample.get("category")
                or "unknown"
            )
            row["is_parseable"] = bool(row.get("answer"))
            row["is_correct"] = answers_match(row.get("answer"), gold, task_type)
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    cfg = init_config(args.config)
    step = cfg.step("step01_bootstrap").to_dict()
    input_dir = Path(args.input_dir or step.get("output_dir", "output/society/bootstrap"))
    trajectory_file = input_dir / "trajectories.jsonl"
    if not trajectory_file.exists():
        raise FileNotFoundError(f"Phase 1 trajectories not found: {trajectory_file}")

    trajectories = read_jsonl(trajectory_file)
    rows = _response_records(trajectories)
    subjects = {
        str(t.get("subject") or t.get("sample", {}).get("subject") or "unknown")
        for t in trajectories
    }
    by_style: dict[str, Counter[str]] = defaultdict(Counter)
    by_temperature: dict[str, Counter[str]] = defaultdict(Counter)
    responses_per_sample = [len(t.get("initial_responses", [])) for t in trajectories]

    for row in rows:
        style = str(row.get("prompted_style") or "unknown")
        temp = str(row.get("temperature"))
        by_style[style]["generated"] += 1
        by_temperature[temp]["generated"] += 1
        if row["is_parseable"]:
            by_style[style]["parseable"] += 1
            by_temperature[temp]["parseable"] += 1
        if row["is_correct"]:
            by_style[style]["correct"] += 1
            by_temperature[temp]["correct"] += 1

    styles = [str(s) for s in step.get("reasoning_styles", [])]
    temperatures = [float(t) for t in step.get("temperatures", [])]
    expected_per_sample = (
        len(styles)
        * len(temperatures)
        * int(step.get("generations_per_temperature", 1))
    )
    min_subject_coverage = (
        args.min_subject_coverage
        if args.min_subject_coverage is not None
        else int(step.get("min_subjects", 0) or 0)
    )
    parse_rate = rate(sum(1 for row in rows if row["is_parseable"]), len(rows))
    accuracy = rate(sum(1 for row in rows if row["is_correct"]), len(rows))
    min_responses = min(responses_per_sample) if responses_per_sample else 0

    gates = [
        gate("trajectories_exist", len(trajectories) > 0, len(trajectories), "> 0"),
        gate(
            "subject_coverage",
            len(subjects) >= min_subject_coverage,
            len(subjects),
            f">= {min_subject_coverage}",
        ),
        gate(
            "responses_per_sample_min",
            min_responses >= expected_per_sample,
            min_responses,
            f">= {expected_per_sample}",
        ),
        gate("parse_rate", parse_rate >= args.min_parse_rate, parse_rate, f">= {args.min_parse_rate}"),
        gate("candidate_accuracy", None, accuracy, "informational"),
    ]
    for style in styles:
        gates.append(gate(
            f"style_generated_{style}",
            by_style[style]["generated"] > 0,
            by_style[style]["generated"],
            "> 0",
        ))

    report = {
        "phase": "phase01_bootstrap",
        "status": overall_status(gates),
        "input_file": str(trajectory_file),
        "summary": {
            "num_samples": len(trajectories),
            "num_responses": len(rows),
            "subject_coverage": len(subjects),
            "responses_per_sample": {
                "expected": expected_per_sample,
                "min": min_responses,
                "max": max(responses_per_sample) if responses_per_sample else 0,
                "mean": mean(responses_per_sample) if responses_per_sample else 0.0,
            },
            "parse_rate": parse_rate,
            "candidate_accuracy": accuracy,
        },
        "by_style": {
            style: {
                **dict(counts),
                "parse_rate": rate(counts["parseable"], counts["generated"]),
                "accuracy": rate(counts["correct"], counts["generated"]),
            }
            for style, counts in sorted(by_style.items())
        },
        "by_temperature": {
            temp: {
                **dict(counts),
                "parse_rate": rate(counts["parseable"], counts["generated"]),
                "accuracy": rate(counts["correct"], counts["generated"]),
            }
            for temp, counts in sorted(by_temperature.items())
        },
        "gates": gates,
    }
    output_path = resolve_output_path(args.output, cfg, "phase01_bootstrap_quality.json")
    write_json(output_path, report)
    print_summary(output_path, gates)


if __name__ == "__main__":
    main()

