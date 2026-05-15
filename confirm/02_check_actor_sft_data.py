"""Confirm Phase 2 classified data is suitable for Actor SFT."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import (
    gate,
    init_config,
    overall_status,
    print_summary,
    read_json,
    resolve_output_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 2 Actor SFT data")
    parser.add_argument("--config", default="configs/society/experiment_mmlu.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _subject_coverage_by_style(actor_report: dict[str, Any], styles: list[str]) -> dict[str, int]:
    coverage = {style: 0 for style in styles}
    by_subject = actor_report.get("by_subject_style", {}) or {}
    for counts in by_subject.values():
        for style in styles:
            if int((counts or {}).get(style, 0) or 0) > 0:
                coverage[style] += 1
    return coverage


def main() -> None:
    args = parse_args()
    cfg = init_config(args.config)
    step2 = cfg.step("step02_classify").to_dict()
    step3 = cfg.step("step03_train_actors_sft").to_dict()
    input_dir = Path(args.input_dir or step2.get("output_dir", "output/society/classified"))

    classified_path = input_dir / "classified_data.json"
    report_path = input_dir / "classification_report.json"
    actor_report_path = input_dir / "actor_sft_candidate_report.json"
    for path in (classified_path, report_path, actor_report_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 2 artifact not found: {path}")

    classified = read_json(classified_path)
    classification_report = read_json(report_path)
    actor_report = read_json(actor_report_path)

    styles = [str(s) for s in step3.get("reasoning_styles", [])]
    by_style = actor_report.get("by_style", {}) or {}
    coverage = _subject_coverage_by_style(actor_report, styles)
    metadata = classified.get("metadata", {}) or {}
    failure_rate = float(metadata.get("classification_failure_rate", 0.0) or 0.0)
    max_failure_rate = float(step2.get("max_classification_failure_rate", 0.0) or 0.0)
    min_examples = int(step3.get("min_examples_per_actor", 0) or 0)
    min_subject_coverage = int(step3.get("min_subject_coverage", 0) or 0)
    style_imbalance = actor_report.get("style_imbalance_ratio")
    max_style_imbalance = float(step3.get("max_style_imbalance_ratio", 0.0) or 0.0)

    gates = [
        gate("classified_samples", len(classified.get("results", [])) > 0, len(classified.get("results", [])), "> 0"),
        gate("classification_failure_rate", failure_rate <= max_failure_rate, failure_rate, f"<= {max_failure_rate}"),
    ]
    for style in styles:
        usable = int((by_style.get(style) or {}).get("usable", 0) or 0)
        gates.append(gate(
            f"usable_examples_{style}",
            usable >= min_examples,
            usable,
            f">= {min_examples}",
        ))
        gates.append(gate(
            f"subject_coverage_{style}",
            coverage.get(style, 0) >= min_subject_coverage,
            coverage.get(style, 0),
            f">= {min_subject_coverage}",
        ))
    if style_imbalance is not None and max_style_imbalance > 0:
        gates.append(gate(
            "style_imbalance_ratio",
            float(style_imbalance) <= max_style_imbalance,
            style_imbalance,
            f"<= {max_style_imbalance}",
        ))
    gates.append(gate(
        "classifier_warnings",
        None,
        classification_report.get("warnings", []),
        "review manually",
    ))

    report = {
        "phase": "phase02_actor_sft_data",
        "status": overall_status(gates),
        "input_dir": str(input_dir),
        "summary": {
            "num_results": len(classified.get("results", [])),
            "classification_attempts": metadata.get("classification_attempts", 0),
            "classification_failures": metadata.get("classification_failures", 0),
            "classification_failure_rate": failure_rate,
            "style_imbalance_ratio": style_imbalance,
            "usable_subject_coverage": actor_report.get("usable_subject_coverage"),
        },
        "by_style": by_style,
        "subject_coverage_by_style": coverage,
        "gate_retention_by_prompted_style": classification_report.get(
            "gate_retention_by_prompted_style",
            {},
        ),
        "prompted_vs_classified_confusion": classification_report.get(
            "prompted_vs_classified_confusion",
            {},
        ),
        "warnings": classification_report.get("warnings", []),
        "gates": gates,
    }
    output_path = resolve_output_path(args.output, cfg, "phase02_actor_sft_data_quality.json")
    write_json(output_path, report)
    print_summary(output_path, gates)


if __name__ == "__main__":
    main()
