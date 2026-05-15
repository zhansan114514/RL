"""Confirm Phase 6 final evaluation and ablation results."""

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
    parser = argparse.ArgumentParser(description="Check Phase 6 final evaluation results")
    parser.add_argument("--config", default="configs/society/experiment_mmlu.yaml")
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-final-accuracy", type=float, default=0.0)
    parser.add_argument("--min-absolute-improvement", type=float, default=0.0)
    parser.add_argument("--min-a5-minus-a1", type=float, default=0.0)
    parser.add_argument("--min-a5-minus-a4", type=float, default=0.0)
    return parser.parse_args()


def _final_accuracy(ablation_results: dict[str, Any], key: str) -> float | None:
    item = ablation_results.get(key)
    if not isinstance(item, dict):
        return None
    value = item.get("final_consensus_accuracy")
    return None if value is None else float(value)


def main() -> None:
    args = parse_args()
    cfg = init_config(args.config)
    step6 = cfg.step("step06_evaluate").to_dict()
    eval_dir = Path(args.eval_dir or step6.get("output_dir", "output/society/eval"))
    results_path = eval_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Evaluation results not found: {results_path}")

    results = read_json(results_path)
    main_metrics = results.get("main_metrics", {}) or {}
    ablations = results.get("ablation_results", {}) or {}
    main_final = main_metrics.get("final_consensus_accuracy")
    main_gain = main_metrics.get("absolute_improvement")
    a5 = _final_accuracy(ablations, "A5_full_system")
    a4 = _final_accuracy(ablations, "A4_no_routing")
    a1 = _final_accuracy(ablations, "A1_acc_collab")

    gates = [
        gate("results_exist", True, str(results_path), "exists"),
        gate(
            "main_final_accuracy",
            main_final is not None and float(main_final) >= args.min_final_accuracy,
            main_final,
            f">= {args.min_final_accuracy}",
        ),
        gate(
            "main_absolute_improvement",
            main_gain is not None and float(main_gain) >= args.min_absolute_improvement,
            main_gain,
            f">= {args.min_absolute_improvement}",
        ),
    ]
    if bool(step6.get("run_ablations", True)):
        gates.extend([
            gate("A1_acc_collab_present", a1 is not None, a1, "present"),
            gate("A4_no_routing_present", a4 is not None, a4, "present"),
            gate("A5_full_system_present", a5 is not None, a5, "present"),
        ])
        if a5 is not None and a1 is not None:
            delta = a5 - a1
            gates.append(gate(
                "A5_minus_A1",
                delta >= args.min_a5_minus_a1,
                delta,
                f">= {args.min_a5_minus_a1}",
            ))
        if a5 is not None and a4 is not None:
            delta = a5 - a4
            gates.append(gate(
                "A5_minus_A4_router_gain",
                delta >= args.min_a5_minus_a4,
                delta,
                f">= {args.min_a5_minus_a4}",
            ))

    report = {
        "phase": "phase06_final_evaluation",
        "status": overall_status(gates),
        "eval_dir": str(eval_dir),
        "summary": {
            "main_metrics": main_metrics,
            "available_ablations": sorted(ablations),
            "a5_final_consensus_accuracy": a5,
            "a1_final_consensus_accuracy": a1,
            "a4_final_consensus_accuracy": a4,
            "a5_minus_a1": None if a5 is None or a1 is None else a5 - a1,
            "a5_minus_a4": None if a5 is None or a4 is None else a5 - a4,
            "deliberation_dynamics": results.get("deliberation_dynamics", {}),
            "critic_metrics": results.get("critic_metrics", {}),
        },
        "gates": gates,
    }
    output_path = resolve_output_path(args.output, cfg, "phase06_final_evaluation_quality.json")
    write_json(output_path, report)
    print_summary(output_path, gates)


if __name__ == "__main__":
    main()

