"""Confirm Phase 5 Society alternating training artifacts and pair quality."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from _common import (
    gate,
    init_config,
    overall_status,
    print_summary,
    rate,
    read_json,
    resolve_output_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Phase 5 Society training artifacts")
    parser.add_argument("--config", default="configs/society/experiment_mmlu.yaml")
    parser.add_argument("--society-dir", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _load_preference_pair_files(society_dir: Path) -> dict[str, Any]:
    files = sorted(society_dir.glob("**/preference_pairs.json"))
    summaries = []
    total_pairs = 0
    for path in files:
        try:
            pairs = read_json(path)
        except Exception as exc:
            summaries.append({
                "path": str(path),
                "error": str(exc),
                "num_pairs": 0,
            })
            continue
        if not isinstance(pairs, list):
            summaries.append({
                "path": str(path),
                "error": "not_a_list",
                "num_pairs": 0,
            })
            continue
        total_pairs += len(pairs)
        unique_keys = {
            (
                (pair.get("sample") or {}).get("question", ""),
                pair.get("chosen", ""),
                pair.get("rejected", ""),
            )
            for pair in pairs
        }
        source_counts = Counter(
            (pair.get("metadata") or {}).get("pair_source")
            or (pair.get("metadata") or {}).get("source_bucket")
            or "unknown"
            for pair in pairs
        )
        summaries.append({
            "path": str(path),
            "num_pairs": len(pairs),
            "unique_pair_count": len(unique_keys),
            "duplicate_rate": 1.0 - rate(len(unique_keys), len(pairs)),
            "source_counts": dict(source_counts),
        })
    return {
        "num_files": len(files),
        "total_pairs": total_pairs,
        "files": summaries,
    }


def _metric_status_counts(metrics: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in metrics.values():
        if isinstance(value, dict):
            counts[str(value.get("status", "unknown"))] += 1
    return counts


def main() -> None:
    args = parse_args()
    cfg = init_config(args.config)
    step3 = cfg.step("step03_train_actors_sft").to_dict()
    step4 = cfg.step("step04_diversify_critics").to_dict()
    step5 = cfg.step("step05_train_society").to_dict()
    society_dir = Path(args.society_dir or step5.get("output_dir", "output/society/society"))
    registry_path = society_dir / "final_agent_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Final Society registry not found: {registry_path}")

    registry = read_json(registry_path)
    metrics = registry.get("metrics", {}) or {}
    actors = registry.get("actors", {}) or {}
    critics = registry.get("critics", {}) or {}
    pair_files = _load_preference_pair_files(society_dir)

    expected_actor_count = len(step3.get("reasoning_styles", []) or [])
    expected_critic_count = len(step4.get("critic_skills", []) or [])
    min_actor_pairs = int(step5.get("min_pairs_per_actor", 0) or 0)
    min_critic_pairs = int(step5.get("min_pairs_per_critic", 0) or 0)
    max_critic_duplicate = float(step5.get("max_critic_pair_duplicate_rate", 1.0) or 1.0)
    max_actor_fallback = float(step5.get("max_actor_fallback_ratio", 1.0) or 1.0)

    status_counts = _metric_status_counts(metrics)
    gates = [
        gate("final_registry_exists", True, str(registry_path), "exists"),
        gate("actor_count", len(actors) >= expected_actor_count, len(actors), f">= {expected_actor_count}"),
        gate("critic_count", len(critics) >= expected_critic_count, len(critics), f">= {expected_critic_count}"),
        gate("preference_pair_files", pair_files["num_files"] > 0, pair_files["num_files"], "> 0"),
        gate("total_preference_pairs", pair_files["total_pairs"] > 0, pair_files["total_pairs"], "> 0"),
        gate(
            "failed_or_skipped_agents",
            not any(status in status_counts for status in ("failed", "failed_low_data", "skipped_low_data")),
            dict(status_counts),
            "no failed/skipped_low_data",
        ),
    ]

    for metric_name, value in sorted(metrics.items()):
        if not isinstance(value, dict) or "pairs" not in value:
            continue
        pairs = int(value.get("pairs", 0) or 0)
        if metric_name.startswith("critic_"):
            gates.append(gate(
                f"{metric_name}_pairs",
                pairs >= min_critic_pairs,
                pairs,
                f">= {min_critic_pairs}",
            ))
            duplicate_rate = (
                (value.get("critic_training_metrics") or {}).get("duplicate_rate")
            )
            if duplicate_rate is not None:
                gates.append(gate(
                    f"{metric_name}_duplicate_rate",
                    float(duplicate_rate) <= max_critic_duplicate,
                    duplicate_rate,
                    f"<= {max_critic_duplicate}",
                ))
        if metric_name.startswith("actor_"):
            gates.append(gate(
                f"{metric_name}_pairs",
                pairs >= min_actor_pairs,
                pairs,
                f">= {min_actor_pairs}",
            ))
            source_ratios = (
                (value.get("actor_training_metrics") or {}).get("pair_source_ratios")
                or {}
            )
            fallback_ratio = source_ratios.get("prompted_fallback")
            if fallback_ratio is not None:
                gates.append(gate(
                    f"{metric_name}_fallback_ratio",
                    float(fallback_ratio) <= max_actor_fallback,
                    fallback_ratio,
                    f"<= {max_actor_fallback}",
                ))

    report = {
        "phase": "phase05_society_training",
        "status": overall_status(gates),
        "society_dir": str(society_dir),
        "summary": {
            "actor_count": len(actors),
            "critic_count": len(critics),
            "metric_status_counts": dict(status_counts),
            "preference_pair_file_count": pair_files["num_files"],
            "total_preference_pairs": pair_files["total_pairs"],
        },
        "metrics": metrics,
        "preference_pair_files": pair_files,
        "gates": gates,
    }
    output_path = resolve_output_path(args.output, cfg, "phase05_society_training_quality.json")
    write_json(output_path, report)
    print_summary(output_path, gates)


if __name__ == "__main__":
    main()

