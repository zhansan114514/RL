"""
Diversify Critics by training specialized LoRA adapters for each critic skill.

For each Critic:
1. Load trained Actor LoRA adapters (from Step 09)
2. Generate Society-native pairwise guided rollout data with Actor + base Critic
3. Extract Critic preference pairs from structured Critic candidates
4. Route by error profile and build a skill-specific training mixture
5. Train with DPO
6. Save to output/society/critics/{agent_id}/

Usage:
    python scripts/10_diversify_critics.py \
        --config configs/society/experiment_mmlu.yaml
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _utils import setup_logging
from src.utils.config import ConfigManager
from src.society.multi_deliberation import LoRAError
from src.society.society_trainer import LoRAModelAdapter
from src.parsing.critic_parser import parse_critic_response
from src.prompts.critic_prompts import render_critic_judgement

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen3-14B",
    "dataset": "math",
    "cache_dir": "output/society",
    "input_dir": "output/society/classified",
    "actor_base_dir": "output/society/actors",
    "output_dir": "output/society/critics",
    "critic_skills": ["reasoning", "knowledge", "grounding", "verification"],
    "max_delib_samples": 300,
    "num_rounds": 5,
    "num_simulations": 5,
    "max_tokens": 1024,
    "reward_threshold": 0.0,
    "lora_r": 256,
    "lora_alpha": 512,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 1,
    "max_length": 2048,
    "beta": 0.1,
    "min_real_specialty_items": 16,
    "target_pairs_per_critic": 512,
    "max_pairs_per_critic": 1024,
    "min_pairs_per_critic": 64,
    "allow_synthetic_critique": True,
    "pair_mix": {"specialty": 0.75, "general": 0.25},
    "seed": 42,
    "device": 0,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.65,
    "max_model_len": 4096,
    "max_lora_rank": 256,
    "api_key": "",
    "api_base": "https://api.labforge.top",
    "api_model": "gpt5.5",
    "strict_classification": True,
    "max_classification_failure_rate": 0.0,
    "max_classification_workers": 4,
    "request_timeout": 60,
    "retry_delay": 5,
    "max_retries": 5,
    "sampling": None,
    "mmlu_load_mode": "by_subject",
}


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Diversify Critics",
    )
    parser.add_argument(
        "--config", type=str, default="configs/society/experiment_mmlu.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    return cfg.step("step04_diversify_critics", defaults=STEP_DEFAULTS).to_namespace()


def _build_adapters(
    engine: Any,
    lora_paths: Dict[str, str],
) -> Dict[str, LoRAModelAdapter]:
    """Create LoRAModelAdapter for each named agent path.

    Every provided path is required.  Continuing with a base actor after a
    failed LoRA load would corrupt the critic-diversification experiment.
    """
    from src.society.multi_deliberation import _load_lora_adapter

    adapters: Dict[str, LoRAModelAdapter] = {}
    for name, path in lora_paths.items():
        lora_req = None
        if path:
            try:
                lora_req = _load_lora_adapter(engine, path)
                logger.info(f"    Loaded LoRA for {name}: {path}")
            except LoRAError as e:
                raise LoRAError(
                    f"Required LoRA adapter for actor '{name}' failed "
                    f"to load from '{path}': {e}"
                ) from e
            if lora_req is None:
                raise LoRAError(
                    f"Required LoRA adapter for actor '{name}' at "
                    f"'{path}' produced no LoRARequest."
                )
        adapters[name] = LoRAModelAdapter(engine, lora_req)
    return adapters


def load_actor_lora_paths(actor_dir: str) -> Dict[str, str]:
    """Load actor LoRA paths from the registry saved by script 09."""
    registry_file = os.path.join(actor_dir, "actor_registry.json")
    if not os.path.exists(registry_file):
        logger.warning(f"Actor registry not found: {registry_file}")
        return {}

    with open(registry_file) as f:
        registry = json.load(f)

    paths = {}
    for style, info in registry.get("actors", {}).items():
        model_path = info.get("model_path", "")
        if model_path:
            paths[style] = model_path

    logger.info(f"  Loaded {len(paths)} actor LoRA paths from registry")
    return paths


def load_classified_data(input_dir: str) -> Dict[str, Any]:
    """Load classified data."""
    classified_file = os.path.join(input_dir, "classified_data.json")

    if not os.path.exists(classified_file):
        raise FileNotFoundError(f"Classified data not found: {classified_file}")

    with open(classified_file) as f:
        data = json.load(f)

    return data


def fingerprint_lora_paths(lora_paths: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Fingerprint adapter artifacts enough to invalidate stale pool caches."""
    fingerprints: Dict[str, Dict[str, Any]] = {}
    for name, path in sorted(lora_paths.items()):
        model_file = os.path.join(path, "adapter_model.safetensors")
        config_file = os.path.join(path, "adapter_config.json")
        entries = []
        for file_path in (model_file, config_file):
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                entries.append({
                    "path": file_path,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                })
        fingerprints[name] = {
            "path": path,
            "files": entries,
        }
    return fingerprints


def build_critic_raw_pairs(
    samples: List[Dict],
    actor_lora_paths: Dict[str, str],
    model_name: str = "Qwen/Qwen3-14B",
    dataset_name: str = "math",
    num_rounds: int = 5,
    num_simulations: int = 5,
    max_tokens: int = 256,
    reward_threshold: float = 0.0,
    max_samples: int = 50,
    seed: int = 42,
    device: int = 0,
    dtype: str = "bfloat16",
    gpu_memory_utilization: float = 0.65,
    max_model_len: int = 4096,
    max_lora_rank: int = 256,
    engine=None,
) -> List[Dict[str, Any]]:
    """
    Build the shared raw Critic-pair pool with Society pairwise rollouts.

    This work is independent of the target critic skill.  Generate it once,
    route it once, then sample skill-specific mixtures from the routed pool.
    """
    from src.inference.vllm_server import VLLMInference
    from src.algorithms.reward import extract_answer, math_answers_equal, normalize_answer
    from src.society.pair_generation import build_pairwise_training_pairs_batch
    from src.society.agent_registry import ReasoningStyle

    logger.info("  Generating shared Critic raw-pair pool via Society pairwise rollouts")

    all_lora_paths = {k: v for k, v in actor_lora_paths.items() if v}
    max_loras = len(all_lora_paths) if all_lora_paths else 0
    n_samples = min(len(samples), max_samples)
    raw_pairs: List[Dict[str, Any]] = []
    _owns_engine = engine is None

    try:
        if _owns_engine:
            engine = VLLMInference(
                model_name,
                cuda_device=device,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enable_lora=bool(all_lora_paths) or None,
                max_loras=max(max_loras, 1),
                max_lora_rank=max_lora_rank,
            )

        adapters = _build_adapters(engine, all_lora_paths)
        if not adapters:
            adapters = {"base_actor": LoRAModelAdapter(engine, None)}
        actor_names = list(adapters.keys())

        # Base Critic adapter (no LoRA)
        critic_adapter = LoRAModelAdapter(engine, None)

        actor_groups: Dict[str, List[Dict]] = {}
        for i, sample in enumerate(samples[:n_samples]):
            actor_groups.setdefault(actor_names[i % len(actor_names)], []).append(sample)

        for actor_name, group_samples in actor_groups.items():
            actor_adapter = adapters[actor_name]
            actor_style = None
            try:
                actor_style = ReasoningStyle(actor_name)
            except ValueError:
                if actor_name.startswith("actor_"):
                    try:
                        actor_style = ReasoningStyle(actor_name.removeprefix("actor_"))
                    except ValueError:
                        actor_style = None
            logger.info(
                "    Pairwise rollout shared batch with actor "
                f"{actor_name}: {len(group_samples)} samples"
            )

            rollout_pairs = build_pairwise_training_pairs_batch(
                actor_model=actor_adapter,
                critic_model=critic_adapter,
                samples=group_samples,
                dataset_name=dataset_name,
                num_rounds=num_rounds,
                num_simulations=num_simulations,
                reward_threshold=reward_threshold,
                max_tokens=max_tokens,
                seed=seed,
                batch_size=len(group_samples),
                actor_style=actor_style,
            )
            logger.info(
                f"    Pairwise rollout batch for actor {actor_name} produced "
                f"{len(rollout_pairs)} pairs"
            )

            for pair in rollout_pairs:
                sample = pair.get("sample", {})
                task_type = sample.get("task_type", "math")
                correct_answer = sample.get("answer", "")
                actor_candidate = pair["actor_candidate"]
                critic_candidate = pair["critic_candidate"]
                rejected_actor_response = actor_candidate["rejected"]
                rejected_answer = extract_answer(rejected_actor_response, task_type)

                # Only keep pairs where the actor's rejected response was wrong
                # (error scenario — this is where Critic feedback matters)
                if task_type == "math":
                    is_wrong = not math_answers_equal(
                        rejected_answer or "", correct_answer,
                    )
                else:
                    is_wrong = normalize_answer(
                        rejected_answer or "", task_type,
                    ) != normalize_answer(correct_answer, task_type)

                if is_wrong:
                    raw_pair_id = f"raw_{len(raw_pairs):06d}"
                    raw_pairs.append({
                        "raw_pair_id": raw_pair_id,
                        "actor_name": actor_name,
                        "sample": sample,
                        "critic_pair": critic_candidate,
                        "actor_pair": actor_candidate,
                        "actor_response": rejected_actor_response,
                        "actor_answer": rejected_answer,
                        "correct_answer": correct_answer,
                        "task_type": task_type,
                        "delta": pair["comparison"]["delta"],
                        "comparison_mode": pair["comparison"]["mode"],
                        "rollout_scores": pair["rollout_scores"],
                    })

        if _owns_engine:
            del engine
            _cleanup_gpu()

    except Exception as e:
        logger.error(f"  Failed to generate shared Critic raw-pair pool: {e}")
        if _owns_engine:
            _cleanup_gpu()
        raise

    logger.info(f"  Pairwise rollouts produced {len(raw_pairs)} shared raw critic pairs")
    return raw_pairs


def route_critic_raw_pairs(
    raw_pairs: List[Dict[str, Any]],
    dataset_name: str,
    input_dir: str,
    api_key: str,
    api_base: str,
    api_model: str,
    seed: int,
    strict_classification: bool,
    max_classification_failure_rate: float,
    max_classification_workers: int,
    request_timeout: int | float,
    max_retries: int,
    retry_delay: int | float,
):
    """Route the shared raw-pair pool by error profile once."""
    from src.society.diversity_split import DiversitySplit

    if not raw_pairs:
        logger.warning("  No shared raw pairs to route")
        return []

    splitter = DiversitySplit(
        balance=False, seed=seed, use_api=True,
        cache_dir=input_dir,
        pre_classified_file=os.path.join(input_dir, "classified_data.json"),
        api_key=api_key,
        api_base=api_base,
        api_model=api_model,
        strict_classification=strict_classification,
        max_classification_failure_rate=max_classification_failure_rate,
        max_classification_workers=max_classification_workers,
        request_timeout=request_timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    routed_items = splitter.split_by_error_profile(
        samples=[p["sample"] for p in raw_pairs],
        responses=[p["actor_response"] for p in raw_pairs],
        correct_answers=[p["correct_answer"] for p in raw_pairs],
        extracted_answers=[p["actor_answer"] or "" for p in raw_pairs],
        dataset_name=dataset_name,
        response_ids=[p["raw_pair_id"] for p in raw_pairs],
    )

    raw_skill_dist = Counter(
        item.skill.value if item.skill else "general"
        for item in routed_items
    )
    unique_raw_pairs = {
        (item.sample.get("question", ""), item.response)
        for item in routed_items
    }
    logger.info(f"  Shared routed profile distribution: {dict(raw_skill_dist)}")
    logger.info(
        f"  Shared routed unique_pairs: {len(unique_raw_pairs)} / "
        f"{len(routed_items)}"
    )
    return routed_items


def select_critic_preference_pairs(
    raw_pairs: List[Dict[str, Any]],
    routed_items: List[Any],
    critic_skill: str,
    max_samples: int,
    seed: int,
    min_specialty_items: int,
    min_specialty_ratio: float,
    specialty_ratio: float,
    general_ratio: float,
    calibration_ratio: float,
) -> List[Dict[str, Any]]:
    """Sample a skill-specific DPO mix from the shared routed pool."""
    from src.society.diversity_split import DiversitySplit
    from src.society.agent_registry import resolve_critic_skill

    skill = resolve_critic_skill(critic_skill)
    splitter = DiversitySplit(balance=False, seed=seed, use_api=False)
    critic_items = splitter.build_critic_training_mix(
        all_items=routed_items,
        target_skill=skill,
        max_items=max_samples,
        min_specialty_items=min_specialty_items,
        min_specialty_ratio=min_specialty_ratio,
        specialty_ratio=specialty_ratio,
        general_ratio=general_ratio,
        calibration_ratio=calibration_ratio,
    )

    if not critic_items:
        logger.info(
            f"  [{critic_skill}] inactive: specialty pool below threshold; "
            "no specialist DPO pairs selected"
        )
        return []

    pair_by_id = {
        p.get("raw_pair_id"): p
        for p in raw_pairs
        if p.get("raw_pair_id")
    }
    pair_by_content = {
        (p.get("sample", {}).get("question", ""), p.get("actor_response", "")): p
        for p in raw_pairs
    }

    preference_pairs: List[Dict[str, Any]] = []
    for item in critic_items:
        p = pair_by_id.get(getattr(item, "response_id", ""))
        if p is None:
            p = pair_by_content.get((item.sample.get("question", ""), item.response))
        if p is None:
            continue
        preference_pairs.append({
            "sample": p["sample"],
            "chosen": p["critic_pair"]["chosen"],
            "rejected": p["critic_pair"]["rejected"],
            "actor_response": p.get("actor_response", ""),
            "metadata": {
                "target_skill": skill.value,
                "assigned_skill": item.skill.value if item.skill else "general",
                "source_bucket": item.source_bucket,
                "routing_weight": item.weight,
                "error_profile": item.profile,
                "raw_pair_id": p.get("raw_pair_id", ""),
                "actor_name": p.get("actor_name", ""),
                "delta": p["delta"],
                "comparison_mode": p["comparison_mode"],
                "rollout_scores": p.get("rollout_scores", {}),
            },
        })

    logger.info(
        f"  {len(preference_pairs)}/{len(raw_pairs)} pairs selected "
        f"for skill '{skill.value}'"
    )

    if preference_pairs:
        skill_dist = Counter(
            p["metadata"]["assigned_skill"]
            for p in preference_pairs
        )
        bucket_dist = Counter(
            p["metadata"]["source_bucket"]
            for p in preference_pairs
        )
        selected_unique_pairs = {
            (p["sample"].get("question", ""), p.get("actor_response", ""))
            for p in preference_pairs
        }
        logger.info(
            f"  [{critic_skill}] source_bucket distribution: "
            f"{dict(bucket_dist)}"
        )
        logger.info(
            f"  [{critic_skill}] assigned_skill distribution: "
            f"{dict(skill_dist)}"
        )
        logger.info(
            f"  [{critic_skill}] selected unique_pairs: "
            f"{len(selected_unique_pairs)} / {len(preference_pairs)}"
        )

    return preference_pairs


def build_structured_critic_pairs(
    raw_pairs: List[Dict[str, Any]],
    routed_items: List[Any],
    critic_skill: str,
    max_pairs: int,
    seed: int,
    min_real_specialty_items: int,
    pair_mix: Dict[str, float] | None = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build DPO pairs where chosen/rejected are structured Critic judgements."""
    import numpy as np
    from src.society.agent_registry import resolve_critic_skill

    rng = np.random.default_rng(seed)
    skill = resolve_critic_skill(critic_skill)
    pair_mix = pair_mix or {"specialty": 0.75, "general": 0.25}

    pair_by_id = {
        p.get("raw_pair_id"): p
        for p in raw_pairs
        if p.get("raw_pair_id")
    }
    pair_by_content = {
        (p.get("sample", {}).get("question", ""), p.get("actor_response", "")): p
        for p in raw_pairs
    }

    specialty_items = [item for item in routed_items if item.skill == skill]
    if len(specialty_items) < min_real_specialty_items:
        return [], {
            "status": "frozen_base",
            "reason": "real_specialty_items_below_threshold",
            "real_specialty_items": len(specialty_items),
            "min_real_specialty_items": min_real_specialty_items,
        }

    general_items = [item for item in routed_items if item.skill is None]
    other_items = [item for item in routed_items if item.skill is not None and item.skill != skill]
    specialty_quota, general_quota = _critic_pair_quotas(max_pairs, pair_mix)

    selected: list[tuple[Any, str]] = []
    selected.extend((item, "specialty") for item in _sample_items(rng, specialty_items, specialty_quota))
    selected.extend((item, "general") for item in _sample_items(rng, general_items or routed_items, general_quota))

    structured_pairs: List[Dict[str, Any]] = []
    for idx, (item, source_bucket) in enumerate(selected):
        p = pair_by_id.get(getattr(item, "response_id", ""))
        if p is None:
            p = pair_by_content.get((item.sample.get("question", ""), item.response))
        if p is None:
            continue

        profile = item.profile if isinstance(item.profile, dict) else {}
        chosen = _render_chosen_judgement(p, profile)
        rejected = _render_rejected_judgement(
            p,
            profile,
            target_skill=skill.value,
            negative_kind=_negative_kind(idx, source_bucket),
        )
        parsed = parse_critic_response(chosen, p.get("sample", {}).get("task_type", "multiple_choice"))
        if not parsed.usable_for_feedback:
            raise RuntimeError(
                f"Generated unusable critic chosen for {p.get('raw_pair_id')}: "
                f"{parsed.parse_errors}"
            )

        structured_pairs.append({
            "sample": p["sample"],
            "chosen": chosen,
            "rejected": rejected,
            "actor_response": p.get("actor_response", ""),
            "metadata": {
                "target_skill": skill.value,
                "assigned_skill": item.skill.value if item.skill else "general",
                "source_bucket": source_bucket,
                "routing_weight": item.weight,
                "error_profile": profile,
                "raw_pair_id": p.get("raw_pair_id", ""),
                "actor_name": p.get("actor_name", ""),
                "actor_answer": p.get("actor_answer", ""),
                "correct_answer": p.get("correct_answer", ""),
                "structured_judgement": True,
            },
        })

    metrics = summarize_structured_critic_pairs(structured_pairs, len(specialty_items))
    metrics["status"] = (
        "trained_specialist"
        if len(specialty_items) >= min_real_specialty_items
        else "trained_low_data"
    )
    return structured_pairs[:max_pairs], metrics


def summarize_structured_critic_pairs(
    preference_pairs: List[Dict[str, Any]],
    real_specialty_items: int,
) -> Dict[str, Any]:
    total = len(preference_pairs)
    parse_usable = 0
    for pair in preference_pairs:
        parsed = parse_critic_response(
            pair.get("chosen", ""),
            pair.get("sample", {}).get("task_type", "multiple_choice"),
        )
        if parsed.usable_for_feedback:
            parse_usable += 1
    bucket_counts = Counter(
        p.get("metadata", {}).get("source_bucket", "unknown")
        for p in preference_pairs
    )
    assigned_counts = Counter(
        p.get("metadata", {}).get("assigned_skill", "unknown")
        for p in preference_pairs
    )
    return {
        "sample_count": total,
        "real_specialty_items": real_specialty_items,
        "chosen_parse_usable_rate": parse_usable / total if total else 0.0,
        "source_bucket_counts": dict(bucket_counts),
        "assigned_skill_counts": dict(assigned_counts),
        "synthetic_pair_count": 0,
        "synthetic_ratio": 0.0,
    }


def _render_chosen_judgement(pair: Dict[str, Any], profile: Dict[str, Any]) -> str:
    confidence = float(profile.get("confidence", 0.8) or 0.8)
    primary = str(profile.get("primary", "")).strip().lower()
    if primary not in {"reasoning", "knowledge", "grounding", "verification"}:
        primary = "verification"
    evidence = str(profile.get("evidence", "")).strip()
    if not evidence:
        evidence = "The actor response is inconsistent with the verified answer and needs correction."
    return render_critic_judgement(
        answer_correct="no",
        suggested_answer=str(pair.get("correct_answer") or "unknown"),
        confidence=max(0.1, min(1.0, confidence)),
        critique=evidence,
    )


def _render_rejected_judgement(
    pair: Dict[str, Any],
    profile: Dict[str, Any],
    target_skill: str,
    negative_kind: str,
) -> str:
    actor_answer = str(pair.get("actor_answer") or "unknown")
    correct_answer = str(pair.get("correct_answer") or "unknown")

    if negative_kind == "answer_correct_wrong":
        return render_critic_judgement(
            answer_correct="yes",
            suggested_answer=actor_answer,
            confidence=0.80,
            critique="The actor answer is acceptable.",
        )
    if negative_kind == "suggested_answer_wrong":
        return render_critic_judgement(
            answer_correct="no",
            suggested_answer=actor_answer,
            confidence=0.75,
            critique="The critique identifies a problem but repeats the actor's wrong answer.",
        )
    if negative_kind == "confidence_badly_calibrated":
        return render_critic_judgement(
            answer_correct="no",
            suggested_answer=correct_answer,
            confidence=0.05,
            critique="The actor response has a clear issue, but this judgement is under-confident.",
        )
    if negative_kind == "critique_misses_core_error":
        return render_critic_judgement(
            answer_correct="no",
            suggested_answer=correct_answer,
            confidence=0.70,
            critique="The response could be improved, but this critique does not identify the key reasoning problem.",
        )
    return render_critic_judgement(
        answer_correct="no",
        suggested_answer=correct_answer,
        confidence=0.35,
        critique="Weak critique.",
    )


def _negative_kind(idx: int, source_bucket: str) -> str:
    kinds = (
        ["answer_correct_wrong"] * 3
        + ["suggested_answer_wrong"] * 3
        + ["confidence_badly_calibrated"] * 2
        + ["critique_misses_core_error"] * 2
        + ["critique_weak"]
    )
    return kinds[idx % len(kinds)]


def _critic_pair_quotas(max_pairs: int, pair_mix: Dict[str, float]) -> tuple[int, int]:
    specialty = float(pair_mix.get("specialty", 0.75))
    general = float(pair_mix.get("general", 0.25))
    total = specialty + general or 1.0
    specialty_q = int(round(max_pairs * specialty / total))
    general_q = max_pairs - specialty_q
    return specialty_q, general_q


def _sample_items(rng, items: list[Any], n: int) -> list[Any]:
    if n <= 0 or not items:
        return []
    replace = len(items) < n
    indices = rng.choice(len(items), size=n if replace else min(n, len(items)), replace=replace)
    return [items[int(i)] for i in indices]


def _routed_item_to_json(item: Any) -> Dict[str, Any]:
    return {
        "sample": item.sample,
        "response": item.response,
        "skill": item.skill.value if item.skill else None,
        "weight": item.weight,
        "profile": item.profile,
        "source_bucket": item.source_bucket,
        "response_id": item.response_id,
    }


def _routed_item_from_json(data: Dict[str, Any]):
    from src.society.agent_registry import CriticSkill
    from src.society.diversity_split import RoutedTrainingItem

    skill_value = data.get("skill")
    return RoutedTrainingItem(
        sample=data.get("sample", {}),
        response=data.get("response", ""),
        skill=CriticSkill(skill_value) if skill_value else None,
        weight=float(data.get("weight", 1.0)),
        profile=data.get("profile", {}),
        source_bucket=data.get("source_bucket", "unknown"),
        response_id=data.get("response_id", ""),
    )


def _load_pool_cache(path: str, metadata: Dict[str, Any]) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    if payload.get("metadata") != metadata:
        logger.info(f"  Ignoring stale pool cache: {path}")
        return None
    logger.info(f"  Loaded pool cache: {path}")
    return payload


def _write_pool_cache(path: str, metadata: Dict[str, Any], key: str, value: Any) -> None:
    with open(path, "w") as f:
        json.dump({"metadata": metadata, key: value}, f)
    logger.info(f"  Cached {key} to {path}")


def _cleanup_gpu():
    """Release GPU memory."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def train_critic_dpo(
    model_name: str,
    preference_pairs: List[Dict],
    critic_skill: str,
    output_dir: str,
    dataset_name: str,
    lora_r: int,
    lora_alpha: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    max_length: int,
    beta: float,
    seed: int,
    device: int,
) -> str:
    """Train Critic with DPO."""
    from datasets import Dataset
    from src.training.dpo_trainer import train_dpo
    from src.utils.model_utils import detect_model_type
    from src.prompts.prompt_builder import build_simple_actor_prompt, build_simple_critic_prompt
    from src.society.agent_registry import resolve_critic_skill

    model_type = detect_model_type(model_name)

    # Create output directory
    critic_output_dir = os.path.join(output_dir, f"critic_{critic_skill}")
    os.makedirs(critic_output_dir, exist_ok=True)

    logger.info(f"  Training DPO for '{critic_skill}'...")
    logger.info(f"    Pairs: {len(preference_pairs)}")
    logger.info(f"    Output: {critic_output_dir}")

    prompts = []
    skill = resolve_critic_skill(critic_skill)
    for p in preference_pairs:
        sample = p.get("sample", {})
        actor_resp = p.get("actor_response", "")
        stored_prompt = str(p.get("prompt") or "").strip()
        if stored_prompt:
            prompt = stored_prompt
        elif actor_resp:
            prompt = build_simple_critic_prompt(
                sample,
                dataset_name,
                actor_resp,
                skill=skill,
            )
        else:
            prompt = build_simple_actor_prompt(sample, dataset_name)
        prompts.append(prompt)

    # Convert preference_pairs to HuggingFace Dataset
    hf_data = {
        "prompt": prompts,
        "chosen": [p["chosen"] for p in preference_pairs],
        "rejected": [p["rejected"] for p in preference_pairs],
    }
    preference_dataset = Dataset.from_dict(hf_data)

    # Train DPO
    checkpoint_path = train_dpo(
        model_name_or_path=model_name,
        preference_dataset=preference_dataset,
        output_dir=critic_output_dir,
        model_type=model_type,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        max_length=max_length,
        beta=beta,
        seed=seed,
        device=device,
    )

    logger.info(f"  Checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def main():
    args = parse_args()

    # Handle API key for live error-profile classification
    api_key = args.api_key
    if not api_key:
        api_key = os.environ.get("GLM_API_KEY", "")
    if api_key:
        os.environ["GLM_API_KEY"] = api_key
    elif getattr(args, "strict_classification", True):
        raise RuntimeError(
            "strict_classification=True requires GLM_API_KEY or "
            "step04_diversify_critics.api_key"
        )
    else:
        logger.warning(
            "GLM_API_KEY not set (neither config nor env var). "
            "Unseen raw pairs from pairwise rollouts will be routed to general pool."
        )
    api_base = getattr(args, "api_base", "https://api.labforge.top")
    api_model = getattr(args, "api_model", "gpt5.5")

    # Setup directories
    input_dir = args.input_dir
    output_dir = args.output_dir
    actor_dir = args.actor_base_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Diversify Critics (Society pairwise rollout-based)")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Input dir: {input_dir}")
    logger.info(f"  Actor dir: {actor_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Critic skills: {args.critic_skills}")
    logger.info(
        "  Active critic thresholds: "
        f"min_pairs={args.min_pairs_per_critic}, "
        f"min_real_specialty_items={getattr(args, 'min_real_specialty_items', 16)}"
    )
    if args.min_pairs_per_critic > args.max_delib_samples:
        logger.warning(
            "min_pairs_per_critic > max_delib_samples; no critic can become "
            "trained unless the selected training mix can exceed the sample cap. "
            "Current mix uses max_items=max_delib_samples, so this configuration "
            "will freeze every critic that relies on generated pairs."
        )
    logger.info(
        "  Training pair mix: "
        f"{getattr(args, 'pair_mix', {'specialty': 0.75, 'general': 0.25})}"
    )
    logger.info("=" * 60)

    # Load classified data to get sample list
    logger.info("[Step 1] Loading classified data...")
    classified_data = load_classified_data(input_dir)
    classified_results = classified_data["results"]

    # Use samples that have at least one incorrect response; skill routing is
    # handled later by DiversitySplit, not by hard pre-filtering labels here.
    incorrect_sample_ids: List[str] = []
    for r in classified_results:
        has_incorrect = r.get("metadata", {}).get("num_incorrect", 0) > 0
        if not has_incorrect:
            has_incorrect = any(
                not label.get("is_correct", False)
                for label in r.get("per_response_labels", [])
            )
        if has_incorrect:
            incorrect_sample_ids.append(r["sample_id"])

    # Load original dataset for pairwise rollout input
    logger.info("[Step 2] Loading dataset for pairwise rollouts...")
    from src.data.loader import load_dataset
    dataset = load_dataset(
        args.dataset,
        seed=args.seed,
        sampling=getattr(args, "sampling", None),
        mmlu_load_mode=getattr(args, "mmlu_load_mode", "by_subject"),
    )

    # Flatten all splits into a single list
    all_samples = []
    for split_data in dataset.values():
        all_samples.extend(split_data)
    logger.info(f"  Total samples across all splits: {len(all_samples)}")

    # Build sample lookup and filter to classified error samples
    sample_lookup = {}
    for sample in all_samples:
        q = sample.get("question", "")
        if q:
            sample_lookup[q] = sample

    # Load trajectories to build sample_id -> sample mapping
    bootstrap_dir = os.path.join(args.cache_dir, "bootstrap")
    trajectory_file = os.path.join(bootstrap_dir, "trajectories.jsonl")

    # Build a mapping from sample_id to the standardized sample
    id_to_sample: Dict[str, Dict] = {}
    if os.path.exists(trajectory_file):
        with open(trajectory_file) as f:
            for line in f:
                if line.strip():
                    traj = json.loads(line)
                    sample = traj.get("sample", {})
                    sid = traj.get("sample_id", "")
                    if sid and sample:
                        id_to_sample[sid] = sample
        logger.info(f"  Loaded {len(id_to_sample)} trajectory samples")
    else:
        # Fallback: use dataset samples directly
        for sample in all_samples:
            q = sample.get("question", "")
            id_to_sample[q] = sample
        logger.info(f"  Using {len(id_to_sample)} dataset samples directly")

    # Load Actor LoRA paths from script 09
    logger.info("[Step 3] Loading Actor LoRA paths...")
    actor_lora_paths = load_actor_lora_paths(actor_dir)

    if not actor_lora_paths:
        raise LoRAError(
            f"No Actor LoRA paths found in '{actor_dir}'. Run actor "
            f"diversification first and provide a valid actor_registry.json."
        )

    # Train each critic
    logger.info("[Step 4] Generating preference pairs & training Critics...")

    critic_paths = {}
    inactive_critics = {}
    critic_metrics = {}

    # Create the vLLM engine lazily only when the shared raw pool is missing.
    shared_engine = None

    def _pairs_cache_path(critic_skill):
        return os.path.join(output_dir, f"pairs_{critic_skill}_adaptive.json")

    try:
        all_pairs = {}
        sample_ids = incorrect_sample_ids
        source_samples = [
            id_to_sample[sid]
            for sid in sample_ids
            if sid in id_to_sample
        ]
        if not source_samples:
            raise RuntimeError("No incorrect samples found for Critic diversification")

        logger.info(f"  Shared source samples: {len(source_samples)}")

        pool_metadata = {
            "version": 3,
            "model_name": args.model_name,
            "dataset": args.dataset,
            "critic_skills": list(args.critic_skills),
            "actor_lora_fingerprints": fingerprint_lora_paths(actor_lora_paths),
            "source_sample_ids": sample_ids,
            "max_delib_samples": args.max_delib_samples,
            "num_rounds": args.num_rounds,
            "num_simulations": args.num_simulations,
            "max_tokens": args.max_tokens,
            "reward_threshold": args.reward_threshold,
            "seed": args.seed,
            "strict_classification": getattr(args, "strict_classification", True),
            "api_base": api_base,
            "api_model": api_model,
        }
        raw_pool_cache = os.path.join(output_dir, "raw_critic_pool.json")
        routed_pool_cache = os.path.join(output_dir, "routed_critic_pool.json")

        raw_cache = _load_pool_cache(raw_pool_cache, pool_metadata)
        if raw_cache is not None:
            raw_pairs = raw_cache.get("raw_pairs", [])
            logger.info(f"  Loaded {len(raw_pairs)} shared raw critic pairs")
        else:
            from src.inference.vllm_server import VLLMInference

            all_lora_paths = {k: v for k, v in actor_lora_paths.items() if v}
            shared_engine = VLLMInference(
                args.model_name,
                cuda_device=args.device,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                enable_lora=bool(all_lora_paths) or None,
                max_loras=max(1, len(all_lora_paths)),
                max_lora_rank=args.max_lora_rank,
            )
            raw_pairs = build_critic_raw_pairs(
                samples=source_samples,
                actor_lora_paths=actor_lora_paths,
                model_name=args.model_name,
                dataset_name=args.dataset,
                num_rounds=args.num_rounds,
                num_simulations=args.num_simulations,
                max_tokens=args.max_tokens,
                reward_threshold=args.reward_threshold,
                max_samples=args.max_delib_samples,
                seed=args.seed,
                device=args.device,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                max_lora_rank=args.max_lora_rank,
                engine=shared_engine,
            )
            _write_pool_cache(raw_pool_cache, pool_metadata, "raw_pairs", raw_pairs)

        routed_cache = _load_pool_cache(routed_pool_cache, pool_metadata)
        if routed_cache is not None:
            routed_items = [
                _routed_item_from_json(item)
                for item in routed_cache.get("routed_items", [])
            ]
            logger.info(f"  Loaded {len(routed_items)} shared routed items")
        else:
            routed_items = route_critic_raw_pairs(
                raw_pairs=raw_pairs,
                dataset_name=args.dataset,
                input_dir=input_dir,
                api_key=api_key,
                api_base=api_base,
                api_model=api_model,
                seed=args.seed,
                strict_classification=getattr(args, "strict_classification", True),
                max_classification_failure_rate=getattr(args, "max_classification_failure_rate", 0.0),
                max_classification_workers=getattr(args, "max_classification_workers", 4),
                request_timeout=getattr(args, "request_timeout", 60),
                max_retries=getattr(args, "max_retries", 5),
                retry_delay=getattr(args, "retry_delay", 5),
            )
            _write_pool_cache(
                routed_pool_cache,
                pool_metadata,
                "routed_items",
                [_routed_item_to_json(item) for item in routed_items],
            )

        # Once the shared raw/routed pool exists, per-skill work is cheap:
        # sample a different training mix from the same routed items.
        for critic_skill in args.critic_skills:
            cache_file = _pairs_cache_path(critic_skill)
            logger.info(f"\n--- Selecting pairs for Critic: {critic_skill} ---")
            preference_pairs, metrics = build_structured_critic_pairs(
                raw_pairs=raw_pairs,
                routed_items=routed_items,
                critic_skill=critic_skill,
                max_pairs=getattr(args, "max_pairs_per_critic", getattr(args, "target_pairs_per_critic", args.max_delib_samples)),
                seed=args.seed,
                min_real_specialty_items=getattr(args, "min_real_specialty_items", 16),
                pair_mix=getattr(args, "pair_mix", {"specialty": 0.75, "general": 0.25}),
            )

            if preference_pairs:
                critic_metrics[critic_skill] = metrics
                if len(preference_pairs) < args.min_pairs_per_critic:
                    inactive_critics[critic_skill] = {
                        "reason": "selected_pairs_below_min_pairs_per_critic",
                        "selected_pairs": len(preference_pairs),
                        "min_pairs_per_critic": args.min_pairs_per_critic,
                    }
                    logger.info(
                        f"  Critic '{critic_skill}' inactive: "
                        f"{len(preference_pairs)} selected pairs < "
                        f"{args.min_pairs_per_critic} minimum; will participate "
                        "as frozen_base with base model only"
                    )
                    continue

                # Save to disk for crash recovery
                with open(cache_file, "w") as f:
                    json.dump(preference_pairs, f)
                logger.info(f"  Cached {len(preference_pairs)} pairs to {cache_file}")
                all_pairs[critic_skill] = preference_pairs
            else:
                inactive_critics[critic_skill] = {
                    "reason": metrics.get("reason", "specialty_pool_below_active_threshold"),
                    "min_real_specialty_items": getattr(args, "min_real_specialty_items", 16),
                    "real_specialty_items": metrics.get("real_specialty_items", 0),
                }
                critic_metrics[critic_skill] = metrics
                logger.warning(f"  No preference pairs for '{critic_skill}', skipping")

        # Clean up shared engine before DPO training (GPU memory intensive)
        if shared_engine is not None:
            del shared_engine
            shared_engine = None
            _cleanup_gpu()

        # Phase 2: Train each critic (no engine needed, DPO runs in subprocess)
        for critic_skill, preference_pairs in all_pairs.items():
            logger.info(f"\n--- Training Critic: {critic_skill} ---")
            if critic_skill not in critic_metrics:
                real_specialty_items = sum(
                    1 for p in preference_pairs
                    if p.get("metadata", {}).get("source_bucket") == "specialty"
                )
                critic_metrics[critic_skill] = summarize_structured_critic_pairs(
                    preference_pairs,
                    real_specialty_items,
                )

            if len(preference_pairs) < args.min_pairs_per_critic:
                inactive_critics[critic_skill] = {
                    "reason": "cached_pairs_below_min_pairs_per_critic",
                    "selected_pairs": len(preference_pairs),
                    "min_pairs_per_critic": args.min_pairs_per_critic,
                }
                logger.info(
                    f"  Skipping '{critic_skill}': {len(preference_pairs)} pairs < "
                    f"{args.min_pairs_per_critic} minimum; will participate as "
                    "frozen_base with base model only"
                )
                continue

            checkpoint_path = train_critic_dpo(
                model_name=args.model_name,
                preference_pairs=preference_pairs,
                critic_skill=critic_skill,
                output_dir=output_dir,
                dataset_name=args.dataset,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_epochs=args.num_epochs,
                max_length=args.max_length,
                beta=args.beta,
                seed=args.seed,
                device=args.device,
            )

            critic_paths[critic_skill] = checkpoint_path

    finally:
        # Clean up shared engine if still alive (e.g. on early exit)
        if shared_engine is not None:
            del shared_engine
            _cleanup_gpu()

    # Save critic registry
    logger.info("\n[Step 5] Saving critic registry...")

    registry_file = os.path.join(output_dir, "critic_registry.json")
    registry_critics = {}
    for critic_skill in args.critic_skills:
        critic_metrics.setdefault(critic_skill, {
            "sample_count": 0,
            "unique_pair_count": 0,
            "duplicate_rate": 0.0,
            "source_bucket_counts": {},
            "source_bucket_ratios": {
                "general": 0.0,
                "specialty": 0.0,
                "calibration": 0.0,
            },
            "assigned_skill_counts": {},
        })
        if critic_skill in critic_paths:
            registry_critics[critic_skill] = {
                "critic_skill": critic_skill,
                "model_path": critic_paths[critic_skill],
                "base_model": args.model_name,
                "status": critic_metrics[critic_skill].get("status", "trained_specialist"),
                "participates": True,
                "base_model_only": False,
                "metrics": critic_metrics[critic_skill],
            }
        else:
            registry_critics[critic_skill] = {
                "critic_skill": critic_skill,
                "model_path": "",
                "base_model": args.model_name,
                "status": "frozen_base",
                "participates": True,
                "base_model_only": True,
                "inactive_reason": inactive_critics.get(critic_skill, {}),
                "metrics": critic_metrics[critic_skill],
            }

    with open(registry_file, "w") as f:
        json.dump({
            "critics": registry_critics,
            "metadata": {
                "base_model": args.model_name,
                "num_critics": len(registry_critics),
                "num_active_critics": len(critic_paths),
                "inactive_critics": inactive_critics,
                "active_selection": {
                    "min_pairs_per_critic": args.min_pairs_per_critic,
                    "min_real_specialty_items": getattr(args, "min_real_specialty_items", 16),
                    "target_pairs_per_critic": getattr(args, "target_pairs_per_critic", args.max_delib_samples),
                    "max_pairs_per_critic": getattr(args, "max_pairs_per_critic", args.max_delib_samples),
                },
                "training_mix": {
                    "pair_mix": getattr(args, "pair_mix", {"specialty": 0.75, "general": 0.25}),
                },
                "strict_classification": getattr(args, "strict_classification", True),
                "max_classification_failure_rate": getattr(args, "max_classification_failure_rate", 0.0),
                "critic_metrics": critic_metrics,
            },
        }, f, indent=2)

    logger.info(f"  Registry saved: {registry_file}")

    metrics_file = os.path.join(output_dir, "critic_training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({
            "critic_metrics": critic_metrics,
            "strict_classification": getattr(args, "strict_classification", True),
            "max_classification_failure_rate": getattr(args, "max_classification_failure_rate", 0.0),
        }, f, indent=2)
    logger.info(f"  Metrics saved: {metrics_file}")

    logger.info("=" * 60)
    logger.info("Critic diversification complete!")
    logger.info(f"  Trained LoRA critics: {len(critic_paths)}")
    for critic_skill, path in critic_paths.items():
        logger.info(f"    {critic_skill}: {path}")
    if inactive_critics:
        logger.info("  Frozen-base critics participate with base model only:")
        for critic_skill in args.critic_skills:
            if critic_skill not in critic_paths:
                logger.info(
                    f"    critic_{critic_skill}: frozen_base, "
                    "participates with base model only"
                )
        logger.info(f"  Frozen-base reasons: {inactive_critics}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
