"""
Diversify Critics by training specialized LoRA adapters for each critic skill.

For each Critic:
1. Load trained Actor LoRA adapters (from Step 09)
2. Run Algorithm 1 (generate_trajectories) with Actor + base Critic
3. Extract Critic preference pairs (chosen=positive_critic, rejected=negative_critic)
4. Route by error profile and build a skill-specific training mixture
5. Train with DPO
6. Save to output/society/critics/{agent_id}/

This uses the same Algorithm 1 approach as society_trainer.py's
_generate_critic_pairs_algorithm1(), ensuring data consistency across
the bootstrap diversification phase and later alternating training.

Usage:
    python scripts/10_diversify_critics.py \
        --config configs/society/experiment_h100.yaml
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

setup_logging()
logger = logging.getLogger(__name__)

STEP_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dataset": "math",
    "cache_dir": "output/society",
    "input_dir": "output/society/classified",
    "actor_base_dir": "output/society/actors",
    "output_dir": "output/society/critics",
    "critic_skills": ["reasoning", "knowledge", "computation", "verification"],
    "max_delib_samples": 300,
    "num_rounds": 5,
    "num_simulations": 5,
    "max_tokens": 256,
    "reward_threshold": 0.0,
    "lora_r": 256,
    "lora_alpha": 512,
    "learning_rate": 5e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 1,
    "max_length": 2048,
    "beta": 0.1,
    "min_pairs_per_critic": 64,
    "min_specialty_items": 32,
    "min_specialty_ratio": 0.08,
    "specialty_ratio": 0.7,
    "general_ratio": 0.2,
    "calibration_ratio": 0.1,
    "seed": 42,
    "device": 0,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.65,
    "max_model_len": 4096,
    "max_lora_rank": 256,
    "api_key": "",
    "api_base": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "api_model": "glm-4-flash",
}


def parse_args():
    parser = __import__("argparse").ArgumentParser(
        description="Diversify Critics",
    )
    parser.add_argument(
        "--config", type=str, default="configs/society/experiment_h100.yaml",
        help="YAML config path.",
    )
    cli_args = parser.parse_args()

    cfg = ConfigManager.initialize(config_path=cli_args.config)
    return cfg.step("step04_diversify_critics", defaults=STEP_DEFAULTS).to_namespace()


# ============================================================
# LoRA Model Adapter — same pattern as society_trainer.py
# Bridges shared VLLMInference + LoRA to the generate() /
# generate_single() interface expected by trajectory.py.
# ============================================================

class _LoRAModelAdapter:
    """Wraps a shared VLLMInference engine with an optional LoRA adapter."""

    def __init__(self, engine: Any, lora_request: Any = None):
        self._engine = engine
        self._lora_request = lora_request

    def generate(self, prompts, max_tokens=256, temperature=0.7, **kwargs):
        if self._lora_request is not None:
            return self._generate_with_lora(prompts, max_tokens, temperature)
        return self._engine.generate(
            prompts, max_tokens=max_tokens, temperature=temperature, **kwargs,
        )

    def generate_single(self, prompt, max_tokens=256, temperature=0.7, **kwargs):
        results = self.generate(
            [prompt], max_tokens=max_tokens, temperature=temperature, **kwargs,
        )
        return results[0] if results else ""

    def _generate_with_lora(self, prompts, max_tokens, temperature):
        if hasattr(self._engine, "generate_with_lora"):
            return self._engine.generate_with_lora(
                prompts, self._lora_request,
                max_tokens=max_tokens, temperature=temperature,
            )
        if hasattr(self._engine, "_llm") and self._engine._llm is not None:
            from vllm import SamplingParams
            params = SamplingParams(max_tokens=max_tokens, temperature=temperature, n=1)
            outputs = self._engine._llm.generate(
                prompts, params, lora_request=self._lora_request,
            )
            return [c.text for o in outputs for c in o.outputs]
        raise RuntimeError(
            "Engine does not support LoRA generation. "
            "Create VLLMInference with enable_lora=True."
        )


def _build_adapters(
    engine: Any,
    lora_paths: Dict[str, str],
) -> Dict[str, _LoRAModelAdapter]:
    """Create _LoRAModelAdapter for each named agent path.

    Every provided path is required.  Continuing with a base actor after a
    failed LoRA load would corrupt the critic-diversification experiment.
    """
    from src.society.multi_deliberation import _load_lora_adapter

    adapters: Dict[str, _LoRAModelAdapter] = {}
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
        adapters[name] = _LoRAModelAdapter(engine, lora_req)
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


def build_critic_preference_pairs(
    samples: List[Dict],
    actor_lora_paths: Dict[str, str],
    critic_skill: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
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
    input_dir: str = "output/society/classified",
    api_key: str = "",
    api_base: str = "",
    api_model: str = "",
    min_specialty_items: int = 32,
    min_specialty_ratio: float = 0.08,
    specialty_ratio: float = 0.7,
    general_ratio: float = 0.2,
    calibration_ratio: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    Build DPO preference pairs for Critic training using Algorithm 1.

    This mirrors society_trainer.py's _generate_critic_pairs_algorithm1():
    1. Run generate_trajectories() with Actor LoRA + base Critic for each sample
    2. Extract Critic pairs: chosen=positive_critic, rejected=negative_critic
       where the Actor's negative response is wrong (error scenario for Critic)
    3. Route by error profile and sample a shared/specialty/calibration mix

    The Critic is always the base model (no LoRA) at this stage since no
    Critic LoRA adapters exist yet. Actors use their trained LoRA adapters
    from script 09, providing diverse real actor responses.
    """
    from src.inference.vllm_server import VLLMInference
    from src.algorithms.trajectory import generate_trajectories_batch
    from src.algorithms.reward import extract_answer, math_answers_equal, normalize_answer
    from src.society.diversity_split import DiversitySplit
    from src.society.agent_registry import resolve_critic_skill

    skill = resolve_critic_skill(critic_skill)
    logger.info(f"  Generating Critic preference pairs for '{critic_skill}' via Algorithm 1")

    # Collect all unique LoRA paths to determine max_loras for engine
    all_lora_paths = {k: v for k, v in actor_lora_paths.items() if v}
    # Need at most 1 actor LoRA at a time
    max_loras = 1 if all_lora_paths else 0

    n_samples = min(len(samples), max_samples)
    preference_pairs: List[Dict] = []
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

        # Build adapters for all actors
        adapters = _build_adapters(engine, all_lora_paths)
        if not adapters:
            adapters = {"base_actor": _LoRAModelAdapter(engine, None)}
        actor_names = list(adapters.keys())

        # Base Critic adapter (no LoRA)
        critic_adapter = _LoRAModelAdapter(engine, None)

        # Step 1: Run Algorithm 1 with samples grouped by round-robin actor.
        # Each group batches natural deliberation, guided prompts, and MC
        # rollout phases across samples.
        raw_pairs: List[Dict] = []
        actor_groups: Dict[str, List[Dict]] = {}
        for i, sample in enumerate(samples[:n_samples]):
            actor_groups.setdefault(actor_names[i % len(actor_names)], []).append(sample)

        for actor_name, group_samples in actor_groups.items():
            actor_adapter = adapters[actor_name]
            logger.info(
                f"    Algorithm 1 batch for '{critic_skill}' with actor "
                f"{actor_name}: {len(group_samples)} samples"
            )

            algo_pairs = generate_trajectories_batch(
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
            )
            logger.info(
                f"    Algorithm 1 batch for actor {actor_name} produced "
                f"{len(algo_pairs)} pairs"
            )

            # Extract Critic preference pairs from Algorithm 1 results
            for pair in algo_pairs:
                # pair has: positive, negative (actor),
                #           positive_critic, negative_critic,
                #           delta, direction
                sample = pair.get("sample", {})
                task_type = sample.get("task_type", "math")
                correct_answer = sample.get("answer", "")
                negative_actor = pair["negative"]
                negative_answer = extract_answer(negative_actor, task_type)

                # Only keep pairs where the actor's negative response was wrong
                # (error scenario — this is where Critic feedback matters)
                if task_type == "math":
                    is_wrong = not math_answers_equal(
                        negative_answer or "", correct_answer,
                    )
                else:
                    is_wrong = normalize_answer(
                        negative_answer or "", task_type,
                    ) != normalize_answer(correct_answer, task_type)

                if is_wrong:
                    raw_pairs.append({
                        "sample": sample,
                        "chosen": pair["positive_critic"],
                        "rejected": pair["negative_critic"],
                        "actor_response": negative_actor,
                        "actor_answer": negative_answer,
                        "correct_answer": correct_answer,
                        "task_type": task_type,
                        "delta": pair["delta"],
                        "direction": pair["direction"],
                    })

        logger.info(
            f"  Algorithm 1 produced {len(raw_pairs)} raw critic pairs "
            f"for '{critic_skill}'"
        )

        # Step 2: Route by error profile, then build the critic training mix.
        if raw_pairs:
            splitter = DiversitySplit(
                balance=False, seed=seed, use_api=True,
                cache_dir=input_dir,
                pre_classified_file=os.path.join(input_dir, "classified_data.json"),
                api_key=api_key,
                api_base=api_base,
                api_model=api_model,
            )
            routed_items = splitter.split_by_error_profile(
                samples=[p["sample"] for p in raw_pairs],
                responses=[p["actor_response"] for p in raw_pairs],
                correct_answers=[p["correct_answer"] for p in raw_pairs],
                extracted_answers=[p["actor_answer"] or "" for p in raw_pairs],
                dataset_name=dataset_name,
            )

            raw_skill_dist = Counter(
                item.skill.value if item.skill else "general"
                for item in routed_items
            )
            unique_raw_pairs = {
                (item.sample.get("question", ""), item.response)
                for item in routed_items
            }
            logger.info(
                f"  [{critic_skill}] raw routed profile distribution: "
                f"{dict(raw_skill_dist)}"
            )
            logger.info(
                f"  [{critic_skill}] raw unique_pairs: "
                f"{len(unique_raw_pairs)} / {len(routed_items)}"
            )

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
                    f"  [{critic_skill}] inactive: specialty pool below "
                    f"threshold; no specialist DPO pairs selected"
                )
                return []

            # O(1) lookup by (question, actor_response)
            pair_index: Dict[tuple, Dict] = {}
            for p in raw_pairs:
                key = (p["sample"].get("question", ""), p["actor_response"])
                pair_index[key] = p

            for item in critic_items:
                key = (item.sample.get("question", ""), item.response)
                p = pair_index.get(key)
                if p is not None:
                    preference_pairs.append({
                        "sample": p["sample"],
                        "chosen": p["chosen"],
                        "rejected": p["rejected"],
                        "actor_response": p.get("actor_response", ""),
                        "metadata": {
                            "target_skill": skill.value,
                            "assigned_skill": item.skill.value if item.skill else "general",
                            "source_bucket": item.source_bucket,
                            "routing_weight": item.weight,
                            "error_profile": item.profile,
                            "delta": p["delta"],
                            "direction": p["direction"],
                        },
                    })

            logger.info(
                f"  {len(preference_pairs)}/{len(raw_pairs)} pairs "
                f"selected for skill '{skill.value}'"
            )

            # Log source-bucket and assigned-skill distributions to verify routing.
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
        else:
            logger.warning(f"  No raw pairs produced for '{critic_skill}'")

        if _owns_engine:
            del engine
            _cleanup_gpu()

    except Exception as e:
        logger.error(f"  Failed to generate Critic pairs for '{critic_skill}': {e}")
        if _owns_engine:
            _cleanup_gpu()

    return preference_pairs


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
    from src.prompts.formatter import format_prompt
    from src.prompts.templates import PromptType

    model_type = detect_model_type(model_name)

    # Create output directory
    critic_output_dir = os.path.join(output_dir, f"critic_{critic_skill}")
    os.makedirs(critic_output_dir, exist_ok=True)

    logger.info(f"  Training DPO for '{critic_skill}'...")
    logger.info(f"    Pairs: {len(preference_pairs)}")
    logger.info(f"    Output: {critic_output_dir}")

    # Reconstruct full prompts using the same template as inference
    # (DELIBERATION_CRITIC with actor_response)
    prompts = []
    for p in preference_pairs:
        sample = p.get("sample", {})
        actor_resp = p.get("actor_response", "")
        if actor_resp:
            prompt = format_prompt(
                dataset_name, PromptType.DELIBERATION_CRITIC, sample,
                actor_response=actor_resp,
            )
        else:
            prompt = format_prompt(
                dataset_name,
                PromptType.SINGLE_SHOT,
                sample,
                include_answer_contract=False,
            )
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
    else:
        logger.warning(
            "GLM_API_KEY not set (neither config nor env var). "
            "Unseen raw pairs from Algorithm 1 will be routed to general pool."
        )
    api_base = getattr(args, "api_base", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
    api_model = getattr(args, "api_model", "glm-4-flash")

    # Setup directories
    input_dir = args.input_dir
    output_dir = args.output_dir
    actor_dir = args.actor_base_dir
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Diversify Critics (Algorithm 1 trajectory-based)")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Input dir: {input_dir}")
    logger.info(f"  Actor dir: {actor_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Critic skills: {args.critic_skills}")
    logger.info(
        "  Active critic thresholds: "
        f"min_pairs={args.min_pairs_per_critic}, "
        f"min_specialty_items={args.min_specialty_items}, "
        f"min_specialty_ratio={args.min_specialty_ratio}"
    )
    if args.min_pairs_per_critic > args.max_delib_samples:
        logger.warning(
            "min_pairs_per_critic > max_delib_samples; no critic can become "
            "trained unless the selected training mix can exceed the sample cap. "
            "Current mix uses max_items=max_delib_samples, so this configuration "
            "will freeze every critic that relies on generated pairs."
        )
    logger.info(
        "  Training mix ratios: "
        f"specialty={args.specialty_ratio}, "
        f"general={args.general_ratio}, "
        f"calibration={args.calibration_ratio}"
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

    # Load original dataset for Algorithm 1 input
    logger.info("[Step 2] Loading dataset for Algorithm 1...")
    from src.data.loader import load_dataset
    dataset = load_dataset(args.dataset)

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

    # Create a single vLLM engine shared across all critic skills
    from src.inference.vllm_server import VLLMInference
    shared_engine = None

    all_lora_paths = {k: v for k, v in actor_lora_paths.items() if v}
    try:
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
    except Exception as e:
        logger.warning(f"Failed to create shared vLLM engine: {e}")
        logger.warning("Will attempt per-error-type engine creation")

    def _pairs_cache_path(critic_skill):
        return os.path.join(output_dir, f"pairs_{critic_skill}_adaptive.json")

    try:
        # Phase 1: Generate preference pairs for ALL critic skills (with disk cache)
        all_pairs = {}
        for critic_skill in args.critic_skills:
            cache_file = _pairs_cache_path(critic_skill)
            if os.path.exists(cache_file):
                logger.info(f"\n--- Loading cached pairs for Critic: {critic_skill} ---")
                with open(cache_file) as f:
                    cached = json.load(f)
                all_pairs[critic_skill] = cached
                logger.info(f"  Loaded {len(cached)} cached pairs for '{critic_skill}'")
                continue

            logger.info(f"\n--- Building pairs for Critic: {critic_skill} ---")

            # Every critic sees the same incorrect-response source pool; the
            # profile router builds skill-specific mixtures after generation.
            sample_ids = incorrect_sample_ids
            samples_for_type = [
                id_to_sample[sid]
                for sid in sample_ids
                if sid in id_to_sample
            ]

            if not samples_for_type:
                logger.warning(
                    f"  No incorrect samples found. Skipping Critic '{critic_skill}'."
                )
                continue

            logger.info(f"  {len(samples_for_type)} source samples for '{critic_skill}'")

            preference_pairs = build_critic_preference_pairs(
                samples=samples_for_type,
                actor_lora_paths=actor_lora_paths,
                critic_skill=critic_skill,
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
                input_dir=input_dir,
                api_key=api_key,
                api_base=api_base,
                api_model=api_model,
                min_specialty_items=args.min_specialty_items,
                min_specialty_ratio=args.min_specialty_ratio,
                specialty_ratio=args.specialty_ratio,
                general_ratio=args.general_ratio,
                calibration_ratio=args.calibration_ratio,
            )

            if preference_pairs:
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
                    "reason": "specialty_pool_below_active_threshold",
                    "min_specialty_items": args.min_specialty_items,
                    "min_specialty_ratio": args.min_specialty_ratio,
                }
                logger.warning(f"  No preference pairs for '{critic_skill}', skipping")

        # Clean up shared engine before DPO training (GPU memory intensive)
        if shared_engine is not None:
            del shared_engine
            shared_engine = None
            _cleanup_gpu()

        # Phase 2: Train each critic (no engine needed, DPO runs in subprocess)
        for critic_skill, preference_pairs in all_pairs.items():
            logger.info(f"\n--- Training Critic: {critic_skill} ---")

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
        if critic_skill in critic_paths:
            registry_critics[critic_skill] = {
                "critic_skill": critic_skill,
                "model_path": critic_paths[critic_skill],
                "base_model": args.model_name,
                "status": "active",
                "participates": True,
                "base_model_only": False,
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
                    "min_specialty_items": args.min_specialty_items,
                    "min_specialty_ratio": args.min_specialty_ratio,
                },
                "training_mix": {
                    "specialty_ratio": args.specialty_ratio,
                    "general_ratio": args.general_ratio,
                    "calibration_ratio": args.calibration_ratio,
                },
            },
        }, f, indent=2)

    logger.info(f"  Registry saved: {registry_file}")

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
