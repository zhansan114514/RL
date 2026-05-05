"""Dataset loading for BoolQ, MMLU, BBH, SCIQ, ARC benchmarks.

MMLU is routed to its own loader (``src.data.mmlu``) for correct
``auxiliary_train -> train`` mapping and per-subject metadata.
All other datasets use the standard HuggingFace path.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from datasets import load_dataset as hf_load_dataset, DatasetDict

from src.data.preprocessor import standardize_sample

logger = logging.getLogger(__name__)

# Supported datasets with their HuggingFace IDs and task types
DATASET_REGISTRY = {
    "boolq": {
        "hf_id": "google/boolq",
        "task_type": "yes_no",
        "splits": ["train", "validation"],
        "loader": "standard",
    },
    "mmlu": {
        "hf_id": "cais/mmlu",
        "task_type": "multiple_choice",
        "splits": ["auxiliary_train", "validation", "test", "dev"],
        "config_name": "all",
        "loader": "mmlu",
    },
    "bbh": {
        "hf_id": "lukaemon/bbh",
        "task_type": "mixed",
        "splits": None,
        "loader": "bbh",
    },
    "sciq": {
        "hf_id": "allenai/sciq",
        "task_type": "multiple_choice",
        "splits": ["train", "validation", "test"],
        "loader": "standard",
    },
    "arc": {
        "hf_id": "allenai/ai2_arc",
        "task_type": "multiple_choice",
        "splits": ["train", "validation", "test"],
        "config_name": "ARC-Challenge",
        "loader": "standard",
    },
    "math": {
        "hf_id": "EleutherAI/hendrycks_math",
        "task_type": "math",
        "splits": ["train", "test"],
        "config_name": None,
        "loader": "math",
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "task_type": "math",
        "splits": ["train", "test"],
        "loader": "standard",
    },
}


def load_dataset(
    name: str,
    split_ratios: Optional[dict[str, float]] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    sampling: Optional[dict[str, dict[str, Any]]] = None,
    mmlu_load_mode: str = "by_subject",
) -> dict[str, list[dict]]:
    """
    Load and standardize a benchmark dataset.

    Args:
        name: Dataset name (boolq, mmlu, bbh, sciq, arc, math, gsm8k).
        split_ratios: Custom split ratios (only for BBH).
        seed: Random seed for reproducibility.
        cache_dir: HuggingFace cache directory.
        sampling: Per-split sampling config, e.g.
            {"train": {"strategy": "random", "max_samples": 100, "seed_offset": 0}}.
        mmlu_load_mode: ``"all"`` or ``"by_subject"`` (default).

    Returns:
        Dictionary with split names as keys, each containing a list of
        standardized samples.

    Raises:
        ValueError: If dataset name is not recognized or data is invalid.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Supported: {list(DATASET_REGISTRY.keys())}"
        )

    config = DATASET_REGISTRY[name]
    task_type = config["task_type"]
    loader_type = config.get("loader", "standard")

    logger.info(f"Loading dataset: {name} (task_type={task_type}, loader={loader_type})")

    # Route to specialized loaders
    if loader_type == "mmlu":
        from src.data.mmlu import load_mmlu
        data = load_mmlu(cache_dir=cache_dir, load_mode=mmlu_load_mode)
    elif loader_type == "math":
        raw = _load_math_all(config["hf_id"], cache_dir)
        data = _standardize_splits(raw, task_type, dataset_name=name)
    elif loader_type == "bbh":
        kwargs = {"path": config["hf_id"]}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        raw: DatasetDict = hf_load_dataset(**kwargs)
        data = _load_bbh(raw, task_type, split_ratios, seed)
    else:
        # Standard datasets
        kwargs = {"path": config["hf_id"]}
        if config.get("config_name"):
            kwargs["name"] = config["config_name"]
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        raw = hf_load_dataset(**kwargs)
        data = _standardize_splits(raw, task_type, dataset_name=name)

    # Validate
    validate_dataset_bundle(name, data)

    # Apply sampling if configured
    if sampling:
        from src.data.sampler import apply_sampling
        data = apply_sampling(data, sampling, base_seed=seed)

    # Log final sizes
    for split_name, samples in data.items():
        logger.info(f"  {split_name}: {len(samples)} samples")

    return data


def _standardize_splits(
    raw: DatasetDict,
    task_type: str,
    dataset_name: str = "",
) -> dict[str, list[dict]]:
    """Standardize splits from a standard HuggingFace DatasetDict.

    Injects dataset, source_split, and source_index metadata before
    standardization, consistent with the MMLU loader.
    """
    result = {}
    for split_name in ["train", "validation", "test"]:
        if split_name in raw:
            standardized = []
            for i, sample in enumerate(raw[split_name]):
                sample = dict(sample)
                sample.setdefault("dataset", dataset_name)
                sample["source_split"] = split_name
                sample["source_index"] = i
                standardized.append(standardize_sample(sample, task_type))
            result[split_name] = standardized
    return result


def validate_dataset_bundle(
    name: str,
    data: dict[str, list[dict]],
) -> None:
    """Validate that a loaded dataset has the required splits and is non-empty.

    Raises:
        ValueError: If required splits are missing or empty.
    """
    if name == "mmlu":
        required = ["train", "validation", "test"]
    else:
        required = ["train"]

    for split in required:
        if split not in data:
            raise ValueError(
                f"Dataset '{name}': missing required split '{split}'. "
                f"Available: {list(data.keys())}"
            )
        if not data[split]:
            raise ValueError(
                f"Dataset '{name}': split '{split}' is empty. "
                f"Check that the data source is correct."
            )


def _load_math_all(
    hf_id: str,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """Load all MATH subconfigs and merge into a single DatasetDict."""
    from datasets import concatenate_datasets

    subconfigs = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
        "linear_algebra", "abstract_algebra", "college_mathematics",
        "miscellaneous",
    ]

    merged = {}
    for split in ["train", "test"]:
        parts = []
        for cfg in subconfigs:
            try:
                ds = hf_load_dataset(hf_id, cfg, split=split, cache_dir=cache_dir)
                parts.append(ds)
            except Exception:
                pass
        if parts:
            merged[split] = concatenate_datasets(parts)
            logger.info(f"  MATH {split}: {len(merged[split])} samples from {len(parts)} subconfigs")

    return DatasetDict(merged)


def _load_bbh(
    raw: DatasetDict,
    task_type: str,
    split_ratios: Optional[dict[str, float]] = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Load BBH with custom train/val/test splits.

    Paper specifies: "roughly 25% and 10% of the questions from each category".
    This means we need to split each BBH task category separately.
    """
    import random

    ratios = split_ratios or {"test": 0.25, "validation": 0.10}
    rng = random.Random(seed)

    train_samples = []
    val_samples = []
    test_samples = []

    if isinstance(raw, DatasetDict):
        for task_name, task_data in raw.items():
            task_samples = list(task_data)
            for sample in task_samples:
                sample["bbh_task"] = task_name

            rng.shuffle(task_samples)

            n_task = len(task_samples)
            n_test = int(n_task * ratios["test"])
            n_val = int(n_task * ratios["validation"])

            task_test = task_samples[:n_test]
            task_val = task_samples[n_test : n_test + n_val]
            task_train = task_samples[n_test + n_val :]

            test_samples.extend(task_test)
            val_samples.extend(task_val)
            train_samples.extend(task_train)

            logger.info(
                f"  Task {task_name}: train={len(task_train)}, "
                f"val={len(task_val)}, test={len(task_test)}"
            )
    else:
        all_samples = []
        for item in raw:
            if isinstance(item, dict):
                sample = item
            else:
                sample = {"data": item}
            sample["bbh_task"] = "unknown"
            all_samples.append(sample)
        rng.shuffle(all_samples)

        n_total = len(all_samples)
        n_test = int(n_total * ratios["test"])
        n_val = int(n_total * ratios["validation"])

        test_samples = all_samples[:n_test]
        val_samples = all_samples[n_test : n_test + n_val]
        train_samples = all_samples[n_test + n_val :]

    result = {}
    for split_name, samples in [
        ("test", test_samples),
        ("validation", val_samples),
        ("train", train_samples),
    ]:
        standardized = []
        for i, sample in enumerate(samples):
            sample.setdefault("dataset", "bbh")
            sample["source_split"] = split_name
            sample["source_index"] = i
            # Preserve bbh_task as subject for per-group analysis
            sample.setdefault("subject", sample.get("bbh_task", "unknown"))
            standardized.append(standardize_sample(sample, task_type))
        result[split_name] = standardized

    logger.info(
        f"  BBH total split: train={len(result['train'])}, "
        f"val={len(result['validation'])}, test={len(result['test'])}"
    )

    return result
