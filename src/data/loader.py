"""Dataset loading for BoolQ, MMLU, BBH, SCIQ, ARC benchmarks."""

from __future__ import annotations

import logging
from typing import Optional

from datasets import load_dataset as hf_load_dataset, DatasetDict

from src.data.preprocessor import standardize_sample

logger = logging.getLogger(__name__)

# Supported datasets with their HuggingFace IDs and task types
DATASET_REGISTRY = {
    "boolq": {
        "hf_id": "google/boolq",
        "task_type": "yes_no",
        "splits": ["train", "validation"],
    },
    "mmlu": {
        "hf_id": "cais/mmlu",
        "task_type": "multiple_choice",
        "splits": ["test", "validation", "dev"],
        "config_name": "all",
    },
    "bbh": {
        "hf_id": "lukaemon/bbh",
        "task_type": "mixed",
        "splits": None,  # BBH needs custom splitting
    },
    "sciq": {
        "hf_id": "sciq",
        "task_type": "multiple_choice",
        "splits": ["train", "validation", "test"],
    },
    "arc": {
        "hf_id": "allenai/ai2_arc",
        "task_type": "multiple_choice",
        "splits": ["train", "validation", "test"],
        "config_name": "ARC-Challenge",
    },
    "math": {
        "hf_id": "EleutherAI/hendrycks_math",
        "task_type": "math",
        "splits": ["train", "test"],
        "config_name": None,  # Load all subconfigs and merge
    },
    "gsm8k": {
        "hf_id": "openai/gsm8k",
        "task_type": "math",
        "splits": ["train", "test"],
    },
}


def load_dataset(
    name: str,
    split_ratios: Optional[dict[str, float]] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> dict[str, list[dict]]:
    """
    Load and standardize a benchmark dataset.

    Args:
        name: Dataset name (boolq, mmlu, bbh, sciq, arc).
        split_ratios: Custom split ratios (only for BBH).
        seed: Random seed for reproducibility.
        cache_dir: HuggingFace cache directory.

    Returns:
        Dictionary with 'train', 'validation', 'test' keys,
        each containing a list of standardized samples.

    Raises:
        ValueError: If dataset name is not recognized.
    """
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Supported: {list(DATASET_REGISTRY.keys())}"
        )

    config = DATASET_REGISTRY[name]
    task_type = config["task_type"]

    logger.info(f"Loading dataset: {name} (task_type={task_type})")

    # Load from HuggingFace
    kwargs = {"path": config["hf_id"]}
    if config.get("config_name"):
        kwargs["name"] = config["config_name"]
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    # Special handling for MATH dataset: load all subconfigs and merge
    if name == "math":
        raw = _load_math_all(config["hf_id"], cache_dir)
    else:
        raw: DatasetDict = hf_load_dataset(**kwargs)

    # Handle BBH special splitting
    if name == "bbh":
        return _load_bbh(raw, task_type, split_ratios, seed)

    # Standard datasets with existing splits
    result = {}
    for split_name in ["train", "validation", "test"]:
        if split_name in raw:
            result[split_name] = [
                standardize_sample(sample, task_type)
                for sample in raw[split_name]
            ]
            logger.info(
                f"  {split_name}: {len(result[split_name])} samples"
            )

    return result


def _load_math_all(
    hf_id: str,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """Load all MATH subconfigs and merge into a single DatasetDict."""
    from datasets import concatenate_datasets, Dataset

    subconfigs = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
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

    # BBH in HuggingFace comes as separate configs per task
    # We need to split each task category separately
    train_samples = []
    val_samples = []
    test_samples = []

    if isinstance(raw, DatasetDict):
        for task_name, task_data in raw.items():
            task_samples = list(task_data)
            # Add task category to each sample
            for sample in task_samples:
                sample["bbh_task"] = task_name

            # Shuffle within this task
            rng.shuffle(task_samples)

            # Split this task's samples
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
        # Fallback for non-DatasetDict format
        # Convert to list and ensure each sample is a mutable dict
        all_samples = []
        for item in raw:
            # If item is already a dict, use it; otherwise create one
            if isinstance(item, dict):
                sample = item
            else:
                # For unexpected formats, wrap in a dict
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

    result = {
        "test": [standardize_sample(s, task_type) for s in test_samples],
        "validation": [standardize_sample(s, task_type) for s in val_samples],
        "train": [standardize_sample(s, task_type) for s in train_samples],
    }

    logger.info(
        f"  BBH total split: train={len(result['train'])}, "
        f"val={len(result['validation'])}, test={len(result['test'])}"
    )

    return result
