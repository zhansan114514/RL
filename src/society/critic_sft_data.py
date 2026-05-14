"""Data construction helpers for first-round Critic SFT."""

from __future__ import annotations

import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.algorithms.reward import (
    extract_answer,
    math_answers_equal,
    normalize_answer,
)
from src.parsing.critic_parser import parse_critic_response
from src.prompts.critic_sft_prompts import (
    build_critic_sft_prompt,
    render_critic_sft_feedback,
)
from src.society.agent_registry import resolve_critic_skill


CRITIC_SFT_CASE_TYPES = {"correction", "keep"}


def sample_id_for(sample: dict[str, Any], fallback_index: int | None = None) -> str:
    """Return a stable sample id from standardized dataset metadata."""

    for key in ("sample_id", "id", "uid"):
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    dataset = str(sample.get("dataset") or "sample")
    split = str(sample.get("source_split") or "")
    index = sample.get("source_index")
    if index is not None:
        return "_".join(part for part in (dataset, split, str(index)) if part)
    if fallback_index is not None:
        return f"{dataset}_{fallback_index:06d}"
    question = re.sub(r"\W+", "_", str(sample.get("question", ""))).strip("_")
    return question[:80] or "sample"


def answers_match(predicted: str | None, gold: str | None, task_type: str) -> bool:
    """Compare predicted and gold answers with task-aware normalization."""

    if predicted is None or gold is None:
        return False
    pred = str(predicted).strip()
    label = str(gold).strip()
    if not pred or not label:
        return False
    if task_type == "math":
        return math_answers_equal(pred, label)
    return normalize_answer(pred, task_type) == normalize_answer(label, task_type)


def build_actor_output(
    *,
    sample_id: str,
    sample: dict[str, Any],
    actor_name: str,
    actor_style: str,
    response: str,
) -> dict[str, Any]:
    """Build one first-round Actor output record."""

    task_type = str(sample.get("task_type") or "multiple_choice")
    answer = extract_answer(response, task_type)
    gold = sample.get("answer")
    return {
        "sample_id": sample_id,
        "actor_name": actor_name,
        "actor_style": actor_style,
        "actor_response": response,
        "actor_answer": answer or "",
        "actor_correct": answers_match(answer, gold, task_type),
        "task_type": task_type,
    }


def build_other_actor_summary(
    *,
    target_actor_name: str,
    actor_outputs: list[dict[str, Any]],
    max_chars_per_actor: int = 500,
) -> str:
    """Summarize peer Actor outputs without revealing correctness labels."""

    peers = [
        output for output in actor_outputs
        if output.get("actor_name") != target_actor_name
    ]
    if not peers:
        return "No other Actor responses were available."

    lines = ["Other actors summary:"]
    answers: list[str] = []
    for idx, peer in enumerate(peers, start=1):
        name = str(peer.get("actor_name") or f"actor_{idx}")
        answer = str(peer.get("actor_answer") or "unknown")
        answers.append(answer)
        rationale = _clip_response(str(peer.get("actor_response") or ""), max_chars_per_actor)
        lines.append(f"{idx}. {name} answered {answer}. Main reason: {rationale}")

    unique_answers = sorted({answer for answer in answers if answer and answer != "unknown"})
    if unique_answers:
        lines.append("")
        lines.append("Peer disagreement:")
        lines.append(f"- Answers observed: {', '.join(unique_answers)}.")
        if len(unique_answers) > 1:
            lines.append("- At least one peer answer conflicts with another peer.")
        else:
            lines.append("- Peer answers agree with each other.")
    return "\n".join(lines).strip()


def build_critic_sft_candidates(
    *,
    samples: list[dict[str, Any]],
    actor_outputs: list[dict[str, Any]],
    critic_skills: Iterable[str],
    dataset_name: str,
    summary_max_chars_per_actor: int = 500,
    allow_correction_without_correct_peer: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build unvalidated Critic SFT candidates from first-round Actor outputs."""

    skills = [resolve_critic_skill(skill).value for skill in critic_skills]
    sample_by_id = {
        sample_id_for(sample, idx): sample
        for idx, sample in enumerate(samples)
    }
    outputs_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for output in actor_outputs:
        sid = str(output.get("sample_id") or "")
        if sid:
            outputs_by_sample[sid].append(output)

    examples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for sid, sample in sample_by_id.items():
        gold = str(sample.get("answer") or "").strip()
        if not gold:
            skipped["missing_gold_answer"] += 1
            continue
        sample_outputs = outputs_by_sample.get(sid, [])
        if not sample_outputs:
            skipped["missing_actor_outputs"] += 1
            continue

        for target in sample_outputs:
            actor_answer = str(target.get("actor_answer") or "").strip()
            if not actor_answer:
                skipped["unparseable_actor_answer"] += 1
                continue
            actor_name = str(target.get("actor_name") or "")
            peers = [
                output for output in sample_outputs
                if output.get("actor_name") != actor_name
            ]
            others_have_correct = any(bool(peer.get("actor_correct")) for peer in peers)
            others_have_wrong = any(
                str(peer.get("actor_answer") or "").strip()
                and not bool(peer.get("actor_correct"))
                for peer in peers
            )
            summary = build_other_actor_summary(
                target_actor_name=actor_name,
                actor_outputs=sample_outputs,
                max_chars_per_actor=summary_max_chars_per_actor,
            )

            if bool(target.get("actor_correct")):
                if not others_have_wrong:
                    skipped["keep_without_wrong_peer"] += 1
                    continue
                case_type = "keep"
                peer_support = "wrong_peer_conflict"
            else:
                if not others_have_correct and not allow_correction_without_correct_peer:
                    skipped["correction_without_correct_peer"] += 1
                    continue
                case_type = "correction"
                peer_support = (
                    "has_correct_peer" if others_have_correct else "none_correct"
                )

            for skill in skills:
                examples.append(_build_example(
                    sample_id=sid,
                    sample=sample,
                    target=target,
                    other_actor_summary=summary,
                    peer_outputs=peers,
                    critic_skill=skill,
                    case_type=case_type,
                    dataset_name=dataset_name,
                    others_have_correct=others_have_correct,
                    others_have_wrong=others_have_wrong,
                    peer_support=peer_support,
                ))

    metrics = summarize_candidates(examples)
    metrics["skipped_counts"] = dict(skipped)
    metrics["source_sample_count"] = len(samples)
    metrics["actor_output_count"] = len(actor_outputs)
    return examples, metrics


def select_examples_per_skill(
    examples: list[dict[str, Any]],
    *,
    critic_skills: Iterable[str],
    max_examples_per_critic: int | None,
    correction_ratio: float,
    keep_ratio: float,
    seed: int = 42,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Deduplicate and select a correction/keep mix for each Critic skill."""

    rng = random.Random(seed)
    skills = [resolve_critic_skill(skill).value for skill in critic_skills]
    by_skill: dict[str, list[dict[str, Any]]] = {skill: [] for skill in skills}
    for example in _dedupe_examples(examples):
        skill = str(example.get("critic_skill") or "")
        if skill in by_skill:
            by_skill[skill].append(example)

    selected: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, Any] = {}
    ratio_total = max(correction_ratio + keep_ratio, 1e-9)
    correction_share = max(0.0, correction_ratio / ratio_total)

    for skill in skills:
        correction = [
            ex for ex in by_skill[skill]
            if ex.get("case_type") == "correction"
        ]
        keep = [ex for ex in by_skill[skill] if ex.get("case_type") == "keep"]
        rng.shuffle(correction)
        rng.shuffle(keep)

        if max_examples_per_critic is None or max_examples_per_critic <= 0:
            rows = correction + keep
        else:
            target_correction = int(round(max_examples_per_critic * correction_share))
            target_keep = max_examples_per_critic - target_correction
            selected_correction = correction[:target_correction]
            selected_keep = keep[:target_keep]
            remaining = (
                max_examples_per_critic
                - len(selected_correction)
                - len(selected_keep)
            )
            if remaining > 0:
                spillover = (
                    correction[len(selected_correction):]
                    + keep[len(selected_keep):]
                )
                selected_extra = spillover[:remaining]
            else:
                selected_extra = []
            rows = selected_correction + selected_keep + selected_extra

        rows = sorted(rows, key=lambda ex: str(ex.get("id") or ""))
        selected[skill] = rows
        metrics[skill] = summarize_candidates(rows)
        metrics[skill]["available_examples"] = len(by_skill[skill])
        metrics[skill]["available_correction"] = len(correction)
        metrics[skill]["available_keep"] = len(keep)

    return selected, metrics


def examples_to_sft_rows(examples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return training rows containing only prompt/response."""

    return [
        {"prompt": str(example.get("prompt") or ""), "response": str(example.get("response") or "")}
        for example in examples
    ]


def attach_revision_result(
    example: dict[str, Any],
    *,
    revised_response: str,
) -> tuple[dict[str, Any], bool, str]:
    """Attach Actor revision output and return whether the case passes."""

    updated = copy.deepcopy(example)
    metadata = updated.setdefault("metadata", {})
    task = metadata.get("task", {})
    task_type = str(task.get("task_type") or "multiple_choice")
    target_answer = str(metadata.get("target_answer") or task.get("answer") or "")
    revised_answer = extract_answer(revised_response, task_type)
    final_correct = answers_match(revised_answer, target_answer, task_type)
    metadata["revised_response"] = revised_response
    metadata["revised_answer"] = revised_answer or ""
    metadata["final_correct"] = final_correct
    if not revised_answer:
        return updated, False, "missing_revised_answer"
    if not final_correct:
        return updated, False, "revised_answer_incorrect"
    return updated, True, "passed"


def summarize_candidates(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize Critic SFT examples."""

    case_counts = Counter(str(ex.get("case_type") or "unknown") for ex in examples)
    skill_counts = Counter(str(ex.get("critic_skill") or "unknown") for ex in examples)
    actor_counts = Counter(
        str(ex.get("metadata", {}).get("actor_name") or "unknown")
        for ex in examples
    )
    subject_counts = Counter(
        str(
            ex.get("metadata", {})
            .get("task", {})
            .get("subject")
            or ex.get("metadata", {})
            .get("task", {})
            .get("category")
            or "unknown"
        )
        for ex in examples
    )
    parseable = 0
    for ex in examples:
        parsed = parse_critic_response(
            str(ex.get("response") or ""),
            str(ex.get("metadata", {}).get("task", {}).get("task_type") or "multiple_choice"),
        )
        if parsed.usable_for_feedback and parsed.has_answer_correct:
            parseable += 1

    return {
        "num_examples": len(examples),
        "case_type_counts": dict(case_counts),
        "correction_count": case_counts.get("correction", 0),
        "keep_count": case_counts.get("keep", 0),
        "critic_skill_counts": dict(skill_counts),
        "actor_counts": dict(actor_counts),
        "subject_counts": dict(subject_counts),
        "subject_coverage": len(subject_counts),
        "parseable_judgement_count": parseable,
        "parseable_judgement_rate": parseable / len(examples) if examples else 0.0,
        "others_have_correct_rate": _metadata_rate(examples, "others_have_correct"),
        "others_have_wrong_rate": _metadata_rate(examples, "others_have_wrong"),
    }


def save_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write JSONL with UTF-8 encoding."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL rows."""

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _build_example(
    *,
    sample_id: str,
    sample: dict[str, Any],
    target: dict[str, Any],
    other_actor_summary: str,
    peer_outputs: list[dict[str, Any]],
    critic_skill: str,
    case_type: str,
    dataset_name: str,
    others_have_correct: bool,
    others_have_wrong: bool,
    peer_support: str,
) -> dict[str, Any]:
    task = _task_metadata(sample)
    actor_name = str(target.get("actor_name") or "")
    actor_answer = str(target.get("actor_answer") or "")
    target_answer = str(sample.get("answer") or "")
    prompt = build_critic_sft_prompt(
        sample=sample,
        dataset_name=dataset_name,
        critic_skill=critic_skill,
        actor_name=actor_name,
        actor_response=str(target.get("actor_response") or ""),
        actor_answer=actor_answer,
        other_actor_summary=other_actor_summary,
    )
    response = render_critic_sft_feedback(
        critic_skill=critic_skill,
        case_type=case_type,
        actor_name=actor_name,
        actor_answer=actor_answer,
        target_answer=target_answer,
        other_actor_summary=other_actor_summary,
    )
    safe_id = _safe_id(
        f"{sample_id}_{actor_name}_critic_{critic_skill}_{case_type}"
    )
    return {
        "id": safe_id,
        "critic_skill": critic_skill,
        "case_type": case_type,
        "prompt": prompt,
        "response": response,
        "metadata": {
            "sample_id": sample_id,
            "task": task,
            "actor_name": actor_name,
            "actor_style": str(target.get("actor_style") or ""),
            "actor_current_answer": actor_answer,
            "actor_current_response": str(target.get("actor_response") or ""),
            "other_actor_summary": other_actor_summary,
            "other_actor_answers": {
                str(peer.get("actor_name") or ""): str(peer.get("actor_answer") or "")
                for peer in peer_outputs
            },
            "target_answer": target_answer,
            "revised_response": "",
            "revised_answer": "",
            "final_correct": None,
            "actor_initial_correct": bool(target.get("actor_correct")),
            "others_have_correct": others_have_correct,
            "others_have_wrong": others_have_wrong,
            "peer_support": peer_support,
        },
    }


def _task_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "question",
        "passage",
        "choices",
        "answer",
        "task_type",
        "subject",
        "category",
        "dataset",
        "source_split",
        "source_index",
    )
    return {key: copy.deepcopy(sample.get(key)) for key in keys if key in sample}


def _clip_response(text: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return "No rationale available."
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def _dedupe_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for example in examples:
        metadata = example.get("metadata", {})
        task = metadata.get("task", {})
        key = (
            str(example.get("critic_skill") or ""),
            str(example.get("case_type") or ""),
            str(task.get("question") or metadata.get("sample_id") or ""),
            str(metadata.get("actor_name") or ""),
            str(metadata.get("actor_current_response") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(example)
    return deduped


def _metadata_rate(examples: list[dict[str, Any]], key: str) -> float:
    if not examples:
        return 0.0
    count = sum(1 for ex in examples if bool(ex.get("metadata", {}).get(key)))
    return count / len(examples)


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return safe[:220] or "critic_sft_example"
