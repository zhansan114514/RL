"""Tests for prompt template management and formatting."""

import pytest

from src.prompts.templates import (
    PromptType,
    get_prompt_template,
    get_available_datasets,
    get_available_prompt_types,
)
from src.prompts.formatter import format_prompt, _format_responses


class TestTemplateRetrieval:
    """Test template lookup and validation."""

    def test_get_boolq_single_shot(self):
        template = get_prompt_template("boolq", PromptType.SINGLE_SHOT)
        assert "yes-no question" in template
        assert "{question}" in template
        assert "{passage}" in template

    def test_get_boolq_guided_single_shot(self):
        template = get_prompt_template("boolq", PromptType.GUIDED_SINGLE_SHOT)
        assert "{target_answer}" in template
        assert "{question}" in template

    def test_get_mmlu_single_shot(self):
        template = get_prompt_template("mmlu", PromptType.SINGLE_SHOT)
        assert "{choice_a}" in template
        assert "{choice_b}" in template
        assert "{choice_c}" in template
        assert "{choice_d}" in template

    def test_get_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_prompt_template("fake_dataset", PromptType.SINGLE_SHOT)

    def test_all_datasets_have_all_6_template_types(self):
        """Each dataset must have all 6 prompt types."""
        required = set(PromptType)
        for ds in get_available_datasets():
            available = set(get_available_prompt_types(ds))
            assert required == available, (
                f"Dataset {ds} missing: {required - available}"
            )

    def test_available_datasets(self):
        datasets = get_available_datasets()
        assert "boolq" in datasets
        assert "mmlu" in datasets
        assert "math" in datasets
        assert "gsm8k" in datasets
        assert len(datasets) == 7  # boolq, mmlu, bbh, sciq, arc, math, gsm8k


class TestPromptFormatting:
    """Test prompt formatting with sample data."""

    def test_single_shot_boolq_renders(self):
        sample = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue during the day.",
        }
        result = format_prompt("boolq", PromptType.SINGLE_SHOT, sample)
        assert "Is the sky blue?" in result
        assert "The sky appears blue" in result
        assert "{question}" not in result

    def test_guided_includes_target_answer(self):
        sample = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue.",
        }
        result = format_prompt(
            "boolq",
            PromptType.GUIDED_SINGLE_SHOT,
            sample,
            target_answer="Yes",
        )
        assert "Yes" in result
        assert "target" in result.lower() or "answer" in result.lower()

    def test_deliberation_lists_responses(self):
        sample = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue.",
        }
        responses = ["Yes, the sky is blue.", "No, it's gray."]
        result = format_prompt(
            "boolq",
            PromptType.DELIBERATION_ACTOR,
            sample,
            responses=responses,
        )
        assert "Person 1 said: Yes, the sky is blue." in result
        assert "Person 2 said: No, it's gray." in result

    def test_critic_prompt_includes_actor_response(self):
        sample = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue.",
        }
        result = format_prompt(
            "boolq",
            PromptType.DELIBERATION_CRITIC,
            sample,
            actor_response="I think the answer is Yes.",
        )
        assert "I think the answer is Yes." in result

    def test_mmlu_with_choices(self):
        sample = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
        }
        result = format_prompt("mmlu", PromptType.SINGLE_SHOT, sample)
        assert "(A) 3" in result
        assert "(B) 4" in result
        assert "(C) 5" in result
        assert "(D) 6" in result


class TestFormatResponses:
    """Test multi-response formatting helper."""

    def test_list_input(self):
        result = _format_responses(["First response", "Second response"])
        assert "Person 1 said: First response" in result
        assert "Person 2 said: Second response" in result

    def test_dict_input(self):
        result = _format_responses({1: "First", 2: "Second"})
        assert "Person 1 said: First" in result

    def test_single_response(self):
        result = _format_responses(["Only one"])
        assert "Person 1 said: Only one" in result

    def test_empty_response(self):
        result = _format_responses([])
        assert result == ""
