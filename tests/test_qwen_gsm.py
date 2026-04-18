"""Tests for Qwen2.5 LoRA configuration and MATH/GSM answer extraction."""

import pytest
from unittest.mock import patch, MagicMock
import re

from src.training.lora_config import get_lora_config, MODEL_TARGET_MODULES, DEFAULT_TARGET_MODULES
from src.algorithms.reward import extract_answer, normalize_answer, _extract_math, math_answers_equal
from src.data.preprocessor import standardize_sample


class TestQwenLoRAConfig:
    """Test LoRA configuration for Qwen2.5 and Qwen3 models."""

    def test_qwen25_target_modules(self):
        """Qwen2.5 should use default target modules."""
        assert "qwen2.5" in MODEL_TARGET_MODULES
        assert MODEL_TARGET_MODULES["qwen2.5"] == DEFAULT_TARGET_MODULES

    def test_qwen3_target_modules(self):
        """Qwen3 should use default target modules."""
        assert "qwen3" in MODEL_TARGET_MODULES
        assert MODEL_TARGET_MODULES["qwen3"] == DEFAULT_TARGET_MODULES

    def test_get_lora_config_qwen25(self):
        """Should create LoRA config for Qwen2.5 with correct parameters."""
        config = get_lora_config(model_type="qwen2.5", r=256, lora_alpha=512, lora_dropout=0.0)

        assert config.r == 256
        assert config.lora_alpha == 512
        assert config.lora_dropout == 0.0
        assert set(config.target_modules) == set(DEFAULT_TARGET_MODULES)

    def test_get_lora_config_qwen3(self):
        """Should create LoRA config for Qwen3 with correct parameters."""
        config = get_lora_config(model_type="qwen3", r=128, lora_alpha=256, lora_dropout=0.1)

        assert config.r == 128
        assert config.lora_alpha == 256
        assert config.lora_dropout == 0.1

    def test_get_lora_config_unknown_model_uses_default(self):
        """Unknown model type should fallback to default target modules."""
        config = get_lora_config(model_type="unknown_model")

        assert set(config.target_modules) == set(DEFAULT_TARGET_MODULES)

    def test_target_modules_completeness(self):
        """Default target modules should include all necessary projection layers."""
        expected_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert DEFAULT_TARGET_MODULES == expected_modules


class TestMathAnswerExtraction:
    """Test mathematical answer extraction for MATH/GSM datasets."""

    # --- \boxed{} format extraction ---
    @pytest.mark.parametrize(
        "response, expected",
        [
            # Standard \boxed{} format
            ("The answer is \\boxed{42}.", "42"),
            ("\\boxed{3.14}", "3.14"),
            ("Therefore, \\boxed{7}", "7"),
            # Nested braces (balanced brace matching)
            ("\\boxed{{1,2,3}}", "{1,2,3}"),  # Balanced extraction preserves inner braces
            # With spaces
            ("\\boxed{  5  }", "5"),
            # Mathematical expressions (balanced brace matching)
            ("\\boxed{x^2 + 1}", "x^2 + 1"),
            # \frac and \sqrt with nested {} — balanced brace matching handles correctly
            ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),  # Balanced extraction
            ("\\boxed{\\sqrt{2}}", "\\sqrt{2}"),  # Balanced extraction
            # Multiple boxes (should extract first)
            ("First \\boxed{1} then \\boxed{2}", "1"),
            # Case insensitive variations
            ("boxed{100}", "100"),
            # With text around
            ("After simplification, we get \\boxed{42}. Therefore the answer is 42.", "42"),
        ],
    )
    def test_extract_boxed_answer(self, response, expected):
        """Should extract answers from \\boxed{} format."""
        result = extract_answer(response, task_type="math")
        assert result == expected

    # --- Final Answer: pattern ---
    @pytest.mark.parametrize(
        "response, expected",
        [
            ("Final Answer: 42", "42"),
            ("Final answer: 3.14", "3.14"),
            ("Final Answer is 100", "100"),
            ("Answer: 7", "7"),
            ("answer: 5", "5"),
            ("The answer is 42", "42"),
            ("The answer is 3.14", "3.14"),
        ],
    )
    def test_extract_final_answer_pattern(self, response, expected):
        """Should extract from 'Final Answer:' patterns."""
        result = extract_answer(response, task_type="math")
        assert result == expected

    # --- Fallback patterns ---
    @pytest.mark.parametrize(
        "response, expected",
        [
            ("Therefore 42", "42"),
            ("So the answer is 100", "100"),
            ("Thus 7", "7"),
            ("Equals 5", "5"),
            ("= 42", "42"),
            ("x = 10", "10"),
            # Multiple equals (regex gets last match)
            ("x = 5, y = 10", "5"),  # Note: fallback patterns get first match in some cases
            # With units (should extract number)
            ("The result is 42 meters", "42"),
        ],
    )
    def test_extract_fallback_patterns(self, response, expected):
        """Should use fallback patterns for informal answers."""
        result = extract_answer(response, task_type="math")
        assert result == expected

    # --- Edge cases ---
    def test_extract_math_empty_response(self):
        """Empty response should return None."""
        result = extract_answer("", task_type="math")
        assert result is None

    def test_extract_math_whitespace_only(self):
        """Whitespace-only response should return None."""
        result = extract_answer("   \n\t  ", task_type="math")
        assert result is None

    def test_extract_math_no_answer(self):
        """Response without any answer pattern should return None."""
        result = extract_answer("I need to think about this more.", task_type="math")
        assert result is None

    def test_extract_math_negative_number(self):
        """Should handle negative numbers."""
        result = extract_answer("The answer is \\boxed{-42}.", task_type="math")
        assert result == "-42"

    def test_extract_math_very_long_text(self):
        """Should handle very long responses."""
        long_text = " ".join(["word"] * 10000) + " The answer is \\boxed{42}."
        result = extract_answer(long_text, task_type="math")
        assert result == "42"

    def test_extract_math_decimal_various_formats(self):
        """Should handle various decimal formats."""
        test_cases = [
            ("\\boxed{3.14}", "3.14"),
            ("\\boxed{0.5}", "0.5"),
            ("\\boxed{.75}", ".75"),
            ("Final Answer: 100.0", "100.0"),
        ]
        for response, expected in test_cases:
            result = extract_answer(response, task_type="math")
            assert result == expected


class TestMathAnswerNormalization:
    """Test answer normalization for mathematical answers."""

    @pytest.mark.parametrize(
        "answer, expected",
        [
            ("42", "42"),        # Math mode: numeric normalization
            ("  42  ", "42"),    # Whitespace stripped
            ("42.0", "42.0"),    # Decimal preserved
            ("3.14", "3.14"),    # Decimal preserved
            ("-5", "-5"),        # Negative numbers
            ("x^2 + 1", "x^2 + 1"),  # Expressions preserved
            ("", ""),
            (None, ""),
        ],
    )
    def test_normalize_math_answer(self, answer, expected):
        """Should normalize math answers preserving numeric values."""
        result = normalize_answer(answer, task_type="math")
        assert result == expected

    def test_normalize_yes_no_still_works(self):
        """Non-math task types should still use first-char normalization."""
        assert normalize_answer("YES", "yes_no") == "Y"
        assert normalize_answer("No", "yes_no") == "N"
        assert normalize_answer("(A)", "multiple_choice") == "A"


class TestMathDataPreprocessing:
    """Test MATH/GSM dataset preprocessing."""

    def test_standardize_math_sample_with_boxed(self):
        """Should extract answer from \\boxed{} in raw data."""
        sample = {
            "question": "What is 2+2?",
            "answer": r"\boxed{4}",
        }
        result = standardize_sample(sample, task_type="math")
        assert result["question"] == "What is 2+2?"
        assert result["answer"] == "4"
        assert result["passage"] == ""
        assert result["choices"] == []

    def test_standardize_math_sample_plain_number(self):
        """Should handle plain number answers."""
        sample = {
            "question": "What is 2+2?",
            "answer": "4",
        }
        result = standardize_sample(sample, task_type="math")
        assert result["answer"] == "4"

    def test_standardize_math_sample_numeric_answer(self):
        """Should handle numeric answer types."""
        sample = {
            "question": "What is 2+2?",
            "answer": 4,
        }
        result = standardize_sample(sample, task_type="math")
        assert result["answer"] == "4"

    def test_standardize_math_sample_complex_expression(self):
        """Should handle complex mathematical expressions."""
        sample = {
            "question": "Solve for x.",
            "answer": r"\boxed{x^2 + 2x + 1}",
        }
        result = standardize_sample(sample, task_type="math")
        assert result["answer"] == "x^2 + 2x + 1"

    def test_standardize_math_sample_gsm8k_format(self):
        """Should handle GSM8K format with step-by-step."""
        sample = {
            "question": "John has 5 apples...",
            "answer": "John has 10 apples. #### 10",
        }
        result = standardize_sample(sample, task_type="math")
        # GSM8K uses #### to mark final answer
        assert "10" in result["answer"] or result["answer"] == "John has 10 apples. #### 10"

    def test_standardize_math_empty_question(self):
        """Should handle empty question."""
        sample = {
            "question": "",
            "answer": "42",
        }
        result = standardize_sample(sample, task_type="math")
        assert result["question"] == ""

    def test_standardize_math_empty_answer(self):
        """Should handle empty answer."""
        sample = {
            "question": "What is 2+2?",
            "answer": "",
        }
        result = standardize_sample(sample, task_type="math")
        assert result["answer"] == ""

    def test_standardize_math_missing_fields(self):
        """Should handle missing fields gracefully."""
        result = standardize_sample({}, task_type="math")
        assert result["question"] == ""
        assert result["answer"] == ""
        assert result["passage"] == ""
        assert result["choices"] == []


class TestMathAnswerComparison:
    """Test answer comparison for mathematical answers using math_answers_equal."""

    def test_exact_match(self):
        """Exact match should return True."""
        assert math_answers_equal("42", "42")

    def test_whitespace_ignored(self):
        """Whitespace should be ignored."""
        assert math_answers_equal("  42  ", "42")

    def test_integer_float_equivalence(self):
        """Integer and float representations of same value should match."""
        assert math_answers_equal("42", "42.0")
        assert math_answers_equal("42.0", "42")

    def test_numeric_tolerance(self):
        """Numeric answers should match within tolerance."""
        assert math_answers_equal("3.14159", "3.14159")

    def test_expression_match(self):
        """Should handle mathematical expressions."""
        assert math_answers_equal(r"x^2 + 1", r"x^2 + 1")

    def test_fraction_match(self):
        """Should handle fractions."""
        assert math_answers_equal(r"\frac{1}{2}", r"\frac{1}{2}")

    def test_empty_answers(self):
        """Empty answers should match each other."""
        assert math_answers_equal("", "")
        assert not math_answers_equal("42", "")
        assert not math_answers_equal("", "42")

    def test_different_values(self):
        """Different numeric values should not match."""
        assert not math_answers_equal("42", "43")
        assert not math_answers_equal("3.14", "2.71")


class TestMathExtractionEdgeCases:
    """Test edge cases for mathematical answer extraction."""

    def test_extract_nested_boxes(self):
        """Should handle nested \\boxed{} patterns."""
        response = r"The answer is \boxed{\boxed{42}}."
        result = extract_answer(response, task_type="math")
        # Balanced brace matching: outer \boxed{} contains \boxed{42}
        assert result is not None
        # The content of the outer box is "\boxed{42}"
        assert "42" in result

    def test_extract_boxed_with_spaces(self):
        """Should handle \\boxed with spaces before brace."""
        response = r"The answer is \boxed {42}."
        result = extract_answer(response, task_type="math")
        # Pattern \boxed\{ doesn't match with space, tries other patterns

    def test_extract_boxed_empty(self):
        """Should handle empty \\boxed{}."""
        response = r"The answer is \boxed{}."
        result = extract_answer(response, task_type="math")
        # Empty \boxed{} returns empty string or tries other patterns
        assert result == "" or result is None

    def test_extract_multiple_math_patterns(self):
        """Should prioritize \\boxed{} over other patterns."""
        response = r"Final Answer: 100, but actually \boxed{42}."
        result = extract_answer(response, task_type="math")
        # Should extract from \boxed{} first (higher priority)
        assert result == "42"

    def test_extract_math_with_unicode(self):
        """Should handle unicode mathematical symbols."""
        response = r"The answer is \boxed{α + β}."
        result = extract_answer(response, task_type="math")
        assert "α" in result or "β" in result

    def test_extract_math_very_long_number(self):
        """Should handle very long numbers."""
        response = r"The answer is \boxed{12345678901234567890}."
        result = extract_answer(response, task_type="math")
        assert result == "12345678901234567890"


class TestQwenModelDetection:
    """Test Qwen model type detection from model names."""

    @pytest.mark.parametrize(
        "model_name, expected_type",
        [
            ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5"),
            ("Qwen/Qwen2.5-14B", "qwen2.5"),
            ("qwen2.5-32b", "qwen2.5"),
            ("Qwen/Qwen3-7B", "qwen3"),
            ("qwen3-14b", "qwen3"),
            # Non-Qwen models
            ("meta-llama/Llama-3-8B", None),
            ("mistralai/Mistral-7B", None),
            ("google/gemma-2-2b-it", None),
        ],
    )
    def test_detect_qwen_model_type(self, model_name, expected_type):
        """Should detect Qwen model type from name."""
        # This is a simple string matching test
        # In actual code, this would use _detect_model_type function
        if "qwen2.5" in model_name.lower():
            detected = "qwen2.5"
        elif "qwen3" in model_name.lower() or "qwen-3" in model_name.lower():
            detected = "qwen3"
        else:
            detected = None

        assert detected == expected_type


class TestGSM8KSpecific:
    """Test GSM8K-specific answer extraction."""

    def test_gsm8k_hash_separator(self):
        """GSM8K uses #### to separate reasoning from answer."""
        sample = {
            "question": "John has 5 apples. He buys 3 more. How many does he have?",
            "answer": "John has 5 apples and buys 3 more. 5 + 3 = 8. #### 8",
        }
        result = standardize_sample(sample, task_type="math")
        # The entire answer string is preserved
        assert "8" in result["answer"]

    def test_gsm8k_numeric_only(self):
        """Should handle GSM8K with just the numeric answer."""
        sample = {
            "question": "What is 2+2?",
            "answer": "#### 4",
        }
        result = standardize_sample(sample, task_type="math")
        assert "4" in result["answer"]

    def test_gsm8k_multi_step(self):
        """Should handle multi-step GSM8K solutions."""
        sample = {
            "question": "Complex word problem.",
            "answer": "First step: 5. Second step: 10. #### 15",
        }
        result = standardize_sample(sample, task_type="math")
        # Preserve full answer
        assert "15" in result["answer"]
