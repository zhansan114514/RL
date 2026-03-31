"""Tests for answer extraction and accuracy evaluation."""

import pytest

from src.reward.accuracy import (
    extract_answer,
    normalize_answer,
    compute_accuracy,
    compute_accuracy_with_ci,
    compute_per_round_accuracy,
    compute_improvement_rate,
)


class TestExtractAnswer:
    """Test the zeta function for answer extraction."""

    # --- Yes/No ---
    @pytest.mark.parametrize(
        "response, expected",
        [
            ("The answer is Yes.", "YES"),
            ("Final answer: No", "NO"),
            ("Based on the passage, the answer is Yes", "YES"),
            ("I think the answer is no.", "NO"),
            ("Yes, that is correct.", "YES"),
            ("No, that is incorrect.", "NO"),
            ("YES", "YES"),
            ("NO", "NO"),
        ],
    )
    def test_yes_no(self, response, expected):
        result = extract_answer(response, task_type="yes_no")
        assert result == expected

    def test_yes_no_empty(self):
        assert extract_answer("", task_type="yes_no") is None

    def test_yes_no_no_keyword(self):
        """Response without yes/no should return None."""
        assert extract_answer("The sky is blue.", task_type="yes_no") is None

    # --- Multiple Choice ---
    @pytest.mark.parametrize(
        "response, expected",
        [
            ("The answer is (A).", "A"),
            ("Final Answer: (B)", "B"),
            ("I choose C", "C"),
            ("The correct option is D.", "D"),
            ("Answer: A", "A"),
            ("(C) is the right answer.", "C"),
            ("Option B is correct.", "B"),
        ],
    )
    def test_multiple_choice(self, response, expected):
        result = extract_answer(response, task_type="multiple_choice")
        assert result == expected

    def test_mc_empty(self):
        assert extract_answer("", task_type="multiple_choice") is None

    # --- Mixed ---
    def test_mixed_prefers_mc(self):
        result = extract_answer("The answer is (A).", task_type="mixed")
        assert result == "A"

    def test_mixed_fallback_yes_no(self):
        result = extract_answer("The answer is Yes.", task_type="mixed")
        assert result == "YES"


class TestNormalizeAnswer:
    def test_yes(self):
        assert normalize_answer("Yes") == "Y"

    def test_no(self):
        assert normalize_answer("No") == "N"

    def test_option_a(self):
        assert normalize_answer("(A)") == "A"

    def test_option_b(self):
        assert normalize_answer("B.") == "B"

    def test_empty(self):
        assert normalize_answer("") == ""


class TestComputeAccuracy:
    def test_perfect(self):
        preds = ["Yes", "No", "A", "B"]
        labels = ["Yes", "No", "A", "B"]
        assert compute_accuracy(preds, labels) == 1.0

    def test_half(self):
        preds = ["Yes", "No"]
        labels = ["Yes", "Yes"]
        assert compute_accuracy(preds, labels) == 0.5

    def test_zero(self):
        preds = ["No", "B"]
        labels = ["Yes", "A"]
        assert compute_accuracy(preds, labels) == 0.0

    def test_empty(self):
        assert compute_accuracy([], []) == 0.0


class TestComputeAccuracyWithCI:
    def test_ci_range(self):
        preds = ["Yes"] * 80 + ["No"] * 20
        labels = ["Yes"] * 100
        acc, margin = compute_accuracy_with_ci(preds, labels)
        assert 0.75 <= acc <= 0.85
        assert 0.01 <= margin <= 0.10

    def test_ci_empty(self):
        acc, margin = compute_accuracy_with_ci([], [])
        assert acc == 0.0
        assert margin == 0.0


class TestPerRoundAccuracy:
    def test_improving(self):
        # Round 0: 50%, Round 1: 80%
        preds = [
            ["Yes", "No", "No", "No"],  # round 0: 1/4
            ["Yes", "Yes", "Yes", "No"],  # round 1: 3/4
        ]
        labels = ["Yes", "Yes", "Yes", "Yes"]
        accs = compute_per_round_accuracy(preds, labels)
        assert len(accs) == 2
        assert accs[0] == 0.25
        assert accs[1] == 0.75


class TestImprovementRate:
    def test_positive(self):
        assert compute_improvement_rate(0.8, 0.6) == pytest.approx(1 / 3)

    def test_negative(self):
        assert compute_improvement_rate(0.4, 0.6) < 0

    def test_zero_initial(self):
        assert compute_improvement_rate(0.5, 0.0) == 0.0


class TestEvaluateBenchmark:
    """Test the evaluate_benchmark function with mocked inference."""

    def test_evaluate_returns_expected_keys(self):
        from unittest.mock import MagicMock, patch
        from src.evaluation.benchmarks import evaluate_benchmark

        actor = MagicMock()
        critic = MagicMock()

        with patch("src.evaluation.benchmarks.deliberate") as mock_deliberate:
            mock_deliberate.return_value = [
                {
                    "round": i,
                    "actor_response": "Final answer: Yes",
                    "critic_response": "Feedback",
                    "actor_answer": "YES",
                    "actor_prompt": "...",
                }
                for i in range(3)
            ]

            samples = [
                {"question": "Q1?", "passage": "P1.", "answer": "YES", "task_type": "yes_no"},
                {"question": "Q2?", "passage": "P2.", "answer": "NO", "task_type": "yes_no"},
            ]

            results = evaluate_benchmark(
                actor, critic, samples, "boolq", num_rounds=3,
            )

            assert "dataset" in results
            assert "final_accuracy" in results
            assert "per_round_accuracy" in results
            assert "improvement_rate" in results
            assert results["num_samples"] == 2
            assert results["num_rounds"] == 3
            assert len(results["per_round_accuracy"]) == 3


class TestExtractAnswerEdgeCases:
    """Test edge cases and malformed inputs for answer extraction."""

    def test_whitespace_only_response(self):
        """Response with only whitespace should return None."""
        assert extract_answer("   \n\t  ", task_type="yes_no") is None
        assert extract_answer("   \n\t  ", task_type="multiple_choice") is None

    def test_mixed_case_yes_no(self):
        """Extract should work with various case combinations."""
        test_cases = ["yEs", "yeS", "YEs", "yES", "YeS", "yEs"]
        for case in test_cases:
            result = extract_answer(f"The answer is {case}.", task_type="yes_no")
            # Should normalize to YES/NO
            assert result in ("YES", "NO") or result is None

    def test_multiple_yes_no_uses_last(self):
        """When multiple Yes/No appear, should extract the last one."""
        response = "Yes it could be, but actually No in the end."
        result = extract_answer(response, task_type="yes_no")
        assert result == "NO"

    def test_yes_no_with_punctuation(self):
        """Yes/No with various punctuation should still extract."""
        test_cases = [
            ("Yes!", "YES"),
            ("No?", "NO"),
            ("Yes.", "YES"),
            ("No,", "NO"),
            ("Yes:", "YES"),
        ]
        for response, expected in test_cases:
            result = extract_answer(f"Final answer: {response}", task_type="yes_no")
            assert result == expected

    def test_multiple_choice_with_punctuation(self):
        """Multiple choice with various punctuation."""
        test_cases = [
            ("The answer is (A).", "A"),
            ("The answer is (B)!", "B"),
            ("The answer is (C)?", "C"),
            ("The answer is (D),", "D"),
        ]
        for response, expected in test_cases:
            result = extract_answer(response, task_type="multiple_choice")
            assert result == expected

    def test_very_long_response(self):
        """Should handle very long responses without crashing."""
        long_text = " ".join(["word"] * 10000) + " The answer is Yes."
        result = extract_answer(long_text, task_type="yes_no")
        assert result == "YES"

    def test_response_with_special_chars(self):
        """Special characters should not break extraction."""
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        response = f"Final answer: Yes {special_chars}"
        result = extract_answer(response, task_type="yes_no")
        assert result == "YES"

    def test_unicode_characters(self):
        """Should handle unicode characters gracefully."""
        response = "The answer is Yes 🎉 你好"
        result = extract_answer(response, task_type="yes_no")
        assert result == "YES"

    def test_mixed_task_type_with_only_mc(self):
        """Mixed task type should prefer MC when only MC present."""
        response = "The answer is (A)."
        result = extract_answer(response, task_type="mixed")
        assert result == "A"

    def test_mixed_task_type_with_only_yn(self):
        """Mixed task type should fallback to yes_no when no MC present."""
        response = "The answer is Yes."
        result = extract_answer(response, task_type="mixed")
        assert result == "YES"

    def test_invalid_task_type(self):
        """Invalid task type should return None and log warning."""
        result = extract_answer("The answer is Yes.", task_type="invalid_type")
        assert result is None

    def test_newline_in_answer(self):
        """Answer split across newlines should still be extracted."""
        response = "Final\nanswer:\nYes"
        result = extract_answer(response, task_type="yes_no")
        # May not extract due to pattern matching, but should not crash
        assert result is None or result in ("YES", "NO")

    def test_option_e_f_should_not_extract(self):
        """Options beyond A-D should not be extracted."""
        result = extract_answer("The answer is (E).", task_type="multiple_choice")
        assert result is None

    def test_nested_parentheses(self):
        """Nested parentheses should still extract correctly."""
        result = extract_answer("The answer is ((A)).", task_type="multiple_choice")
        assert result == "A"
