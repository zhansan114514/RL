"""Tests for data loading, preprocessing, and answer extraction."""

import pytest
from unittest.mock import patch, MagicMock

from src.data.preprocessor import extract_answer, normalize_answer
from src.data.loader import load_dataset


class TestAnswerExtraction:
    """Test zeta function for extracting answers from LLM responses."""

    # --- BoolQ: Yes/No extraction ---
    @pytest.mark.parametrize(
        "response, expected",
        [
            ("The answer is Yes.", "yes"),
            ("Final answer: No", "no"),
            ("Based on the passage, the answer is Yes", "yes"),
            ("I think the answer is no.", "no"),
            ("Yes, that is correct.", "yes"),
            ("No, that is incorrect.", "no"),
            ("YES", "yes"),
            ("NO", "no"),
            # "The sky is blue" has no Yes/No → None

        ],
    )
    def test_extract_yes_no(self, response, expected):
        result = extract_answer(response, task_type="yes_no")
        assert result == expected

    def test_extract_yes_no_empty_response(self):
        result = extract_answer("", task_type="yes_no")
        assert result is None or result == ""

    # --- Multiple Choice: A/B/C/D extraction ---
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
            ("The answer is (A) rated the group.", "A"),
        ],
    )
    def test_extract_multiple_choice(self, response, expected):
        result = extract_answer(response, task_type="multiple_choice")
        assert result == expected

    def test_extract_multiple_choice_empty(self):
        result = extract_answer("", task_type="multiple_choice")
        assert result is None or result == ""

    def test_extract_multiple_choice_no_option(self):
        """Response without a clear option should return None or best guess."""
        result = extract_answer(
            "I don't know the answer.", task_type="multiple_choice"
        )
        # Should not crash, may return None
        assert result is None or isinstance(result, str)


class TestNormalizeAnswer:
    """Test answer normalization for consistent comparison."""

    def test_normalize_yes(self):
        # normalize_answer takes first char and upper: "Yes" -> "Y"
        assert normalize_answer("Yes") == "Y"

    def test_normalize_no(self):
        assert normalize_answer("No") == "N"

    def test_normalize_option(self):
        assert normalize_answer("(A)") == "A"
        assert normalize_answer("A.") == "A"


class TestDatasetLoader:
    """Test dataset loading functions (mocked for unit tests)."""

    @patch("src.data.loader.hf_load_dataset")
    def test_load_boolq_shape(self, mock_load):
        """BoolQ should return train/val/test splits."""
        # Mock a DatasetDict-like object
        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(return_value=iter([]))
        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))
        mock_ds = {"train": mock_train, "validation": mock_val}
        mock_load.return_value = mock_ds
        result = load_dataset("boolq")
        assert "train" in result

    @patch("src.data.loader.hf_load_dataset")
    def test_load_mmlu_shape(self, mock_load):
        """MMLU should load with correct splits."""
        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter([]))
        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))
        mock_ds = {"test": mock_test, "validation": mock_val}
        mock_load.return_value = mock_ds
        result = load_dataset("mmlu")
        assert isinstance(result, dict)

    def test_load_unknown_dataset_raises(self):
        """Loading an unknown dataset should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")


class TestDataFormat:
    """Test that loaded data has consistent format."""

    def test_standardized_format_keys(self):
        """Each sample should have question, passage, answer, choices keys."""
        from src.data.preprocessor import standardize_sample

        # BoolQ sample
        sample = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue...",
            "answer": True,
        }
        result = standardize_sample(sample, task_type="yes_no")
        assert "question" in result
        assert "answer" in result

        # MMLU sample
        sample = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,  # index
        }
        result = standardize_sample(sample, task_type="multiple_choice")
        assert "question" in result
        assert "choices" in result
        assert "answer" in result
