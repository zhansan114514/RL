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
            ("The answer is Yes.", "YES"),
            ("Final answer: No", "NO"),
            ("Based on the passage, the answer is Yes", "YES"),
            ("I think the answer is no.", "NO"),
            ("Yes, that is correct.", "YES"),
            ("No, that is incorrect.", "NO"),
            ("YES", "YES"),
            ("NO", "NO"),
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


class TestDataPreprocessorEdgeCases:
    """Test edge cases and malformed inputs for data preprocessing."""

    def test_standardize_empty_sample(self):
        """Empty sample should return default structure."""
        from src.data.preprocessor import standardize_sample

        result = standardize_sample({}, task_type="yes_no")
        assert result["question"] == ""
        assert result["passage"] == ""
        assert result["answer"] == "no"  # Default for False/None

    def test_standardize_missing_fields(self):
        """Missing fields should use empty defaults."""
        from src.data.preprocessor import standardize_sample

        sample = {"answer": True}  # Missing question and passage
        result = standardize_sample(sample, task_type="yes_no")
        assert result["question"] == ""
        assert result["passage"] == ""
        assert result["answer"] == "yes"

    def test_standardize_boolq_varied_answer_formats(self):
        """Should handle different BoolQ answer formats."""
        from src.data.preprocessor import standardize_sample

        test_cases = [
            (True, "yes"),
            (False, "no"),
            ("true", "yes"),
            ("True", "yes"),
            ("false", "no"),
            ("False", "no"),
            (1, "no"),
            (0, "no"),
        ]
        for input_answer, expected in test_cases:
            sample = {"question": "Test?", "answer": input_answer}
            result = standardize_sample(sample, task_type="yes_no")
            assert result["answer"] == expected

    def test_standardize_mc_empty_choices(self):
        """Should handle empty choices list."""
        from src.data.preprocessor import standardize_sample

        sample = {
            "question": "Test?",
            "choices": [],
            "answer": "A",
        }
        result = standardize_sample(sample, task_type="multiple_choice")
        assert result["choices"] == []

    def test_standardize_mc_answer_as_index_out_of_bounds(self):
        """Answer index beyond choices length should return empty."""
        from src.data.preprocessor import standardize_sample

        sample = {
            "question": "Test?",
            "choices": ["A", "B"],
            "answer": 5,  # Out of bounds
        }
        result = standardize_sample(sample, task_type="multiple_choice")
        assert result["answer"] == ""

    def test_standardize_mc_answer_with_parentheses(self):
        """Should strip parentheses from answer string."""
        from src.data.preprocessor import standardize_sample

        sample = {
            "question": "Test?",
            "choices": ["A", "B", "C", "D"],
            "answer": "(B)",
        }
        result = standardize_sample(sample, task_type="multiple_choice")
        assert result["answer"] == "B"

    def test_standardize_mc_answer_lowercase(self):
        """Should uppercase lowercase answer letters."""
        from src.data.preprocessor import standardize_sample

        sample = {
            "question": "Test?",
            "choices": ["A", "B", "C", "D"],
            "answer": "c",
        }
        result = standardize_sample(sample, task_type="multiple_choice")
        assert result["answer"] == "C"

    def test_normalize_answer_with_none(self):
        """normalize_answer should handle None input."""
        from src.data.preprocessor import normalize_answer

        result = normalize_answer(None)
        assert result == ""

    def test_normalize_answer_with_empty_string(self):
        """Empty string should return empty."""
        from src.data.preprocessor import normalize_answer

        result = normalize_answer("")
        assert result == ""

    def test_normalize_answer_with_whitespace(self):
        """Should strip whitespace."""
        from src.data.preprocessor import normalize_answer

        result = normalize_answer("  YES  ")
        # Takes first char and upper: "YES" -> "Y"
        assert result == "Y"

    def test_extract_answer_with_whitespace_only(self):
        """Whitespace-only response should return None."""
        from src.data.preprocessor import extract_answer

        result = extract_answer("   \n\t  ", task_type="yes_no")
        assert result is None

    def test_extract_answer_with_very_long_text(self):
        """Should handle very long responses."""
        from src.data.preprocessor import extract_answer

        long_text = " ".join(["word"] * 10000) + " The answer is Yes."
        result = extract_answer(long_text, task_type="yes_no")
        assert result == "YES"

    def test_extract_answer_mixed_type_fallback(self):
        """Mixed type should fallback to yes_no when MC fails."""
        from src.data.preprocessor import extract_answer

        result = extract_answer("The answer is Yes.", task_type="mixed")
        assert result == "YES"

    def test_extract_answer_invalid_task_type(self):
        """Invalid task type should return None."""
        from src.data.preprocessor import extract_answer

        result = extract_answer("The answer is Yes.", task_type="invalid")
        assert result is None


class TestGenerateWrongAnswer:
    """Test wrong answer generation for guided trajectories."""

    def test_flip_yes_no(self):
        """Should flip yes to no and vice versa."""
        from src.data.preprocessor import generate_wrong_answer

        assert generate_wrong_answer("yes") == "no"
        assert generate_wrong_answer("no") == "yes"
        assert generate_wrong_answer("YES") == "no"
        assert generate_wrong_answer("No") == "yes"

    def test_pick_wrong_choice(self):
        """Should pick a different choice from the list."""
        from src.data.preprocessor import generate_wrong_answer

        # Mock random to return first wrong option
        import random
        with patch.object(random, "choice", side_effect=lambda x: x[0]):
            result = generate_wrong_answer("C", ["A", "B", "C", "D"])
            assert result != "C"
            assert result in ["A", "B", "D"]

    def test_empty_choices_uses_defaults(self):
        """Should use default A/B/C/D when choices is empty."""
        from src.data.preprocessor import generate_wrong_answer

        result = generate_wrong_answer("A", [])
        assert result in ["B", "C", "D"]

    def test_all_options_wrong(self):
        """When all options are 'wrong' (same as correct), should still return something."""
        from src.data.preprocessor import generate_wrong_answer

        # Edge case: choices list only contains the correct answer
        result = generate_wrong_answer("A", ["A"])
        # Should fallback to default behavior
        assert result in ["B", "C", "D"]

    def test_case_insensitive_comparison(self):
        """Should compare case-insensitively."""
        from src.data.preprocessor import generate_wrong_answer

        result = generate_wrong_answer("YES", ["yes", "no"])
        assert result == "no" or result == "NO"


class TestBBHDatasetLoading:
    """Test BBH dataset loading with custom splitting."""

    @patch("src.data.loader.hf_load_dataset")
    def test_bbh_custom_split_ratios(self, mock_hf_load):
        """BBH should accept custom split ratios."""
        from src.data.loader import load_dataset
        from datasets import DatasetDict, Dataset

        # Create proper Dataset structure for BBH
        task1_data = Dataset.from_dict({
            "question": [f"Task1 Q{i}" for i in range(50)],
            "answer": ["A"] * 50,
        })
        task2_data = Dataset.from_dict({
            "question": [f"Task2 Q{i}" for i in range(50)],
            "answer": ["B"] * 50,
        })

        mock_dataset_dict = DatasetDict({
            "task1": task1_data,
            "task2": task2_data,
        })
        mock_hf_load.return_value = mock_dataset_dict

        result = load_dataset(
            "bbh",
            split_ratios={"test": 0.3, "validation": 0.2},
            seed=42,
        )

        # Should have all three splits
        assert "train" in result
        assert "validation" in result
        assert "test" in result

    @patch("src.data.loader.hf_load_dataset")
    def test_bbh_default_split_ratios(self, mock_hf_load):
        """BBH should use default split ratios when none provided."""
        from src.data.loader import load_dataset
        from datasets import DatasetDict, Dataset

        task_data = Dataset.from_dict({
            "question": [f"Q{i}" for i in range(100)],
            "answer": ["A"] * 100,
        })

        mock_dataset_dict = DatasetDict({
            "task1": task_data,
        })
        mock_hf_load.return_value = mock_dataset_dict

        result = load_dataset("bbh")

        # Default ratios: test=0.25, validation=0.10
        # So train should be 0.65
        assert "train" in result
        assert "validation" in result
        assert "test" in result

    @patch("src.data.loader.hf_load_dataset")
    def test_bbh_stratified_by_task(self, mock_hf_load):
        """BBH samples should preserve task information."""
        from src.data.loader import load_dataset
        from datasets import DatasetDict, Dataset

        # Create mock data with different task names
        task1_data = Dataset.from_dict({
            "question": [f"task1 Q{i}" for i in range(50)],
            "answer": ["A"] * 50,
        })
        task2_data = Dataset.from_dict({
            "question": [f"task2 Q{i}" for i in range(50)],
            "answer": ["B"] * 50,
        })

        mock_dataset_dict = DatasetDict({
            "task1": task1_data,
            "task2": task2_data,
        })
        mock_hf_load.return_value = mock_dataset_dict

        result = load_dataset("bbh", seed=42)

        # All samples should be processed
        all_samples = (
            result.get("train", []) +
            result.get("validation", []) +
            result.get("test", [])
        )
        assert len(all_samples) == 100

    @patch("src.data.loader.hf_load_dataset")
    def test_bbh_empty_dataset(self, mock_hf_load):
        """BBH should handle empty dataset gracefully."""
        from src.data.loader import load_dataset
        from datasets import DatasetDict, Dataset

        empty_data = Dataset.from_dict({
            "question": [],
            "answer": [],
        })

        mock_dataset_dict = DatasetDict({
            "task1": empty_data,
        })
        mock_hf_load.return_value = mock_dataset_dict

        result = load_dataset("bbh")

        # Should return empty splits
        assert len(result.get("train", [])) == 0
        assert len(result.get("validation", [])) == 0
        assert len(result.get("test", [])) == 0

    @patch("src.data.loader.hf_load_dataset")
    def test_bbh_single_task(self, mock_hf_load):
        """BBH with single task should still work."""
        from src.data.loader import load_dataset
        from datasets import DatasetDict, Dataset

        task_data = Dataset.from_dict({
            "question": [f"Q{i}" for i in range(100)],
            "answer": ["A"] * 100,
        })

        mock_dataset_dict = DatasetDict({
            "only_task": task_data,
        })
        mock_hf_load.return_value = mock_dataset_dict

        result = load_dataset("bbh")

        total = len(result.get("train", [])) + len(result.get("validation", [])) + len(result.get("test", []))
        assert total == 100
