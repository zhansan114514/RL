"""Tests for deliberation engine."""

import pytest


class MockInference:
    """Mock inference for testing without GPU."""

    def __init__(self, responses: list[str] = None):
        self.responses = responses or ["The answer is Yes.", "Good point."]
        self._call_count = 0

    def generate_single(self, prompt, **kwargs):
        resp = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1
        return resp


class TestDeliberateEngine:
    """Test the natural deliberation loop."""

    def test_returns_correct_round_count(self):
        from src.algorithms.deliberation import deliberate

        actor = MockInference(responses=[
            "Final answer: Yes.", "Final answer: Yes.",
            "Final answer: No.", "Final answer: Yes.",
            "Final answer: Yes.",
        ])
        critic = MockInference(responses=[
            "Feedback 0.", "Feedback 1.",
            "Feedback 2.", "Feedback 3.", "Feedback 4.",
        ])
        sample = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue.",
            "answer": "yes",
            "task_type": "yes_no",
        }
        traj = deliberate(actor, critic, sample, "boolq", num_rounds=5)

        assert len(traj) == 5
        for i, t in enumerate(traj):
            assert t["round"] == i
            assert "actor_response" in t
            assert "critic_response" in t
            assert "actor_answer" in t

    def test_single_round(self):
        from src.algorithms.deliberation import deliberate

        actor = MockInference(responses=["Final answer: No."])
        critic = MockInference(responses=["Feedback."])
        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }
        traj = deliberate(actor, critic, sample, "boolq", num_rounds=1)
        assert len(traj) == 1
        assert traj[0]["actor_answer"] == "NO"

    def test_round0_uses_single_shot_prompt(self):
        from src.algorithms.deliberation import deliberate

        actor = MockInference(responses=["Final answer: Yes."])
        critic = MockInference(responses=["Feedback."])
        sample = {
            "question": "Is X true?",
            "passage": "Evidence for X.",
            "answer": "yes",
            "task_type": "yes_no",
        }
        traj = deliberate(actor, critic, sample, "boolq", num_rounds=1)
        # Round 0 prompt should be the single-shot prompt
        assert "yes-no question" in traj[0]["actor_prompt"]


class TestDeliberationEdgeCases:
    """Test edge cases and malformed inputs for deliberation engine."""

    def test_deliberate_with_zero_rounds(self):
        """Should handle num_rounds=0 gracefully."""
        from src.algorithms.deliberation import deliberate

        actor = MockInference(responses=["Yes."])
        critic = MockInference(responses=["Feedback."])
        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }
        traj = deliberate(actor, critic, sample, "boolq", num_rounds=0)
        assert len(traj) == 0

    def test_deliberate_with_empty_actor_response(self):
        """Should handle empty actor response."""
        from src.algorithms.deliberation import deliberate

        actor = MockInference(responses=[""])
        critic = MockInference(responses=["Feedback."])
        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }
        traj = deliberate(actor, critic, sample, "boolq", num_rounds=1)
        assert len(traj) == 1
        assert traj[0]["actor_response"] == ""
        # Empty response should result in None answer
        assert traj[0]["actor_answer"] is None

    def test_deliberate_with_whitespace_only_response(self):
        """Should handle whitespace-only response."""
        from src.algorithms.deliberation import deliberate

        actor = MockInference(responses=["   \n\t  "])
        critic = MockInference(responses=["Feedback."])
        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }
        traj = deliberate(actor, critic, sample, "boolq", num_rounds=1)
        assert len(traj) == 1
        assert traj[0]["actor_answer"] is None

    def test_deliberate_with_very_long_response(self):
        """Should handle very long responses."""
        from src.algorithms.deliberation import deliberate

        long_response = " ".join(["word"] * 10000) + " Yes."
        actor = MockInference(responses=[long_response])
        critic = MockInference(responses=["Feedback."])
        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }
        traj = deliberate(actor, critic, sample, "boolq", num_rounds=1)
        assert len(traj) == 1
        assert len(traj[0]["actor_response"]) > 10000

    def test_deliberate_with_missing_task_type(self):
        """Should handle missing task_type in sample."""
        from src.algorithms.deliberation import deliberate

        actor = MockInference(responses=["Yes."])
        critic = MockInference(responses=["Feedback."])
        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            # Missing task_type
        }
        traj = deliberate(actor, critic, sample, "boolq", num_rounds=1)
        assert len(traj) == 1
        # Should default to yes_no task type
        assert traj[0]["actor_answer"] in ("YES", "NO", None)
