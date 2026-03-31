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
        from src.deliberation.engine import deliberate

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
        from src.deliberation.engine import deliberate

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
        from src.deliberation.engine import deliberate

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


class TestGuidedDeliberate:
    """Test guided deliberation round."""

    def test_guided_actor_includes_target(self):
        from src.deliberation.engine import guided_deliberate_round

        model = MockInference(responses=["I believe the answer is Yes."])
        sample = {
            "question": "Is X true?",
            "passage": "Evidence.",
            "answer": "yes",
            "task_type": "yes_no",
        }
        result = guided_deliberate_round(
            model, sample, "boolq",
            target_answer="Yes",
            previous_responses=[],
            agent="actor",
        )
        assert "Yes" in result

    def test_guided_critic_uses_actor_response(self):
        from src.deliberation.engine import guided_deliberate_round

        model = MockInference(responses=["Here's why the answer is No."])
        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "no",
            "task_type": "yes_no",
        }
        result = guided_deliberate_round(
            model, sample, "boolq",
            target_answer="No",
            previous_responses=["Previous answer."],
            agent="critic",
            previous_actor_response="I think Yes.",
        )
        assert isinstance(result, str)
        assert len(result) > 0
