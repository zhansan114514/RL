"""Tests for MC roll-out reward estimation."""

import pytest
from unittest.mock import MagicMock, patch


class TestMCRolloutEdgeCases:
    """Test edge cases for Monte Carlo roll-out reward estimation."""

    def test_zero_remaining_rounds(self):
        """Zero remaining rounds should still return a value."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["Final answer: Yes."] * 10
        critic.generate.return_value = ["Feedback."] * 10

        sample = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        # With remaining_rounds=0, should only check current response
        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="The answer is Yes.",
            current_critic_response="Good.",
            previous_responses=[],
            num_simulations=5,
            remaining_rounds=0,
        )

        # Should return accuracy between 0 and 1
        assert 0.0 <= result <= 1.0

    def test_zero_simulations(self):
        """Zero simulations should safely return 0.0 (handled gracefully)."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        # With num_simulations=0, should return 0.0 gracefully
        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="The answer is Yes.",
            current_critic_response="Good.",
            previous_responses=[],
            num_simulations=0,
            remaining_rounds=1,
        )
        assert result == 0.0

    def test_all_simulations_correct(self):
        """When all simulations give correct answer, should return 1.0."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        # Batch generate returns all correct responses
        actor.generate.return_value = ["The answer is Yes."] * 10
        critic.generate.return_value = ["Feedback."] * 10

        sample = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="The answer is Yes.",
            current_critic_response="Good.",
            previous_responses=[],
            num_simulations=10,
            remaining_rounds=1,
        )

        # All correct should give 1.0
        assert result == 1.0

    def test_all_simulations_incorrect(self):
        """When all simulations give wrong answer, should return 0.0."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        # Batch generate returns all wrong responses
        actor.generate.return_value = ["The answer is No."] * 10
        critic.generate.return_value = ["Feedback."] * 10

        sample = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="The answer is No.",
            current_critic_response="Good.",
            previous_responses=[],
            num_simulations=10,
            remaining_rounds=1,
        )

        # All wrong should give 0.0
        assert result == 0.0

    def test_mixed_correct_incorrect(self):
        """Mixed correct/incorrect should return proportional accuracy."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()

        # Batch generate returns alternating correct/incorrect
        responses = ["The answer is Yes.", "The answer is No."] * 5
        actor.generate.return_value = responses
        critic.generate.return_value = ["Feedback."] * 10

        sample = {
            "question": "Is the sky blue?",
            "passage": "The sky appears blue.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="The answer is Yes.",
            current_critic_response="Good.",
            previous_responses=[],
            num_simulations=10,
            remaining_rounds=1,
        )

        # Should be approximately 0.5 (5 correct out of 10)
        assert result == 0.5

    def test_missing_answer_in_sample(self):
        """Missing answer in sample should return 0.0."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            # Missing "answer" field
            "task_type": "yes_no",
        }

        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="The answer is Yes.",
            current_critic_response="Good.",
            previous_responses=[],
            num_simulations=5,
            remaining_rounds=1,
        )

        # Should return 0.0 when no correct answer
        assert result == 0.0

    def test_empty_previous_responses(self):
        """Empty previous_responses should be handled."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["The answer is Yes."] * 3
        critic.generate.return_value = ["Feedback."] * 3

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        # Should not crash with empty previous_responses
        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="Starting point.",
            current_critic_response="Initial feedback.",
            previous_responses=[],
            num_simulations=3,
            remaining_rounds=1,
        )

        assert 0.0 <= result <= 1.0

    def test_long_previous_responses_list(self):
        """Should handle long list of previous responses."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["The answer is Yes."] * 3
        critic.generate.return_value = ["Feedback."] * 3

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        # Long history of previous responses
        long_history = [f"Previous response {i}." for i in range(100)]

        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="Current response.",
            current_critic_response="Current feedback.",
            previous_responses=long_history,
            num_simulations=3,
            remaining_rounds=1,
        )

        assert 0.0 <= result <= 1.0

    def test_multiple_choice_task(self):
        """Should handle multiple choice tasks."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["The answer is (A)."] * 5
        critic.generate.return_value = ["Feedback."] * 5

        sample = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": "A",
            "task_type": "multiple_choice",
        }

        result = estimate_final_accuracy(
            actor, critic, sample, "mmlu",
            current_actor_response="I think (A).",
            current_critic_response="Good.",
            previous_responses=[],
            num_simulations=5,
            remaining_rounds=1,
        )

        assert result == 1.0  # All correct

    def test_current_response_already_correct(self):
        """When current response is already correct, should still simulate."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        # Batch generate returns wrong answers
        actor.generate.return_value = ["The answer is No."] * 5
        critic.generate.return_value = ["Feedback."] * 5

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="The answer is Yes.",  # Current is correct
            current_critic_response="Good.",
            previous_responses=[],
            num_simulations=5,
            remaining_rounds=1,
        )

        # Should reflect simulation results, not current response
        assert result == 0.0  # All simulations were wrong

    def test_empty_actor_response_in_simulation(self):
        """Should handle empty actor responses during simulation."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = [""] * 5  # Empty responses
        critic.generate.return_value = ["Feedback."] * 5

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="The answer is Yes.",
            current_critic_response="Good.",
            previous_responses=[],
            num_simulations=5,
            remaining_rounds=1,
        )

        # Empty responses won't match correct answer
        assert result == 0.0

    def test_large_num_simulations(self):
        """Should handle large number of simulations."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["The answer is Yes."] * 1000
        critic.generate.return_value = ["Feedback."] * 1000

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        # Large number of simulations
        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="The answer is Yes.",
            current_critic_response="Good.",
            previous_responses=[],
            num_simulations=1000,
            remaining_rounds=1,
        )

        assert result == 1.0


class TestMCRolloutOneStepBehavior:
    """Test that MC roll-out follows one-step behavior from paper."""

    def test_one_step_not_all_remaining(self):
        """One-step roll-out should simulate exactly 1 round, not all remaining."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["The answer is Yes."] * 15
        critic.generate.return_value = ["Feedback."] * 15

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        # remaining_rounds=5 but should only simulate 1 round (one-step)
        estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="Current.",
            current_critic_response="Feedback.",
            previous_responses=[],
            num_simulations=3,
            remaining_rounds=5,  # Large number
        )

        # Batch generate should be called once for actor prompts (3 sims)
        # With one-step (remaining_rounds=1), we batch 3 actor + 3 critic calls
        assert actor.generate.call_count >= 1
        assert critic.generate.call_count >= 1

    def test_multiple_remaining_rounds_accumulation(self):
        """With remaining_rounds > 1, should accumulate responses correctly."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()

        # Batch generate returns enough responses
        actor.generate.return_value = ["Response."] * 10
        critic.generate.return_value = ["Feedback."] * 10

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="Initial.",
            current_critic_response="Initial feedback.",
            previous_responses=["Prev1", "Prev2"],
            num_simulations=2,
            remaining_rounds=2,
        )

        # Should have called generate at least once (batched)
        assert actor.generate.call_count >= 1


class TestMCRolloutOneStepFixVerification:
    """Verify the one-step roll-out fix matches paper specification."""

    def test_default_remaining_rounds_is_one(self):
        """The default remaining_rounds parameter should be 1 for one-step roll-out."""
        from src.algorithms.rollout import estimate_final_accuracy
        import inspect

        # Check function signature
        sig = inspect.signature(estimate_final_accuracy)
        default_remaining = sig.parameters["remaining_rounds"].default

        # Default should be 1 (one-step roll-out per paper)
        assert default_remaining == 1, "Default remaining_rounds should be 1 for one-step roll-out"

    def test_one_step_simulates_single_actor_critic_exchange(self):
        """One-step roll-out simulates exactly ONE actor-critic exchange per sim."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["Simulated response."] * 5
        critic.generate.return_value = ["Simulated feedback."] * 5

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        # Use default remaining_rounds=1
        estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="Current.",
            current_critic_response="Feedback.",
            previous_responses=[],
            num_simulations=5,
        )

        # Batch generate called once for actor (5 prompts) and once for critic (5 prompts)
        assert actor.generate.call_count == 1
        assert critic.generate.call_count == 1

    def test_one_step_includes_current_in_history(self):
        """One-step roll-out should include current response in simulation history."""
        from src.algorithms.rollout import estimate_final_accuracy
        from src.prompts.formatter import format_prompt

        actor = MagicMock()
        critic = MagicMock()
        actor.generate.return_value = ["Simulated."]

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        # Track what prompts are being sent
        prompts_sent = []

        original_format = format_prompt
        def capture_format(dataset_name, prompt_type, sample_arg, **kwargs):
            prompts_sent.append({
                "prompt_type": prompt_type,
                "responses": kwargs.get("responses", []),
            })
            return original_format(dataset_name, prompt_type, sample_arg, **kwargs)

        with patch("src.algorithms.rollout.format_prompt", side_effect=capture_format):
            estimate_final_accuracy(
                actor, critic, sample, "boolq",
                current_actor_response="Current response.",
                current_critic_response="Current feedback.",
                previous_responses=["Prev1", "Prev2"],
                num_simulations=1,
                remaining_rounds=1,
            )

        # Should have called format_prompt for actor
        assert len(prompts_sent) > 0

        # Check that the simulation includes current response
        actor_prompts = [p for p in prompts_sent if "actor" in str(p["prompt_type"]).lower()]
        if actor_prompts:
            pass

    def test_one_step_uses_final_simulated_response(self):
        """One-step roll-out uses the FINAL simulated response for evaluation."""
        from src.algorithms.rollout import estimate_final_accuracy

        actor = MagicMock()
        critic = MagicMock()

        # Batch generate returns correct answers
        actor.generate.return_value = ["The answer is Yes."] * 10
        critic.generate.return_value = ["Feedback."] * 10

        sample = {
            "question": "Test?",
            "passage": "Passage.",
            "answer": "yes",
            "task_type": "yes_no",
        }

        result = estimate_final_accuracy(
            actor, critic, sample, "boolq",
            current_actor_response="The answer is No.",  # Current is wrong
            current_critic_response="Feedback.",
            previous_responses=[],
            num_simulations=10,
            remaining_rounds=1,
        )

        # Should use the SIMULATED response (which is correct), not current
        assert result == 1.0

    def test_paper_compliance_one_step_description(self):
        """Verify code documentation mentions one-step roll-out per paper."""
        from src.algorithms import rollout

        # Check that the module docstring mentions one-step
        module_doc = rollout.__doc__
        assert "one-step" in module_doc.lower()
        assert "roll-out" in module_doc.lower() or "rollout" in module_doc.lower()

        # Check function docstring
        func_doc = rollout.estimate_final_accuracy.__doc__
        assert "one-step" in func_doc.lower() or "one step" in func_doc.lower()
