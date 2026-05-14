from __future__ import annotations

from src.prompts.critic_sft_prompts import (
    build_critic_sft_prompt,
    render_critic_sft_feedback,
)


def _sample():
    return {
        "question": "Which option is best supported?",
        "choices": ["Wrong A", "Right B", "Wrong C", "Wrong D"],
        "answer": "B",
        "task_type": "multiple_choice",
    }


def test_critic_sft_prompt_contains_role_actor_and_peer_summary():
    prompt = build_critic_sft_prompt(
        sample=_sample(),
        dataset_name="mmlu",
        critic_skill="grounding",
        actor_name="actor_direct",
        actor_response="I choose C.\nThe final result is C.",
        actor_answer="C",
        other_actor_summary="Actor-evidence answered B. Main reason: wording.",
    )

    assert prompt.startswith("/no_think\n")
    assert "You are Critic-grounding" in prompt
    assert "Target Actor:\nactor_direct" in prompt
    assert "Target Actor current response:" in prompt
    assert "Target Actor extracted answer:\nC" in prompt
    assert "Actor-evidence answered B" in prompt
    assert "Judgement:" in prompt


def test_critic_sft_feedback_varies_by_skill():
    reasoning = render_critic_sft_feedback(
        critic_skill="reasoning",
        case_type="correction",
        actor_name="actor_direct",
        actor_answer="C",
        target_answer="B",
        other_actor_summary="Actor-evidence answered B.",
    )
    verification = render_critic_sft_feedback(
        critic_skill="verification",
        case_type="correction",
        actor_name="actor_direct",
        actor_answer="C",
        target_answer="B",
        other_actor_summary="Actor-evidence answered B.",
    )

    assert reasoning != verification
    assert "reasoning chain" in reasoning
    assert "final answer check" in verification


def test_keep_feedback_tells_actor_to_preserve_answer():
    feedback = render_critic_sft_feedback(
        critic_skill="verification",
        case_type="keep",
        actor_name="actor_evidence",
        actor_answer="B",
        target_answer="B",
        other_actor_summary="Actor-direct answered C.",
    )

    assert "already correct" in feedback
    assert "keep the final answer as B" in feedback
    assert "Answer correct: yes" in feedback
    assert "Suggested answer: B" in feedback
