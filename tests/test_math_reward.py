import pytest

from examples.reward_function.math import compute_score


def test_math_reward_uses_hallucination_score_when_present():
    scores = compute_score(
        [
            {
                "response": r"<answer>\boxed{2}</answer>",
                "response_length": 1,
                "ground_truth": "2",
                "verifier_hallucination_score": 0.0,
            }
        ],
        format_weight=0.1,
        hallucination_weight=0.2,
    )

    assert scores[0]["accuracy"] == 1.0
    assert scores[0]["format"] == 1.0
    assert scores[0]["hallucination"] == 0.0
    assert scores[0]["overall"] == pytest.approx(0.8)


def test_math_reward_keeps_original_weighting_without_hallucination_score():
    scores = compute_score(
        [
            {
                "response": r"<answer>\boxed{2}</answer>",
                "response_length": 1,
                "ground_truth": "2",
            }
        ],
        format_weight=0.1,
        hallucination_weight=0.2,
    )

    assert "hallucination" not in scores[0]
    assert scores[0]["overall"] == pytest.approx(1.0)
