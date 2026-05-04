# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Optional

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    # Only enforce output structure for the final answer:
    # - exactly one <answer>...</answer> block
    # - exactly one \boxed{...} inside that answer block
    if response.count("<answer>") != 1 or response.count("</answer>") != 1:
        return 0.0

    match = re.search(r"<answer>(.*)</answer>", response, re.DOTALL)
    if match is None:
        return 0.0

    answer_content = match.group(1)
    boxed = re.findall(r"\\boxed\{.*?\}", answer_content, re.DOTALL)
    return 1.0 if len(boxed) == 1 else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def hallucination_reward(reward_input: dict[str, Any]) -> Optional[float]:
    if "verifier_hallucination_score" not in reward_input:
        return None

    try:
        score = float(reward_input["verifier_hallucination_score"])
    except (TypeError, ValueError):
        score = 0.5

    return max(0.0, min(1.0, score))


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    hallucination_weight: float = 0.0,
) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    if not 0 <= format_weight <= 1:
        raise ValueError("`format_weight` must be within [0, 1].")

    if not 0 <= hallucination_weight <= 1:
        raise ValueError("`hallucination_weight` must be within [0, 1].")

    if format_weight + hallucination_weight > 1:
        raise ValueError("`format_weight + hallucination_weight` must be <= 1.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        hallucination_score = hallucination_reward(reward_input)

        if hallucination_score is None:
            overall = (1 - format_weight) * accuracy_score + format_weight * format_score
            score = {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        else:
            accuracy_weight = 1 - format_weight - hallucination_weight
            overall = (
                accuracy_weight * accuracy_score
                + format_weight * format_score
                + hallucination_weight * hallucination_score
            )
            score = {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
                "hallucination": hallucination_score,
            }

        scores.append(score)

    return scores
