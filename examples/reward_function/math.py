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
from typing import Any

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


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    if not 0 <= format_weight <= 1:
        raise ValueError("`format_weight` must be within [0, 1].")

    accuracy_weight = 1 - format_weight
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (accuracy_weight * accuracy_score + format_weight * format_score),
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
