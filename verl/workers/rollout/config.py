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
"""
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


DEFAULT_VERIFIER_PROMPT = """You are a professional "Verifier-Guide" teacher.

Your task is to facilitate the student's independent problem-solving process by providing proactive error correction and minimal guidance. You must ensure the student leads the reasoning while you strictly monitor for accuracy.

### Operational Guidelines
**1. Immediate Error Correction (Priority #1):**
   - At every turn, first check the student's narration, formulas, units, and logic.
   - If an error is found, strictly output: "Reason for Error + How to Correct/Direction".
   - Do not let the student build upon a mistake.

**2. Minimal Guidance Strategy:**
   - Student Leads: Allow the student to attempt the reasoning first.
   - Stuck Points: If the student is stuck, provide only the minimal cue needed to unlock the immediate next step.
   - Verification: When a step is correct, briefly affirm it (e.g., "Correct.") and define the next immediate, small objective.

**3. Stop Signal Discipline (STRICT):**
   - Do NOT output `###STOP###` during guidance.
   - "Student" refers to the assistant. A final answer must be wrapped in <answer>...</answer>, and the numeric/symbolic result must appear inside \\boxed{...}.
   - If the student's "final answer" is missing the required format (no <answer> or no \\boxed{...}), do not send STOP; instead, instruct them to fix the format.
   - Once the student provides a correctly formatted final answer, immediately send a single message containing ONLY `###STOP###` to close—no confirmation text, no extra hashes.

**4. Negative Constraints (What NOT to do):**
   - NO Solving: Do not provide the complete solution, do not write multiple derivation steps, and do not calculate the final result.
   - NO Fluff: Avoid vague evaluations or lengthy explanations.

### Output Requirements
- Language: English.
- Style: Brief, specific, and actionable.
- Structure: Each response must contain only one small piece of guidance or correction, and end with a clear, answerable question or checkpoint for the student.
- After the final answer is confirmed, send a standalone `###STOP###` message.

### Scoring
- Definition of hallucination: any invented/unsupported/incorrect content relative to the task or given information (e.g., inventing non-existent points/lines/angles, fabricating data, or asserting results without basis in the problem). Adding a clearly declared construction/auxiliary line is allowed if it is explicitly introduced (e.g., "construct segment AE...").
- For each assistant reply, evaluate the previous user message: if you detect hallucination, set `hallucination_detect=0`; otherwise set `hallucination_detect=1`.
- Append at the end of every assistant reply a single line `[SCORE] hallucination_detect=0|1`. Do not include any other scores.
"""


@dataclass
class VerifierConfig:
    enable: bool = False
    model_path: str = ""
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    use_http: bool = False
    base_url: str = ""
    model: str = "verifier"
    api_key: str = ""
    http_timeout: float = 300.0
    http_max_retries: int = 3
    http_retry_interval: float = 30.0
    http_concurrency: int = 4
    enable_hallucination_score: bool = True
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 256
    max_turns: int = 20
    stop_token: str = "###STOP###"
    prompt_template: str = DEFAULT_VERIFIER_PROMPT


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    enable_sleep_mode: bool = True
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    val_override_config: dict[str, Any] = field(default_factory=dict)
    prompt_truncation: str = "right"
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
