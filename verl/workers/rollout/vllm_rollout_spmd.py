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

import copy
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Optional, Union
import urllib.error
import urllib.request

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from ...utils.tokenizer import get_tokenizer
from .base import BaseRollout
from .config import RolloutConfig

VERIFIER_CONTINUE_PROMPT = "Please continue and provide the final answer in the required format."


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


@contextmanager
def _patch_vllm_dist_env():
    """
    vLLM init internally reads RANK/LOCAL_RANK/WORLD_SIZE to pick a CUDA device.
    Under Ray each actor is already constrained to a single GPU (CUDA_VISIBLE_DEVICES set
    to a single physical id), so keeping the original FSDP ranks can produce an invalid
    device ordinal (e.g., LOCAL_RANK=7 but only one visible GPU). Temporarily reset the
    dist env so vLLM always sees rank 0 / world size 1 during engine construction.
    """
    keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    backup = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.update({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1"})
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)
        yield
    finally:
        for key, val in backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.processor = processor
        self.verifier_tokenizer = None
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        with _patch_vllm_dist_env():
            self.inference_engine = LLM(
                model=model_path,
                skip_tokenizer_init=False,
                trust_remote_code=config.trust_remote_code,
                load_format="dummy",
                dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
                seed=config.seed,
                max_model_len=config.max_model_len or config.prompt_length + config.response_length,
                distributed_executor_backend="external_launcher",
                tensor_parallel_size=config.tensor_parallel_size,
                gpu_memory_utilization=config.gpu_memory_utilization,
                max_num_batched_tokens=config.max_num_batched_tokens,
                disable_log_stats=config.disable_log_stats,
                enforce_eager=config.enforce_eager,
                disable_custom_all_reduce=True,
                enable_chunked_prefill=config.enable_chunked_prefill,
                enable_sleep_mode=config.enable_sleep_mode,
                **engine_kwargs,
            )

        # Offload vllm model to reduce peak memory usage.
        if config.enable_sleep_mode:
            self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.use_multi_turn = config.verifier.enable
        self.verifier_engine = None
        self.verifier_sampling_params = None
        self.verifier_tokenizer = None
        self.verifier_use_http = False
        self.verifier_http_base_url = ""
        self.verifier_http_model = ""
        self.verifier_http_api_key = ""
        self.verifier_http_timeout = 30.0
        self.verifier_http_max_retries = 3
        self.verifier_http_concurrency = 1
        self.verifier_prompt_template = config.verifier.prompt_template.strip()
        self.verifier_stop_token = config.verifier.stop_token.strip()
        self.prompt_truncation = config.prompt_truncation
        self.multi_turn_samples = 1
        if self.use_multi_turn:
            self.multi_turn_samples = max(1, self.sampling_params.n)
            self.sampling_params.n = 1
            self.verifier_use_http = getattr(config.verifier, "use_http", False)
            if self.verifier_use_http:
                if not config.verifier.base_url:
                    raise ValueError("`worker.rollout.verifier.base_url` must be set when use_http=True.")
                self.verifier_http_base_url = config.verifier.base_url.rstrip("/")
                self.verifier_http_model = config.verifier.model or "verifier"
                self.verifier_http_api_key = config.verifier.api_key
                self.verifier_http_timeout = config.verifier.http_timeout
                self.verifier_http_max_retries = config.verifier.http_max_retries
                self.verifier_http_concurrency = max(1, int(getattr(config.verifier, "http_concurrency", 4)))
            else:
                verifier_model_path = config.verifier.model_path
                if not verifier_model_path:
                    raise ValueError("`worker.rollout.verifier.model_path` must be set when multi-turn is enabled.")
                # Sleep mode只能在一个进程里被一个 vLLM 实例使用；主引擎需要 sleep 以配合权重 offload，
                # 验证器这里禁用 sleep_mode 以避免 “Sleep mode can only be used for one instance per process”.
                verifier_enable_sleep_mode = False
                # 验证器走 eager，避免 Torch Compile/triton autotune 在 profile_run 里触发 driver “invalid argument”。
                verifier_enforce_eager = True
                with _patch_vllm_dist_env():
                    self.verifier_engine = LLM(
                        model=verifier_model_path,
                        skip_tokenizer_init=False,
                        trust_remote_code=config.verifier.trust_remote_code,
                        load_format="dummy",
                        dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
                        seed=config.seed,
                        max_model_len=config.max_model_len or config.prompt_length + config.response_length,
                        distributed_executor_backend="external_launcher",
                        tensor_parallel_size=config.tensor_parallel_size,
                        gpu_memory_utilization=config.gpu_memory_utilization,
                        max_num_batched_tokens=config.max_num_batched_tokens,
                        disable_log_stats=config.disable_log_stats,
                        enforce_eager=verifier_enforce_eager,
                        disable_custom_all_reduce=True,
                        enable_chunked_prefill=config.enable_chunked_prefill,
                        enable_sleep_mode=verifier_enable_sleep_mode,
                    )
                if verifier_enable_sleep_mode:
                    self.verifier_engine.sleep(level=1)
                verifier_sampling_kwargs = {
                    "max_tokens": config.verifier.max_tokens,
                    "temperature": config.verifier.temperature,
                    "top_p": config.verifier.top_p,
                    "top_k": config.verifier.top_k,
                    "detokenize": True,
                }
                self.verifier_sampling_params = SamplingParams(**verifier_sampling_kwargs)
                verifier_tokenizer_path = config.verifier.tokenizer_path or verifier_model_path
                self.verifier_tokenizer = get_tokenizer(
                    verifier_tokenizer_path,
                    trust_remote_code=config.verifier.trust_remote_code,
                    use_fast=True,
                )

    def _normalize_verifier_text(self, text: str) -> str:
        text = str(text or "").strip()
        if not text:
            return self.verifier_stop_token

        # Treat any prefix that starts with the stop token as stop. This makes the loop robust to
        # common verifier outputs like "###STOP###\\n..." or "###STOP####".
        if self.verifier_stop_token and text.startswith(self.verifier_stop_token):
            return self.verifier_stop_token

        # Some verifiers may emit role prefixes; strip a leading "User:" to reduce prompt pollution.
        lowered = text.lower()
        if lowered.startswith("user:"):
            text = text.split(":", 1)[1].lstrip()

        return text

    def _call_http_verifier_batch(self, prompts: list[str]) -> list[str]:
        concurrency = max(1, int(self.verifier_http_concurrency))
        if concurrency <= 1 or len(prompts) <= 1:
            return [self._call_http_verifier(prompt) for prompt in prompts]

        results: list[str] = [self.verifier_stop_token for _ in range(len(prompts))]
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(self._call_http_verifier, prompt): i for i, prompt in enumerate(prompts)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:  # pragma: no cover - runtime guard
                    results[idx] = self.verifier_stop_token
        return results

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        prompts.meta_info["eos_token_id"] = eos_token_id
        rollout_sharding_manager = getattr(self, "rollout_sharding_manager", None)
        if rollout_sharding_manager is not None:
            prompts = rollout_sharding_manager.preprocess_data(prompts)
        if self.use_multi_turn:
            output = self._generate_multi_turn(prompts)
        else:
            output = self._generate_single_turn(prompts)
        if rollout_sharding_manager is not None:
            output = rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        return output

    def _generate_single_turn(self, prompts: DataProto) -> DataProto:
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=prompts.meta_info["eos_token_id"], dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def _generate_multi_turn(self, prompts: DataProto) -> DataProto:
        device = prompts.batch["input_ids"].device
        base_batch_size = prompts.batch["input_ids"].size(0)
        sample_repeats = int(prompts.meta_info.get("n", self.multi_turn_samples)) if prompts.meta_info else self.multi_turn_samples
        sample_repeats = max(1, sample_repeats)
        meta_info = prompts.meta_info
        non_tensor_batch = prompts.non_tensor_batch
        raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids", None)
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        messages_array = non_tensor_batch.pop("messages", None)
        if messages_array is None:
            # 兼容缺失 messages 的场景，回退用 raw_prompt_ids 重建用户首轮消息，避免直接崩溃
            if raw_prompt_ids is not None:
                base_messages = [
                    [{"role": "user", "content": self.tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=True)}]
                    for prompt_ids in raw_prompt_ids
                ]
                messages_array = np.array(base_messages, dtype=object)
            else:
                raise ValueError("Multi-turn rollout requires `messages` field in the dataset.")
        ground_truth_array = non_tensor_batch.pop("ground_truth", None)

        base_messages = messages_array.tolist()
        repeated_messages = self._repeat_with_copy(base_messages, sample_repeats)
        total_batch_size = len(repeated_messages)

        if ground_truth_array is not None:
            base_ground_truth = ground_truth_array.tolist()
        else:
            base_ground_truth = [None] * base_batch_size
        ground_truth_list = self._repeat_plain_items(base_ground_truth, sample_repeats)

        if batch_multi_modal_data is not None:
            base_multi_modal = batch_multi_modal_data.tolist()
            repeated_multi_modal = self._repeat_with_copy(base_multi_modal, sample_repeats)
            raw_multi_modal_np = np.array(repeated_multi_modal, dtype=object)
            processed_multi_modal = [
                _process_multi_modal_data(
                    mm,
                    meta_info["min_pixels"],
                    meta_info["max_pixels"],
                    meta_info["video_fps"],
                )
                for mm in repeated_multi_modal
            ]
        else:
            raw_multi_modal_np = None
            processed_multi_modal = [None] * total_batch_size

        conversation_states = []
        for messages, ground_truth, mm_data in zip(repeated_messages, ground_truth_list, processed_multi_modal):
            convo_messages = self._ensure_message_list(messages)
            state = {
                "messages": convo_messages,
                "prompt_history": [],
                "assistant_tokens": [],
                "multi_modal": mm_data,
                "ground_truth": ground_truth,
                "hallucination_score_sum": 0.0,
                "hallucination_score_count": 0,
                "finished": False,
            }
            conversation_states.append(state)

        active_indices = list(range(len(conversation_states)))
        with self.update_sampling_params(**meta_info):
            for _ in range(self.config.verifier.max_turns):
                if not active_indices:
                    break

                vllm_inputs = []
                for idx in active_indices:
                    state = conversation_states[idx]
                    if state["multi_modal"] is not None:
                        state["messages"], state["multi_modal"] = self._align_mm_with_messages(
                            state["messages"], state["multi_modal"]
                        )
                    if state["multi_modal"] is None:
                        # 文本对话保留 assistant 历史与最新 verifier 反馈，超长时从最早 assistant 起截断
                        state["messages"] = self._truncate_messages(state["messages"], self.config.prompt_length)
                    prompt_text = self._render_prompt(state["messages"], state["multi_modal"])
                    if state["multi_modal"] is not None:
                        prompt_token_ids = self._encode_prompt_with_mm(prompt_text, state["multi_modal"])
                        if self.rank == 0 and not getattr(self, "_mm_debug_printed", False):
                            vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
                            image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                            vision_start_count = (
                                prompt_token_ids.count(vision_start_id)
                                if isinstance(vision_start_id, int) and vision_start_id >= 0
                                else 0
                            )
                            image_pad_count = (
                                prompt_token_ids.count(image_pad_id)
                                if isinstance(image_pad_id, int) and image_pad_id >= 0
                                else 0
                            )
                            num_images = len(state["multi_modal"].get("image", []))
                            print(
                                "[mm-debug] images="
                                + str(num_images)
                                + " vision_start_count="
                                + str(vision_start_count)
                                + " image_pad_count="
                                + str(image_pad_count)
                                + " prompt_len="
                                + str(len(prompt_token_ids))
                            )
                            self._mm_debug_printed = True
                    else:
                        prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                    state["prompt_history"].append(prompt_token_ids)
                    if state["multi_modal"] is not None:
                        vllm_input = {"prompt": prompt_text, "multi_modal_data": state["multi_modal"]}
                    else:
                        vllm_input = {"prompt_token_ids": prompt_token_ids}
                    vllm_inputs.append(vllm_input)

                completions: list[RequestOutput] = self.inference_engine.generate(
                    prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
                )
                for idx, completion in zip(active_indices, completions):
                    output = completion.outputs[0]
                    response_tokens = output.token_ids
                    conversation_states[idx]["assistant_tokens"].append(response_tokens)
                    response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                    conversation_states[idx]["messages"].append({"role": "assistant", "content": response_text})
                    conversation_states[idx]["last_assistant_has_final"] = self._has_final_answer(response_text)

                verifier_prompts = self._build_verifier_prompts(conversation_states, active_indices)
                if self.verifier_use_http:
                    verifier_texts = self._call_http_verifier_batch(verifier_prompts)
                else:
                    verifier_completions: list[RequestOutput] = self.verifier_engine.generate(
                        prompts=verifier_prompts,
                        sampling_params=self.verifier_sampling_params,
                        use_tqdm=False,
                    )
                    verifier_texts = [completion.outputs[0].text.strip() for completion in verifier_completions]

                next_active_indices = []
                for idx, verifier_text in zip(active_indices, verifier_texts):
                    verifier_text = self._normalize_verifier_text(verifier_text)
                    assistant_has_final = conversation_states[idx].get("last_assistant_has_final", False)
                    if self._is_abnormal_verifier_text(verifier_text):
                        verifier_text = self.verifier_stop_token
                    if self.config.verifier.enable_hallucination_score:
                        score = self._parse_hallucination_score(verifier_text)
                        if score is not None:
                            conversation_states[idx]["hallucination_score_sum"] += score
                            conversation_states[idx]["hallucination_score_count"] += 1
                    if verifier_text == self.verifier_stop_token:
                        if assistant_has_final:
                            conversation_states[idx]["finished"] = True
                            conversation_states[idx]["messages"].append({"role": "user", "content": verifier_text})
                        else:
                            conversation_states[idx]["messages"].append(
                                {"role": "user", "content": VERIFIER_CONTINUE_PROMPT}
                            )
                            next_active_indices.append(idx)
                    else:
                        conversation_states[idx]["messages"].append({"role": "user", "content": verifier_text})
                        next_active_indices.append(idx)

                active_indices = next_active_indices
                if not active_indices:
                    break

        prompt_inputs, prompt_attention_masks, prompt_position_ids = [], [], []
        response_token_list = []
        for state in conversation_states:
            if not state["prompt_history"] or not state["assistant_tokens"]:
                raise RuntimeError("Each sample must have at least one assistant response in multi-turn mode.")
            prompt_tokens = state["prompt_history"][-1]
            prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
            attention_tensor = torch.ones_like(prompt_tensor, dtype=torch.long)
            position_tensor = torch.arange(len(prompt_tokens), dtype=torch.long, device=device)
            prompt_tensor, attention_tensor, position_tensor = VF.postprocess_data(
                input_ids=prompt_tensor,
                attention_mask=attention_tensor,
                position_ids=position_tensor,
                max_length=self.config.prompt_length,
                pad_token_id=self.pad_token_id,
                left_pad=True,
                truncation=self.prompt_truncation,
            )
            prompt_inputs.append(prompt_tensor)
            prompt_attention_masks.append(attention_tensor)
            prompt_position_ids.append(position_tensor)
            response_token_list.append(state["assistant_tokens"][-1])

        input_ids = torch.stack(prompt_inputs, dim=0)
        attention_mask = torch.stack(prompt_attention_masks, dim=0)
        position_ids = torch.stack(prompt_position_ids, dim=0)
        response_ids = VF.pad_2d_list_to_length(
            response_token_list, self.pad_token_id, max_length=self.config.response_length
        ).to(device)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=device)
        batch_size = len(conversation_states)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=meta_info["eos_token_id"], dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if raw_multi_modal_np is not None:
            non_tensor_batch = {"multi_modal_data": raw_multi_modal_np}
        else:
            non_tensor_batch = {}

        if self.config.verifier.enable_hallucination_score:
            hallucination_scores = []
            for state in conversation_states:
                count = int(state.get("hallucination_score_count", 0))
                if count > 0:
                    score = float(state.get("hallucination_score_sum", 0.0)) / float(count)
                else:
                    score = 0.5
                hallucination_scores.append(score)
            non_tensor_batch["verifier_hallucination_score"] = np.array(hallucination_scores, dtype=np.float32)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def _repeat_with_copy(self, items: list[Any], repeats: int) -> list[Any]:
        repeated = []
        for item in items:
            for _ in range(repeats):
                repeated.append(copy.deepcopy(item))
        return repeated

    def _repeat_plain_items(self, items: list[Any], repeats: int) -> list[Any]:
        return [item for item in items for _ in range(repeats)]

    def _ensure_message_list(self, messages: Any) -> list[dict[str, Any]]:
        if isinstance(messages, list):
            return copy.deepcopy(messages)
        if isinstance(messages, np.ndarray):
            return copy.deepcopy(messages.tolist())
        return [{"role": "user", "content": str(messages)}]

    def _count_message_images(self, messages: list[dict[str, Any]]) -> int:
        count = 0
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        count += 1
        return count

    def _trim_message_images(self, messages: list[dict[str, Any]], max_images: int) -> None:
        remaining = max_images
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    if remaining > 0:
                        new_content.append(item)
                        remaining -= 1
                    continue
                new_content.append(item)
            if not new_content:
                message["content"] = ""
            else:
                message["content"] = new_content

    def _align_mm_with_messages(
        self,
        messages: list[dict[str, Any]],
        multi_modal_inputs: Optional[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        if not multi_modal_inputs or "image" not in multi_modal_inputs:
            return messages, multi_modal_inputs
        images = list(multi_modal_inputs.get("image", []))
        num_images = len(images)
        if num_images == 0:
            self._trim_message_images(messages, 0)
            return messages, None

        max_images = num_images
        if self.config.limit_images:
            max_images = min(max_images, self.config.limit_images)
        placeholder_count = self._count_message_images(messages)
        if placeholder_count == 0:
            if self.rank == 0 and not getattr(self, "_mm_missing_placeholder_warned", False):
                print("[mm-warn] images present but no image placeholders in messages; drop images to avoid crash.")
                self._mm_missing_placeholder_warned = True
            return messages, None

        max_images = min(max_images, placeholder_count)
        if max_images != placeholder_count:
            self._trim_message_images(messages, max_images)
        if max_images != num_images:
            multi_modal_inputs = dict(multi_modal_inputs)
            multi_modal_inputs["image"] = images[:max_images]
        return messages, multi_modal_inputs

    def _render_prompt(self, messages: list[dict[str, Any]], multi_modal_inputs: Optional[dict[str, Any]]) -> str:
        if multi_modal_inputs is not None and self.processor is not None:
            return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _encode_prompt_with_mm(self, prompt_text: str, multi_modal_inputs: dict[str, Any]) -> list[int]:
        if self.processor is None:
            return self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if "image" in multi_modal_inputs:
            model_inputs = self.processor(
                multi_modal_inputs["image"],
                [prompt_text],
                add_special_tokens=False,
                return_tensors="pt",
            )
        elif "video" in multi_modal_inputs:
            model_inputs = self.processor(
                videos=multi_modal_inputs["video"],
                text=[prompt_text],
                add_special_tokens=False,
                return_tensors="pt",
            )
        else:
            return self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return model_inputs["input_ids"][0].tolist()

    def _flatten_message_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "image":
                        parts.append("[IMAGE]")
                    elif item.get("type") == "video":
                        parts.append("[VIDEO]")
            return "\n".join(part for part in parts if part)
        return str(content)

    def _extract_hallucination_ratio(self, messages: list[dict[str, Any]]) -> float:
        """Parse verifier `[SCORE] hallucination_detect=` lines and average them."""
        scores = []
        for message in messages:
            if message.get("role") != "user":
                continue
            content = self._flatten_message_content(message.get("content", ""))
            if content.strip() == self.verifier_stop_token:
                continue
            match = re.search(r"\[SCORE\]\s*hallucination_detect\s*=\s*([01])", content)
            if match:
                scores.append(int(match.group(1)))

        if not scores:
            return 0.5  # neutral when verifier did not return scores
        return float(sum(scores)) / len(scores)

    def _format_conversation_for_verifier(self, messages: list[dict[str, Any]]) -> str:
        lines = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            role = "Assistant"
            text = self._flatten_message_content(message.get("content", ""))
            lines.append(f"{role}: {text}")
        return "\n".join(lines)

    def _has_final_answer(self, response: str) -> bool:
        normalized = re.sub(r"\s*(<|>|/)\s*", r"\1", response)
        if normalized.count("<answer>") != 1 or normalized.count("</answer>") != 1:
            return False
        match = re.search(r"<answer>(.*)</answer>", normalized, re.DOTALL)
        if match is None:
            return False
        answer_content = match.group(1)
        boxed = re.findall(r"\\boxed\{.*?\}", answer_content, re.DOTALL)
        return len(boxed) == 1

    def _is_abnormal_verifier_text(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        return re.search(r"[A-Za-z0-9]", stripped) is None

    def _parse_hallucination_score(self, text: str) -> Optional[int]:
        match = re.search(r"\[SCORE\]\s*hallucination_detect\s*=\s*([01])", text)
        if match:
            return int(match.group(1))
        return None

    def _extract_problem_statement(self, messages: list[dict[str, Any]]) -> str:
        for message in messages:
            if message.get("role") == "user":
                return self._flatten_message_content(message.get("content", ""))
        return ""

    def _build_verifier_prompts(
        self, conversation_states: list[dict[str, Any]], active_indices: list[int]
    ) -> list[str]:
        prompts = []
        for idx in active_indices:
            state = conversation_states[idx]
            conversation_text = self._format_conversation_for_verifier(state["messages"])
            problem_text = self._extract_problem_statement(state["messages"])
            reference_answer = state.get("ground_truth") or "N/A"
            user_content = (
                f"### Problem Statement\n{problem_text}\n\n"
                f"### Reference Answer\n{reference_answer}\n\n"
                f"### Conversation History\n{conversation_text}\n\n"
                "Provide the next piece of guidance following the rules above."
            )
            if self.verifier_use_http:
                # For OpenAI-style HTTP chat/completions, the system prompt is already provided
                # separately in `_call_http_verifier`, so only send the user content here.
                prompts.append(user_content)
                continue
            if self.verifier_tokenizer is not None:
                verifier_messages = [
                    {"role": "system", "content": self.verifier_prompt_template},
                    {"role": "user", "content": user_content},
                ]
                prompt = self.verifier_tokenizer.apply_chat_template(
                    verifier_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            else:
                prompt = (
                    f"{self.verifier_prompt_template}\n\n"
                    f"{user_content}"
                )
            prompts.append(prompt)

        return prompts

    def _call_http_verifier(self, prompt: str) -> str:
        url = f"{self.verifier_http_base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.verifier_http_api_key:
            headers["Authorization"] = f"Bearer {self.verifier_http_api_key}"

        payload = {
            "model": self.verifier_http_model or "verifier",
            "messages": [
                {"role": "system", "content": self.verifier_prompt_template},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.verifier.max_tokens,
            "temperature": self.config.verifier.temperature,
            "top_p": self.config.verifier.top_p,
            "top_k": self.config.verifier.top_k,
        }
        data = json.dumps(payload).encode("utf-8")
        retries = max(1, int(self.verifier_http_max_retries))
        for attempt in range(retries):
            try:
                request = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.verifier_http_timeout) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                choices = parsed.get("choices", [])
                if choices:
                    if isinstance(choices[0], dict):
                        content = choices[0].get("message", {}).get("content") or choices[0].get("text", "")
                    else:
                        content = str(choices[0])
                else:
                    content = ""
                content = str(content).strip()
                if not content:
                    return self.verifier_stop_token
                return self._normalize_verifier_text(content)
            except Exception as e:  # pragma: no cover - runtime guard
                if attempt + 1 >= retries:
                    print(
                        f"[verifier-http] failed (attempt {attempt + 1}/{retries}): {e}. "
                        "Falling back to stop token."
                    )
                else:
                    sleep_time = getattr(self.config.verifier, "http_retry_interval", 45.0)
                    try:
                        import time

                        time.sleep(sleep_time)
                    except Exception:
                        pass
                continue

        return self.verifier_stop_token

    def _truncate_messages(self, messages: list[dict[str, Any]], max_length: int) -> list[dict[str, Any]]:
        """Keep system +首个 user +全部 assistant +最新 verifier(user)，超长时从最早 assistant 起删除。仅用于纯文本对话。"""
        if max_length <= 0 or len(messages) <= 2:
            return messages

        truncated = copy.deepcopy(messages)
        system_msg = truncated[0] if truncated and truncated[0].get("role") == "system" else None
        search_start = 1 if system_msg is not None else 0

        first_user_idx = None
        for i in range(search_start, len(truncated)):
            if truncated[i].get("role") == "user":
                first_user_idx = i
                break

        last_user_idx = None
        for i in range(len(truncated) - 1, -1, -1):
            if truncated[i].get("role") == "user":
                last_user_idx = i
                break

        assistant_msgs = [msg for msg in truncated if msg.get("role") == "assistant"]

        kept = []
        if system_msg is not None:
            kept.append(system_msg)
        if first_user_idx is not None:
            kept.append(truncated[first_user_idx])
        kept.extend(assistant_msgs)
        if last_user_idx is not None and last_user_idx != first_user_idx:
            kept.append(truncated[last_user_idx])

        prefix_count = (1 if system_msg is not None else 0) + (1 if first_user_idx is not None else 0)
        suffix_count = 1 if last_user_idx is not None and last_user_idx != first_user_idx else 0
        while len(kept) > prefix_count + suffix_count:
            try:
                token_ids = self.tokenizer.apply_chat_template(
                    kept, add_generation_prompt=True, tokenize=True
                )
                if len(token_ids) <= max_length:
                    break
            except Exception:
                break

            # 删除最早的 assistant 轮次（保留 system + 首个 user + 最新 verifier）
            kept.pop(prefix_count)

        return kept
