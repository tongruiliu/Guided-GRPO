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
from typing import Iterable, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel

from ...protocol import DataProto
from ...utils.fsdp_utils import load_fsdp_model, offload_fsdp_model
from ...utils.model_utils import print_gpu_memory_usage
from .base import BaseShardingManager


class FSDPSGLangShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine,
        use_param_offload: bool,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.use_param_offload = use_param_offload
        self.loaded = False
        self.world_size = dist.get_world_size()

        self.freed_bytes = 0
        self.torch_random_states = torch.cuda.get_rng_state()
        self.gen_random_states = torch.cuda.get_rng_state()

    def _rename_weight_keys(self, actor_weights: dict[str, Union[torch.Tensor, DTensor]], model: PreTrainedModel):
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return actor_weights

        reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
        original_weights = {}
        for key, value in actor_weights.items():
            for pattern, replacement in reverse_key_mapping.items():
                replacement = replacement.lstrip("^")
                replacement = re.sub(r"\(.*\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, key)
                if n_replace > 0:
                    break

            original_weights[key] = value

        return original_weights

    def _make_weight_iterator(
        self, actor_weights: dict[str, Union[torch.Tensor, DTensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in actor_weights.items():
            yield name, tensor.full_tensor() if self.world_size != 1 else tensor

    def _sync_weight_to_sglang(self):
        if self.use_param_offload:
            load_fsdp_model(self.module)

        actor_weights = get_model_state_dict(self.module)
        actor_weights = self._rename_weight_keys(actor_weights, self.module._fsdp_wrapped_module)
        print_gpu_memory_usage("After gather model weights in SGLang sharding manager")

        self.inference_engine.update_weights_from_tensor(
            self._make_weight_iterator(actor_weights),
            load_format=None,
            flush_cache=True,
        )

        del actor_weights
        if self.use_param_offload:
            offload_fsdp_model(self.module)

        torch.cuda.empty_cache()
        print_gpu_memory_usage("After sync model weights in SGLang sharding manager")

    def load_vllm_and_sync_weights(self):
        torch.cuda.empty_cache()
        assert self.loaded is False, "SGLang engine has already been loaded"
        self.loaded = True

        print_gpu_memory_usage("Before SGLang wake up in sharding manager")
        self.inference_engine.wake_up(tags=["weights"])
        self._sync_weight_to_sglang()
        self.inference_engine.wake_up(tags=["kv_cache"])
        print_gpu_memory_usage("After SGLang wake up in sharding manager")

        self.torch_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.gen_random_states)

    def offload_vllm(self):
        assert self.loaded is True, "SGLang engine has not been loaded"
        self.loaded = False

        print_gpu_memory_usage("Before SGLang offload in sharding manager")
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
        self.inference_engine.sleep(level=1)
        free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
        self.freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        print_gpu_memory_usage("After SGLang offload in sharding manager")

        self.module.train()
        torch.cuda.empty_cache()

        self.gen_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        return data
