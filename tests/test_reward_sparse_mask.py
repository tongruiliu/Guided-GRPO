import numpy as np
import torch

from verl.protocol import DataProto
from verl.workers.reward.config import RewardConfig
from verl.workers.reward.function import BatchFunctionRewardManager


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(token_id)) for token_id in token_ids)


def test_batch_reward_decodes_full_response_and_rewards_last_trainable_token(tmp_path):
    reward_file = tmp_path / "reward_fn.py"
    reward_file.write_text(
        """
def compute_score(reward_inputs):
    assert reward_inputs[0]["response"] == "AVB"
    assert reward_inputs[0]["response_length"] == 3
    return [{"overall": 1.0, "decoded": 1.0}]
""".strip(),
        encoding="utf-8",
    )
    config = RewardConfig(reward_function=f"{reward_file}:compute_score")
    config.post_init()
    manager = BatchFunctionRewardManager(config, FakeTokenizer())

    data = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[ord("A"), ord("V"), ord("B"), 0]]),
            "response_mask": torch.tensor([[1, 0, 1, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        },
        non_tensors={"ground_truth": np.array(["unused"], dtype=object)},
    )

    reward_tensor, metrics = manager.compute_reward(data)

    assert reward_tensor.tolist() == [[0.0, 0.0, 1.0, 0.0]]
    assert metrics["decoded"] == [1.0]
