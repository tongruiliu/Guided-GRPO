#!/bin/bash
# ray start --head --num-gpus=8 --dashboard-host=0.0.0.0

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

MODEL_PATH= # Replace with the actual model path
VERIFIER_PATH= # Replace with the actual model path
CONFIG_PATH=examples/config_multi_turn.yaml

NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

export TRAINER_NNODES=${NNODES}
export TRAINER_N_GPUS_PER_NODE=${GPUS_PER_NODE}

# export http_proxy=
# export https_proxy=

CKPT_DIR=${CHECKPOINT_SAVE:?Please set CHECKPOINT_SAVE}

python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.verifier.model_path=${VERIFIER_PATH} \
    trainer.save_checkpoint_path=${CKPT_DIR} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE}
