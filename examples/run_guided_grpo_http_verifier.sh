#!/bin/bash

set -x

# export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_CUDA_GRAPH=0
export CUDA_LAUNCH_BLOCKING=1
export VLLM_LOGGING_LEVEL=DEBUG

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.." || exit 1

MODEL_PATH= # Replace with the actual model path
CONFIG_PATH=examples/config_multi_turn_http.yaml

NNODES=2
GPUS_PER_NODE=8

export TRAINER_NNODES=${NNODES}
export TRAINER_N_GPUS_PER_NODE=${GPUS_PER_NODE}

CKPT_DIR=${CHECKPOINT_SAVE:?Please set CHECKPOINT_SAVE}/test1

if [ -n "${VERIFIER_BASE_URL:-}" ]; then
  EXTRA_VERIFIER_OPTS="worker.rollout.verifier.base_url=${VERIFIER_BASE_URL}"
fi
if [ -n "${VERIFIER_MODEL:-}" ]; then
  EXTRA_VERIFIER_OPTS="${EXTRA_VERIFIER_OPTS} worker.rollout.verifier.model=${VERIFIER_MODEL}"
fi
if [ -n "${VERIFIER_API_KEY:-}" ]; then
  EXTRA_VERIFIER_OPTS="${EXTRA_VERIFIER_OPTS} worker.rollout.verifier.api_key=${VERIFIER_API_KEY}"
fi

if [ -n "${SAVE_FREQ:-}" ]; then
  EXTRA_SAVE_OPTS="trainer.save_freq=${SAVE_FREQ}"
fi
if [ -n "${SAVE_LIMIT:-}" ]; then
  EXTRA_SAVE_OPTS="${EXTRA_SAVE_OPTS} trainer.save_limit=${SAVE_LIMIT}"
fi
if [ -n "${SAVE_KEEP_STEPS:-}" ]; then
  EXTRA_SAVE_OPTS="${EXTRA_SAVE_OPTS} trainer.save_keep_steps=${SAVE_KEEP_STEPS}"
fi

python3 -m verl.trainer.main \
    config=${CONFIG_PATH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.save_checkpoint_path=${CKPT_DIR} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    worker.rollout.verifier.http_concurrency=${VERIFIER_HTTP_CONCURRENCY:-1} \
    ${EXTRA_VERIFIER_OPTS} \
    ${EXTRA_SAVE_OPTS}
