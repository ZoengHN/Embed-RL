#!/bin/bash
set -x
export PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH=./flash-attention:$PYTHONPATH
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
PROJECT_DIR=./
CONFIG_PATH="$PROJECT_DIR/examples/vlm2vec/8gpu"
python3 -m verl.trainer.main_ppo --config-path="$CONFIG_PATH" --config-name vlm2vec_grpo $@