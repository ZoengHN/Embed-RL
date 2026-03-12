#!/usr/bin/env bash
set -euo pipefail

export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

cd ../../code/VR/Embed_RL

HOSTFILE="../../code/VR/Embed_RL/scripts/train_mmeb_embed/qwen3vl4b/hostfile"
MASTER_PORT=28533
DEEPSPEED_CONFIG="../../code/VR/Embed_RL/scripts/train_mmeb_embed/qwen3vl4b/zero2.json"

MODEL_NAME_OR_PATH="../../model/Qwen/Qwen3-VL-4B-Instruct"
MODEL_LOCAL_PATH="../../model/Qwen/Qwen3-VL-4B-Instruct"
DATA_CONFIG="../../code/VR/Embed_RL/scripts/train_mmeb_embed/qwen3vl4b/weight_train_data_config.yaml"

current_time=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./checkpoints/qwen3vl4b_${current_time}"
RUN_ID="genvr_qwen3vl4b_subbs256_bs512_wocls"

mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Starting Multi-Node Training with DeepSpeed"
echo "=========================================="
echo "Hostfile: ${HOSTFILE}"
echo "Master Port: ${MASTER_PORT}"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Data config: ${DATA_CONFIG}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Current time: ${current_time}"
echo ""

if [ ! -f "$HOSTFILE" ]; then
    echo "Error: Hostfile not found at $HOSTFILE"
    exit 1
fi

echo "Hostfile content:"
cat $HOSTFILE
echo ""

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Error: DeepSpeed config not found at $DEEPSPEED_CONFIG"
    exit 1
fi

deepspeed --hostfile $HOSTFILE --master_port $MASTER_PORT \
    train/train_mmeb_embed_cliploss_zero2.py \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_local_path $MODEL_LOCAL_PATH \
    --data_config $DATA_CONFIG \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_ID \
    --do_train \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 30 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_steps 10 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory True \
    --train_vision_encoder False \
    --train_vision_projector False \
    --use_lora True \
    --q_lora False \
    --lora_r 96 \
    --lora_alpha 192 \
    --report_to tensorboard \
    --num_sub_batches_per_batch 2 \
    --max_video_sub_batches_per_batch 1 \
    --mini_batch_size 8

echo "Training completed!"