#!/bin/bash

# Configurable paths (replace with your actual paths)
ORIGINAL_MODEL="PATH_TO_QWEN3_VL_BASE_MODEL"
CHECKPOINT_DIR="PATH_TO_LORA_CHECKPOINT_DIR"
SAVE_PATH="PATH_TO_SAVE_MERGED_MODEL"

echo "========================================"
echo "Merging LoRA Checkpoint"
echo "========================================"
echo "Original Model: ${ORIGINAL_MODEL}"
echo "Checkpoint:     ${CHECKPOINT_DIR}"
echo "Save Path:      ${SAVE_PATH}"
echo "========================================"

CUDA_VISIBLE_DEVICES='1' python merge_lora/merge_qwen3vl.py \
    --original_model_id ${ORIGINAL_MODEL} \
    --model_id ${CHECKPOINT_DIR} \
    --save_path ${SAVE_PATH}

echo ""
echo "✓ Merge completed!"
echo "Merged model saved to: ${SAVE_PATH}"