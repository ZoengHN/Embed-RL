set -e

MODEL_ID="./checkpoints/embed-rl-4b"

CONFIG_DIR="./scripts/eval_uvrb/eval_uvrb_config"

COT_ROOT="./test_data/Alibaba-NLP/UVRB_cot"

OUTPUT_DIR="./results/qwen3vl4b_uvrb"

mkdir -p "${OUTPUT_DIR}"

DATASETS=(
  "CMRB.yaml"
  "CRB_G.yaml"
  "CRB_S.yaml"
  "CRB_T.yaml"
  "DiDeMo.yaml"
  "DREAM_E.yaml"
  "LoVR_C2V.yaml"
  "LoVR_TH.yaml"
  "LoVR_V.yaml"
  "MSRVTT.yaml"
  "MSRVTT_I2V.yaml"
  "MS_TI.yaml"
  "MS_TV.yaml"
  "PEV_K.yaml"
  "VDC_D.yaml"
  "VDC_O.yaml"
)

for cfg in "${DATASETS[@]}"; do
  CONFIG_PATH="${CONFIG_DIR}/${cfg}"
  echo "====== Evaluating ${CONFIG_PATH} with CoT ======"
  CUDA_VISIBLE_DEVICES=0 python -m eval.eval_uvrb_recall \
    --config "${CONFIG_PATH}" \
    --model_id "${MODEL_ID}" \
    --output_dir "${OUTPUT_DIR}" \
    --cot_root "${COT_ROOT}" \
    --concat_query_cot \
    --concat_corpus_cot
done

echo "====== All evaluations completed! ======"