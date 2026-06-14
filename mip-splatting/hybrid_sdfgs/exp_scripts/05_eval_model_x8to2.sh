#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/root/autodl-tmp/HBSR"
DATASET_PATH="/root/autodl-tmp/kitchen"
EVAL_IMAGES="${EVAL_IMAGES:-images_2}"
MODEL_PATH="${1:-/root/autodl-tmp/HBSR/outputs/hybrid_gsbootstrap_sdfdensify_4x_kitchen_x8to2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[eval-x8to2] model path not found: ${MODEL_PATH}"
  exit 1
fi

cd "${HBSR_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python render.py \
  -s "${DATASET_PATH}" \
  -i "${EVAL_IMAGES}" \
  -m "${MODEL_PATH}" \
  --iteration -1 \
  --resolution -1 \
  --skip_train \
  --white_background

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python metrics.py \
  -m "${MODEL_PATH}" \
  -r -1

echo "[eval-x8to2] done: ${MODEL_PATH} (eval_images=${EVAL_IMAGES})"
