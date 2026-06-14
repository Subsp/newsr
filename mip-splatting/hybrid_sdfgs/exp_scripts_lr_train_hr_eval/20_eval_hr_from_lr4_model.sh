#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/data3/liutl/HBSR"
DATASET_PATH="/data3/liutl/nerf_synthetic/lego"
MODEL_PATH="${1:-/data3/liutl/HBSR/outputs/hybrid_external_video_prior_lego_lr4_train}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[eval-hr] model path not found: ${MODEL_PATH}"
  exit 1
fi

cd "${HBSR_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python render.py \
  -s "${DATASET_PATH}" \
  -m "${MODEL_PATH}" \
  --iteration -1 \
  --resolution 1 \
  --skip_train \
  --white_background

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python metrics.py \
  -m "${MODEL_PATH}" \
  -r 1

echo "[eval-hr] done: ${MODEL_PATH}"
