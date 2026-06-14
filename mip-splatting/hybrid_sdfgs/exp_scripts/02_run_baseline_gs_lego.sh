#!/usr/bin/env bash
set -euo pipefail
HBSR_ROOT="/data3/liutl/HBSR"
DATASET_PATH="/data3/liutl/nerf_synthetic/lego"
OUTPUT_PATH="/data3/liutl/HBSR/outputs/baseline_gs_lego"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${HBSR_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python train.py \
  -s "${DATASET_PATH}" \
  -m "${OUTPUT_PATH}" \
  --eval \
  --white_background \
  --iterations 30000 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000
echo "[baseline] done: ${OUTPUT_PATH}"
