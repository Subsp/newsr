#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/data3/liutl/HBSR"
DATASET_PATH="/data3/liutl/nerf_synthetic/lego"
OUTPUT_PATH="/data3/liutl/HBSR/outputs/hybrid_analytic_lego_lr4_train"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "${OUTPUT_PATH}"
cd "${HBSR_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python hybrid_sdfgs/train.py \
  -s "${DATASET_PATH}" \
  -m "${OUTPUT_PATH}" \
  --resolution 4 \
  --eval \
  --white_background \
  --disable_gui \
  --iterations 30000 \
  --hybrid_enable \
  --sdf_mode analytic \
  --hybrid_points_per_iter 4096 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000

echo "[train-lr4-hybrid-analytic] done: ${OUTPUT_PATH}"
