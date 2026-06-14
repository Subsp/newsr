#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/data3/liutl/HBSR"
DATASET_PATH="/data3/liutl/nerf_synthetic/lego"
OUTPUT_PATH="/data3/liutl/HBSR/outputs/hybrid_external_video_prior_lego_lr4_train"
EXTERNAL_PRIOR_ROOT="/data3/liutl/HBSR/priors/lego_video_realbasic"
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
  --external_prior_root "${EXTERNAL_PRIOR_ROOT}" \
  --external_prior_subdir "priors" \
  --prior_l1_weight 0.02 \
  --prior_hf_weight 0.10 \
  --prior_delta_clip 0.08 \
  --prior_consistency_threshold 0.08 \
  --prior_min_valid_ratio 0.80 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000

echo "[train-lr4-external-video-prior] done: ${OUTPUT_PATH}"
