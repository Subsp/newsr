#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/root/autodl-tmp/HBSR"
INPUT_DIR="/root/autodl-tmp/kitchen/images_8"
GT_DIR="/root/autodl-tmp/kitchen/images_2"
PRIOR_DIR="/root/autodl-tmp/priors/kitchen_video_flashvsr_seqmat_x8to2/priors"
OUTPUT_DIR="/root/autodl-tmp/priors/kitchen_video_flashvsr_seqmat_x8to2/analysis"
COLMAP_SPARSE_DIR="/root/autodl-tmp/kitchen/sparse/0"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${HBSR_ROOT}"

python hybrid_sdfgs/tools/analyze_flashvsr_prior_quality.py \
  --prior_dir "${PRIOR_DIR}" \
  --input_dir "${INPUT_DIR}" \
  --gt_dir "${GT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --view_group_mode seqmat_pose_als \
  --colmap_sparse_dir "${COLMAP_SPARSE_DIR}" \
  --view_group_max_len 6 \
  --view_group_min_len 3 \
  --view_group_thresholds "30,50" \
  --view_dir_weight 0.0

echo "[analyze-flashvsr-prior-x8to2] done: ${OUTPUT_DIR}"
