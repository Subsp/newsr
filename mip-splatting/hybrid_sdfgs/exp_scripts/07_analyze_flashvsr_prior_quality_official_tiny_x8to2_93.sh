#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/root/autodl-tmp/HBSR"
INPUT_DIR="/root/autodl-tmp/kitchen/images_8"
GT_DIR="/root/autodl-tmp/kitchen/images_2"
PRIOR_DIR="/root/autodl-tmp/priors/kitchen_video_flashvsr_official_tiny_x8to2_93/priors"
OUTPUT_DIR="/root/autodl-tmp/priors/kitchen_video_flashvsr_official_tiny_x8to2_93/analysis"

cd "${HBSR_ROOT}"

python hybrid_sdfgs/tools/analyze_flashvsr_prior_quality.py \
  --prior_dir "${PRIOR_DIR}" \
  --input_dir "${INPUT_DIR}" \
  --gt_dir "${GT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --view_group_mode none

echo "[analyze-flashvsr-prior-official-tiny-x8to2-93] done: ${OUTPUT_DIR}"
