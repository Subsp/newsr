#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="${HBSR_ROOT:-/root/autodl-tmp/HBSR}"
ROOT_DIR="${ROOT_DIR:-/root/autodl-tmp/priors/kitchen_video_uav_x8bicubic_to_x4_then_x1_chunked}"
SR_DIR="${SR_DIR:-${ROOT_DIR}/priors}"
INPUT_DIR="${INPUT_DIR:-${ROOT_DIR}/input_bicubic_images_4}"
GT_DIR="${GT_DIR:-${ROOT_DIR}/ref_images}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/highfreq_analysis}"
PYTHON_EXE="${PYTHON_EXE:-python}"

echo "[hf-analysis-x8b4to1] SR_DIR=${SR_DIR}"
echo "[hf-analysis-x8b4to1] INPUT_DIR=${INPUT_DIR}"
echo "[hf-analysis-x8b4to1] GT_DIR=${GT_DIR}"
echo "[hf-analysis-x8b4to1] OUTPUT_DIR=${OUTPUT_DIR}"

"${PYTHON_EXE}" "${HBSR_ROOT}/hybrid_sdfgs/tools/analyze_sr_highfreq_regions.py" \
  --sr_dir "${SR_DIR}" \
  --input_dir "${INPUT_DIR}" \
  --gt_dir "${GT_DIR}" \
  --output_dir "${OUTPUT_DIR}"

echo "[hf-analysis-x8b4to1] done: ${OUTPUT_DIR}"
