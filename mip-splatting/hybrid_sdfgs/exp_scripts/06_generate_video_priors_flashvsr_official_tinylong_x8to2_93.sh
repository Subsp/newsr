#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/root/autodl-tmp/HBSR"
INPUT_IMAGES="/root/autodl-tmp/kitchen/images_8"
OUTPUT_ROOT="/root/autodl-tmp/priors/kitchen_video_flashvsr_official_tinylong_x8to2_93"
VIDEO_SR_REPO="/root/autodl-tmp/FlashVSR"
FLASHVSR_PYTHON="/root/miniconda3/envs/flashvsr/bin/python"
FLASHVSR_MODEL_DIR="/root/autodl-tmp/hub/FlashVSR-v1.1"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${HBSR_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${FLASHVSR_PYTHON}" hybrid_sdfgs/tools/run_flashvsr_official_baseline.py \
  --repo_root "${VIDEO_SR_REPO}" \
  --model_dir "${FLASHVSR_MODEL_DIR}" \
  --input_dir "${INPUT_IMAGES}" \
  --output_root "${OUTPUT_ROOT}" \
  --model_type "tiny_long" \
  --chunk_len 93

echo "[generate-video-prior-flashvsr-official-tinylong-x8to2-93] done: ${OUTPUT_ROOT}/priors"
