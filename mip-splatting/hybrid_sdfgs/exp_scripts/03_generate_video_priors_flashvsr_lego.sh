#!/usr/bin/env bash
set -euo pipefail

HBSR_ROOT="/root/autodl-tmp/HBSR"
INPUT_IMAGES="/root/autodl-tmp/kitchen/images_4"
OUTPUT_ROOT="/root/autodl-tmp/priors/kitchen_video_flashvsr"
VIDEO_SR_REPO="/root/autodl-tmp/FlashVSR"
FLASHVSR_PYTHON="/root/miniconda3/envs/flashvsr/bin/python"
FLASHVSR_MODEL_DIR="/root/autodl-tmp/hub/FlashVSR-v1.1"
FLASHVSR_CACHE_ROOT="${OUTPUT_ROOT}/cache"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
INPUT_DOWNSAMPLE=1

if [[ ! -d "${VIDEO_SR_REPO}" ]]; then
  echo "[video-prior-flashvsr] repo not found: ${VIDEO_SR_REPO}"
  exit 1
fi

if [[ ! -x "${FLASHVSR_PYTHON}" ]]; then
  echo "[video-prior-flashvsr] python not found or not executable: ${FLASHVSR_PYTHON}"
  exit 1
fi

if [[ ! -d "${FLASHVSR_MODEL_DIR}" ]]; then
  echo "[video-prior-flashvsr] model folder missing: ${FLASHVSR_MODEL_DIR}"
  exit 1
fi

cd "${HBSR_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python hybrid_sdfgs/tools/generate_video_sr_priors.py \
  --model flashvsr \
  --repo_root "${VIDEO_SR_REPO}" \
  --python_exe "${FLASHVSR_PYTHON}" \
  --input_dir "${INPUT_IMAGES}" \
  --output_root "${OUTPUT_ROOT}" \
  --input_downsample "${INPUT_DOWNSAMPLE}" \
  --flashvsr_variant "v1.1" \
  --flashvsr_model_type "tiny" \
  --flashvsr_model_dir "${FLASHVSR_MODEL_DIR}" \
  --flashvsr_fps 24 \
  --flashvsr_cache_root "${FLASHVSR_CACHE_ROOT}" \
  --flashvsr_keep_runtime

echo "[generate-video-prior-flashvsr] done: ${OUTPUT_ROOT}/priors"
